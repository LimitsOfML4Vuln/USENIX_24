import os
import sys
import random
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import List
import argparse
import wandb

from clang import *
from clang import cindex

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from transformers import RobertaForSequenceClassification, AutoTokenizer
from transformers import get_constant_schedule
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
print(device)

# deterministic settings for exact reproducibility
seedlist = [42, 834, 692, 489, 901, 408, 819, 808, 531, 166]

NUM_EPOCHS = 20
TRAINING_SET_FRAC = 1.0
RANDOM_SEED = 408
LEARNING_RATE = 3e-6
BATCH_SIZE = 8
SCHEDULER_NAME = "constant"
NUM_SCHEDULER_RESTARTS = 0
SCHEDULER_WARMUP_PERCENTAGE = 0

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['WANDB_MODE'] = 'offline'



pathprefix = ""

my_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

vpp_train = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Train-Cleaned.json')
vpp_valid = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Valid-Cleaned.json')
vpp_test = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Test-Cleaned.json')

train_index=set()
valid_index=set()
test_index=set()

with open('./' + pathprefix + 'datasets/CodeXGLUE/train-IDs.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))

with open('./' + pathprefix + 'datasets/CodeXGLUE/valid-IDs.txt') as f:
    for line in f:
        line=line.strip()
        valid_index.add(int(line))

with open('./' + pathprefix + 'datasets/CodeXGLUE/test-IDs.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))

codexglue = pd.read_json('./' + pathprefix + 'datasets/CodeXGLUE/CodeXGLUE.json')

codexglue_train = codexglue.iloc[list(train_index)]
codexglue_valid = codexglue.iloc[list(valid_index)]
codexglue_test = codexglue.iloc[list(test_index)]

def encodeDataframe(df):
    
    encodings = my_tokenizer.batch_encode_plus(df.func.tolist(), max_length=512, padding='max_length', truncation=True)
    
    return encodings, df.target.tolist()

class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings['input_ids']) == len(self.encodings['attention_mask']) ==  len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def separate_vulns_and_patches(df):
    
    patches = df.drop('vulnerable_func', axis=1)
    patches = patches.rename(columns={'patched_func': 'func'})
    patches["target"] = 0
    
    vulns = df.drop('patched_func', axis=1)
    vulns = vulns.rename(columns={'vulnerable_func': 'func'})
    vulns["target"] = 1
    
    fused_df = patches.append(vulns)
    fused_df = fused_df.reset_index()
    
    return fused_df

def get_scheduler(optim, name, num_restarts, num_training_steps, warmup_percentage):
    
    if name == "linear":
        num_warmup_steps = int(num_training_steps * warmup_percentage)
        return get_linear_schedule_with_warmup(optimizer=optim, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    elif name == "cosine_with_restarts":
        num_warmup_steps = int(num_training_steps * warmup_percentage / (num_restarts + 1))
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optim, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps, num_cycles=num_restarts)
    elif name == "constant":
        return get_constant_schedule(optimizer=optim)
    else:
        raise Exception("schedule not available")

vpp_train = vpp_train.sample(frac=TRAINING_SET_FRAC)
vpp_test = vpp_test.sample(frac=1.0)

vpp_train = separate_vulns_and_patches(vpp_train)
vpp_valid = separate_vulns_and_patches(vpp_valid)
vpp_test = separate_vulns_and_patches(vpp_test)



vpp_train_encodings, vpp_train_labels = encodeDataframe(vpp_train)
vpp_valid_encodings, vpp_valid_labels = encodeDataframe(vpp_valid)
vpp_test_encodings, vpp_test_labels = encodeDataframe(vpp_test)
codexglue_test_encodings, codexglue_test_labels = encodeDataframe(codexglue_test)

vpp_train_dataset = MyCustomDataset(vpp_train_encodings, vpp_train_labels)
vpp_valid_dataset = MyCustomDataset(vpp_valid_encodings, vpp_valid_labels)
vpp_test_dataset = MyCustomDataset(vpp_test_encodings, vpp_test_labels)
codexglue_test_dataset = MyCustomDataset(codexglue_test_encodings, codexglue_test_labels)

vpp_train_dataloader = DataLoader(vpp_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
vpp_valid_dataloader = DataLoader(vpp_valid_dataset, shuffle=False, batch_size=BATCH_SIZE)
vpp_test_dataloader = DataLoader(vpp_test_dataset, shuffle=False, batch_size=BATCH_SIZE)
codexglue_test_dataloader = DataLoader(codexglue_test_dataset, shuffle=False, batch_size=BATCH_SIZE)

print(len(vpp_train_dataloader))
print(len(vpp_valid_dataloader))
print(len(vpp_test_dataloader))
print(len(codexglue_test_dataloader))

num_training_steps = NUM_EPOCHS * len(vpp_train_dataloader)
num_validation_steps = NUM_EPOCHS * len(vpp_valid_dataloader)
num_testing_steps = NUM_EPOCHS * len(vpp_test_dataloader) + NUM_EPOCHS * len(codexglue_test_dataloader)

model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")
model.to(device)
    
criterion = torch.nn.CrossEntropyLoss() 
criterion.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = get_scheduler(optim=optimizer, name=SCHEDULER_NAME, num_restarts=NUM_SCHEDULER_RESTARTS, num_training_steps=num_training_steps, warmup_percentage=SCHEDULER_WARMUP_PERCENTAGE)

results = dict()

progress_bar = tqdm(range(num_training_steps + num_testing_steps))

for epoch in range(NUM_EPOCHS):
    
    model.train()
    
    losses = []
    predictions = []
    labels = []
    
    for batch in vpp_train_dataloader:
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])
    
        loss.backward()
        
        logits = outputs.logits
        
        losses += [loss.item()]
        
        predictions += torch.argmax(logits, dim=-1).tolist()
        labels += batch["labels"].tolist()

        optimizer.step()
        lr_scheduler.step()
        
        progress_bar.update(1)
    
    model.eval()
    
    predictions = []
    labels = []

    for batch in vpp_test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
            
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        
        predictions += torch.argmax(logits, dim=-1).tolist()
        labels += batch["labels"].tolist()
        
        progress_bar.update(1)
    
    test_accuracy = accuracy_score(labels, predictions)
    _, _, test_f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)
    
    predictions_codexglue = []
    labels_codexglue = []

    for batch in codexglue_test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
            
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        
        predictions_codexglue += torch.argmax(logits, dim=-1).tolist()
        labels_codexglue += batch["labels"].tolist()
        
        progress_bar.update(1)
    
    codexglue_test_accuracy = accuracy_score(labels_codexglue, predictions_codexglue)
    _, _, codexglue_test_f1, _ = precision_recall_fscore_support(labels_codexglue, predictions_codexglue, average='binary', zero_division=1)
    
    results[epoch] = dict()
    results[epoch]["epoch"] = epoch
    results[epoch]["train/loss"] = np.array(losses).mean()
    results[epoch]["vpp/test/accuracy"] = test_accuracy
    results[epoch]["vpp/test/f1"] = test_f1
    results[epoch]["codexglue/test/accuracy"] = codexglue_test_accuracy
    results[epoch]["codexglue/test/f1"] = codexglue_test_f1
        

save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1]
    
with open('./' + pathprefix + 'results/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(results, f)

