import os
import sys
import random
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import List
import argparse

from clang import *
from clang import cindex

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from transformations import *

STR_TO_TRAFO = {
    "no_transformation" : no_transformation,
    "tf_1" : tf_1,
    "tf_2" : tf_2,
    "tf_3" : tf_3,
    "tf_4" : tf_4,
    "tf_5" : tf_5,
    "tf_6" : tf_6,
    "tf_7" : tf_7,
    "tf_8" : tf_8,
    "tf_9" : tf_9,
    "tf_10" : tf_10,
    "tf_11" : tf_11,
    "tf_12" : tf_12,
    "tf_13" : tf_13,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--trafo", required=True)
parser.add_argument("--random", default=False, required=False)
args = parser.parse_args()
trafo_from_args = STR_TO_TRAFO[args.trafo]
random_guessing = args.random == "True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
print(device)

# deterministic settings for exact reproducibility
seedlist = [42, 834, 692, 489, 901, 408, 819, 808, 531, 166]
fixed_seed = 489 #random.choice(seedlist)

os.environ['PYTHONHASHSEED'] = str(fixed_seed)
torch.manual_seed(fixed_seed)
torch.cuda.manual_seed(fixed_seed)
torch.cuda.manual_seed_all(fixed_seed)
np.random.seed(fixed_seed)
random.seed(fixed_seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



pathprefix = ""

tokenizer = T5Tokenizer.from_pretrained('razent/cotext-1-ccg')

ids_neg = tokenizer.encode('negative </s>')
ids_pos = tokenizer.encode('positive </s>')


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


mydata = pd.read_json('./' + pathprefix + 'datasets/CodeXGLUE/CodeXGLUE.json')
developer_commit_data_test = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Test.json')

m1=mydata.iloc[list(train_index)]
m2=mydata.iloc[list(valid_index)]
m3=mydata.iloc[list(test_index)]
m4 = developer_commit_data_test


def prependTaskDesc(value):
    
    return "defect_detection: " + value

def toStrTargets(value):
    
    assert value == "0" or value == "1"
    
    if value == "0":
        return "negative </s>"
    return "positive </s>"

def cleanup(code):

    code = code.replace("\n", " NEW_LINE ")
    code = code.replace("}", " CLOSE_CURLY_TOKEN ")
    code = code.replace("{", " OPEN_CURLY_TOKEN ")
    code = code.replace("]", " CLOSE_SQUARE_TOKEN ")
    code = code.replace("[", " OPEN_SQUARE_TOKEN ")
    code = code.replace("<", " SMALLER_TOKEN ")
    code = code.replace(">", " GREATER_TOKEN ")
    code = code.replace("#", " SHARP_TOKEN ")
    
    return code

def m4_cleanup(code):
    
    if code[-1] == "\n":
        code = code[:-1]
        
    code = code.replace("\n", " NEW_LINE ")
    code = code.replace("NEW_LINE  NEW_LINE", "NEW_LINE")
    code = code.replace("NEW_LINE  NEW_LINE", "NEW_LINE")
    code = code.replace("}", " CLOSE_CURLY_TOKEN ")
    code = code.replace("{", " OPEN_CURLY_TOKEN ")
    code = code.replace("]", " CLOSE_SQUARE_TOKEN ")
    code = code.replace("[", " OPEN_SQUARE_TOKEN ")
    code = code.replace("<", " SMALLER_TOKEN ")
    code = code.replace(">", " GREATER_TOKEN ")
    code = code.replace("#", " SHARP_TOKEN ")
    
    return code
    

def prepareDataframe(df):
    
    df['target'] = df['target'].map(str)
    df['target'] = df['target'].map(toStrTargets)
    df['func'] = df['func'].map(prependTaskDesc)
    
    return df

def encodeDataframe(df):
    
    encodings = []
    labels = []

    for row in df.itertuples(index=True, name='Pandas'):
    
        enc = tokenizer.batch_encode_plus([row.func], max_length=1024, padding='max_length', truncation=True, return_tensors="pt")
        lab = tokenizer.batch_encode_plus([row.target], max_length=2, padding='max_length', truncation=True, return_tensors="pt")
        
        encodings.append(enc)
        labels.append(lab)
        
    
    return encodings, labels

class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        source_ids = self.encodings[index]["input_ids"].squeeze()
        target_ids = self.labels[index]["input_ids"].squeeze()

        src_mask    = self.encodings[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.labels[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.labels)

def apply_transformation(df, all_trafos, trafo_to_apply, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply_in_random_selection = None):
    
    if trafo_to_apply.__name__ == "tf_11":
        df.func = df.func.apply(trafo_to_apply, args=(all_trafos, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply_in_random_selection))
    elif trafo_to_apply.__name__ == "tf_10":
        df.func = df.func.apply(trafo_to_apply, args=(training_set_sample_neg,))
    elif trafo_to_apply.__name__ == "tf_13":
        df.func = df.func.apply(trafo_to_apply, args=(training_set_sample_pos,))
    else:
        df.func = df.func.apply(trafo_to_apply)
        
    return df

    
TF_TO_APPLY_ON_TRAIN = trafo_from_args

semantic_preserving_transformations = [no_transformation, tf_1, tf_2, tf_3, tf_4, tf_5, tf_6, tf_7, tf_8, tf_9, tf_12, tf_11, tf_10, tf_13]

m1_not_vulnerable = m1[m1["target"] == 0]
training_set_sample_neg = m1_not_vulnerable.iloc[0]["func"]

m1_vulnerable = m1[m1["target"] == 1]
training_set_sample_pos = m1_vulnerable.iloc[0]["func"]
    
m1 = apply_transformation(m1, semantic_preserving_transformations, TF_TO_APPLY_ON_TRAIN, training_set_sample_neg, training_set_sample_pos, None)

m4_changed = m4.drop('func', axis=1)
m4_changed = m4_changed.rename(columns={'changed_func': 'func'})
m4_changed = m4_changed[m4_changed["target"] == 1]
m4_changed = m4_changed.assign(target=0)
m4_changed = m4_changed.drop('new_path', axis=1)
m4_changed = m4_changed.drop('old_path', axis=1)
m4_changed = m4_changed.drop('new_commit', axis=1)
m4_changed = m4_changed.drop('old_commit', axis=1)

m4 = m4.drop('changed_func', axis=1)
m4 = m4.drop('new_path', axis=1)
m4 = m4.drop('old_path', axis=1)
m4 = m4.drop('new_commit', axis=1)
m4 = m4.drop('old_commit', axis=1)
m4 = m4[m4["target"] == 1]

print(m4_changed.count())
print(m4.count())

print(m4_changed.head())
print(m4.head())

m1.func = m1.func.apply(cleanup)
m3.func = m3.func.apply(cleanup)
m4_changed.func = m4_changed.func.apply(m4_cleanup)
m4.func = m4.func.apply(m4_cleanup)

m1 = prepareDataframe(m1)
m3 = prepareDataframe(m3)
m4_changed = prepareDataframe(m4_changed)
m4 = prepareDataframe(m4)

train_encodings, train_labels = encodeDataframe(m1)
test_encodings, test_labels = encodeDataframe(m3)
fixed_encodings, fixed_labels = encodeDataframe(m4_changed)
not_fixed_encodings, not_fixed_labels = encodeDataframe(m4)

train_dataset = MyCustomDataset(train_encodings, train_labels)
test_dataset = MyCustomDataset(test_encodings, test_labels)
fixed_dataset = MyCustomDataset(fixed_encodings, fixed_labels)
not_fixed_dataset = MyCustomDataset(not_fixed_encodings, not_fixed_labels)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)
fixed_dataloader = DataLoader(fixed_dataset, shuffle=False, batch_size=8)
not_fixed_dataloader = DataLoader(not_fixed_dataset, shuffle=False, batch_size=8)

print(len(train_dataloader))
print(len(test_dataloader))
print(len(fixed_dataloader))
print(len(not_fixed_dataloader))

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
num_testing_steps = num_epochs * len(test_dataloader)
num_fixed_steps = num_epochs * len(fixed_dataloader)
num_not_fixed_steps = num_epochs * len(not_fixed_dataloader)

model = T5ForConditionalGeneration.from_pretrained('razent/cotext-1-ccg')
model.to(device)

optimizer = AdamW(model.parameters(), lr=0.00003)

results = dict()

progress_bar = tqdm(range(num_training_steps + num_testing_steps * len(semantic_preserving_transformations) + num_fixed_steps + num_not_fixed_steps))

for epoch in range(num_epochs):
    
    model.train()
    
    losses = []
    predictions = []
    labels = []
    
    for batch in train_dataloader:
        
        batch = {k: v for k, v in batch.items()}
        
        if not random_guessing:
            optimizer.zero_grad()
            
            target_ids = batch["target_ids"]
            target_ids[target_ids == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids=batch["source_ids"].to(device), attention_mask=batch["source_mask"].to(device), labels=target_ids.to(device))
            
            loss = outputs.loss
        
            loss.backward()
            
            losses += [loss.item()]

            optimizer.step()
        
        progress_bar.update(1)
    
    model.eval()

    for k, TF_TO_APPLY_ON_TEST in enumerate(semantic_preserving_transformations):
        
        m3=mydata.iloc[list(test_index)]
        
        m3 = apply_transformation(m3, semantic_preserving_transformations, TF_TO_APPLY_ON_TEST, training_set_sample_neg, training_set_sample_pos, TF_TO_APPLY_ON_TEST)

        m3.func = m3.func.apply(cleanup)

        m3 = prepareDataframe(m3)

        test_encodings, test_labels = encodeDataframe(m3)

        test_dataset = MyCustomDataset(test_encodings, test_labels)

        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)
        
        predictions = []
        labels = []
    
        for batch in test_dataloader:
            batch = {k: v for k, v in batch.items()}
            
            if random_guessing:
                
                decoded_labels = tokenizer.batch_decode(batch["target_ids"].to(device), skip_special_tokens=True)
                
                labels_as_numbers = [1 if d == "positive" else 0 for d in decoded_labels]
                
                predictions += len(labels_as_numbers) * [random.randint(0, 1)]
                labels += labels_as_numbers
            else:
                with torch.no_grad():
                    outputs = model.generate(input_ids=batch["source_ids"].to(device), attention_mask=batch["source_mask"].to(device), max_length=5)
            
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(batch["target_ids"].to(device), skip_special_tokens=True)
                
                labels_as_numbers = [1 if d == "positive" else 0 for d in decoded_labels]
                
                predictions_as_numbers = []
                
                for i, output in enumerate(decoded_outputs):
                    if output == "positive":
                        predictions_as_numbers.append(1)
                    elif output == "negative":
                        predictions_as_numbers.append(0)
                    else:
                        predictions_as_numbers.append(random.randint(0,1))
                
                labels += labels_as_numbers
                predictions += predictions_as_numbers
            
            progress_bar.update(1)
        
        test_accuracy = accuracy_score(labels, predictions)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(labels, predictions).ravel()
        
        if epoch == 0:
            results[TF_TO_APPLY_ON_TEST.__name__] = dict()
        results[TF_TO_APPLY_ON_TEST.__name__][epoch] = dict()
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["epoch"] = epoch
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["train/loss"] = np.array(losses).mean()
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/accuracy"] = test_accuracy
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/precision"] = test_precision
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/recall"] = test_recall
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/f1"] = test_f1
        
    predictions_fixed = []
    labels_fixed = []

    for batch in fixed_dataloader:
        batch = {k: v for k, v in batch.items()}
        
        if random_guessing:
            decoded_labels = tokenizer.batch_decode(batch["target_ids"].to(device), skip_special_tokens=True)
            
            labels_as_numbers = [1 if d == "positive" else 0 for d in decoded_labels]
            
            predictions_fixed += len(labels_as_numbers) * [random.randint(0, 1)]
            labels_fixed += labels_as_numbers
        else:
            with torch.no_grad():
                outputs = model.generate(input_ids=batch["source_ids"].to(device), attention_mask=batch["source_mask"].to(device), max_length=5)
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["target_ids"].to(device), skip_special_tokens=True)
            
            labels_as_numbers = [1 if d == "positive" else 0 for d in decoded_labels]
            
            predictions_as_numbers = []
            
            for i, output in enumerate(decoded_outputs):
                if output == "positive":
                    predictions_as_numbers.append(1)
                elif output == "negative":
                    predictions_as_numbers.append(0)
                else:
                    predictions_as_numbers.append(random.randint(0,1))
            
            labels_fixed += labels_as_numbers
            predictions_fixed += predictions_as_numbers
        
        progress_bar.update(1)
        
    assert 0 == np.sum(np.array(labels_fixed))
    
    predictions_not_fixed = []
    labels_not_fixed = []

    for batch in not_fixed_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if random_guessing:
            decoded_labels = tokenizer.batch_decode(batch["target_ids"].to(device), skip_special_tokens=True)
            
            labels_as_numbers = [1 if d == "positive" else 0 for d in decoded_labels]
            
            predictions_not_fixed += len(labels_as_numbers) * [random.randint(0, 1)]
            labels_not_fixed += labels_as_numbers.tolist()
        else:
            batch = {k: v for k, v in batch.items()}
        
            with torch.no_grad():
                outputs = model.generate(input_ids=batch["source_ids"].to(device), attention_mask=batch["source_mask"].to(device), max_length=5)
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["target_ids"].to(device), skip_special_tokens=True)
            
            labels_as_numbers = [1 if d == "positive" else 0 for d in decoded_labels]
            
            predictions_as_numbers = []
            
            for i, output in enumerate(decoded_outputs):
                if output == "positive":
                    predictions_as_numbers.append(1)
                elif output == "negative":
                    predictions_as_numbers.append(0)
                else:
                    predictions_as_numbers.append(random.randint(0,1))
            
            labels_not_fixed += labels_as_numbers
            predictions_not_fixed += predictions_as_numbers
        
        progress_bar.update(1)
        
    assert len(labels_not_fixed) == np.sum(np.array(labels_not_fixed))
    
    predictions = predictions_fixed + predictions_not_fixed
    labels = labels_fixed + labels_not_fixed
    
    test_accuracy = accuracy_score(labels, predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(labels, predictions).ravel()
    
    switch = np.sum(np.array(predictions_fixed) != np.array(predictions_not_fixed)) / len(predictions_not_fixed)
    
    if epoch == 0:
        results["fixed_nonfixed"] = dict()
    results["fixed_nonfixed"][epoch] = dict()
    results["fixed_nonfixed"][epoch]["epoch"] = epoch
    results["fixed_nonfixed"][epoch]["train/loss"] = np.array(losses).mean()
    results["fixed_nonfixed"][epoch]["test/accuracy"] = test_accuracy
    results["fixed_nonfixed"][epoch]["test/precision"] = test_precision
    results["fixed_nonfixed"][epoch]["test/recall"] = test_recall
    results["fixed_nonfixed"][epoch]["test/f1"] = test_f1
    results["fixed_nonfixed"][epoch]["test/switch"] = switch
        

if random_guessing:
    save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] + "-RG"
else:
    save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] + "-" + TF_TO_APPLY_ON_TRAIN.__name__
    
with open('./' + pathprefix + 'results/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(results, f)

