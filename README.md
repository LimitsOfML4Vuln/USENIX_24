# Code, Data and Results for "Uncovering the Limits of Large Language Models for Automatic Vulnerability Detection"

## Structure

Below is an annotated map of the directory structure of this repository.

```
.
├── scripts................................ Scripts to exactly reproduce all experiments presented in our paper.
│   └── <dataset>.......................... One directory for each dataset (CodeXGLUE, VulDeePecker, VulnPatchPairs).
│       └── <technique>.................... One directory for each ML technique (UniXcoder, CodeBERT, GraphCodeBERT, VulBERTa, CoTexT, PLBart).
│           └── run.py..................... Compute experimental results for selected ML technique and dataset.
│
├── datasets............................... All datasets that we use in the experiments for the paper + our own created dataset VulnPatchPairs.
│   └── README.md.......................... Instructions for downloading all datasets used in this repository.
│
├── models................................. All pretrained models that are not downloaded in the training scripts.
│   └── README.md.......................... Instructions for downloading all models used in this repository.
│
├── plots.................................. Generates all plots shown in the paper.
│   ├── generate_plots.py.................. Script that generates all plots from the experimental results.
│   └── plots.............................. The produced plots and tables that can be found in the paper.
│
├── install_requirements.sh................ Script to install Python environment and required packages.
├── requirements.txt....................... All Python packages that you need to run the experiments.
│
├── run_experiments.sh..................... Script to reproduce all experiments presented in our paper.
│
└── README.md
```

## Setup

### Step 1: Install Anaconda

Anaconda is an open-source package and environment management tool for Python. Instructions for Installation can be found [here](https://www.anaconda.com/products/distribution).

### Step 2: Install Requirements

We assume that you have Anaconda installed.

Running the following script from the root directory of this repository creates a virtual environment in Anaconda, and installs the required Python packages.

```
bash install_requirements.sh
```

Activate the environment with the following command.

```
conda activate LimitsOfMl4Vuln
```

### Step 3: Download the required datasets

Go to [datasets/README.md](https://github.com/LimitsOfML4Vuln/USENIX_24/blob/main/datasets/README.md) and follow the instructions to download all datasets needed to run our experiments.

### Step 4: Download the required models

Go to [models/README.md](https://github.com/LimitsOfML4Vuln/USENIX_24/blob/main/models/README.md) and follow the instructions to download all models needed to run our experiments.

### Step 5: Ready to go

Run

```
bash run_experiments.sh
```

to reproduce all experimental results presented in our paper. The script also serves as an entry point into the scripts for the different experiments.

### Vulnerability Detection Datasets

We use two external Vulnerability Detection datasets. We provide the full datasets as part of this repository (see [datasets/README.md](https://github.com/LimitsOfML4Vuln/USENIX_24/blob/main/datasets/README.md)). These are the references for the original datasets.

#### CodeXGLUE/Devign:

```
@inbook{Devign,
author = {Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
title = {Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks},
year = {2019},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
booktitle = {Proceedings of the 33rd International Conference on Neural Information Processing Systems},
articleno = {915},
numpages = {11}
}
@inproceedings{CODEXGLUE,
 author = {Lu, Shuai and Guo, Daya and Ren, Shuo and Huang, Junjie and Svyatkovskiy, Alexey and Blanco, Ambrosio and Clement, Colin and Drain, Dawn and Jiang, Daxin and Tang, Duyu and Li, Ge and Zhou, Lidong and Shou, Linjun and Zhou, Long and Tufano, Michele and GONG, MING and Zhou, Ming and Duan, Nan and Sundaresan, Neel and Deng, Shao Kun and Fu, Shengyu and LIU, Shujie},
 booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
 editor = {J. Vanschoren and S. Yeung},
 pages = {},
 title = {CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation},
 url = {https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c16a5320fa475530d9583c34fd356ef5-Paper-round1.pdf},
 volume = {1},
 year = {2021}
}
```

GitHub: [https://github.com/epicosy/devign](https://github.com/epicosy/devign)

GitHub: [https://github.com/microsoft/CodeXGLUE](https://github.com/microsoft/CodeXGLUE)

#### VulDeePecker:

```
@inproceedings{VulDeePecker,
	doi = {10.14722/ndss.2018.23158},

	url = {https://doi.org/10.14722%2Fndss.2018.23158},

	year = 2018,
	publisher = {Internet Society},

	author = {Zhen Li and Deqing Zou and Shouhuai Xu and Xinyu Ou and Hai Jin and Sujuan Wang and Zhijun Deng and Yuyi Zhong},

	title = {{VulDeePecker}: A Deep Learning-Based System for Vulnerability Detection},

	booktitle = {Proceedings 2018 Network and Distributed System Security Symposium}
}
```

GitHub: [https://github.com/CGCL-codes/VulDeePecker](https://github.com/CGCL-codes/VulDeePecker)
