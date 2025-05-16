# Robustness Impacts of Coreset Selection

This repository contains code for studying the robustness impacts of coreset selection methods on various datasets.

## Environment Setup

1. Create a conda environment:
```bash
conda create -n bias-select python=3.8
conda activate bias-select
```

2. Install PyTorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install the deepcore package and its dependencies:
```bash
# Install the package in development mode (make sure you're in the repository root directory)
pip install -e .

# Or install dependencies first, then the package
pip install -r requirements.txt
pip install -e .
```

The `-e` flag installs the package in "editable" mode, which means you can modify the code without reinstalling.

## Dataset Preparation

The codebase supports the following datasets:

1. **Cmnist** - Colored MNIST with spurious correlations
2. **waterbirds** - Bird species classification with background bias
3. **Urbancars_cooccur** - Car classification with co-occurrence bias
4. **Urbancars_bg** - Car classification with background bias
5. **Urbancars_both** - Car classification with combined biases
6. **Nico_95_spurious** - Natural Image Classification with Context
7. **MultiNLI** - Natural Language Inference dataset
8. **Metashift** - Dataset for studying distribution shifts
9. **Civilcomments** - Toxic comment classification dataset
10. **CelebAhair** - CelebA dataset with hair color attributes

### Dataset Download Instructions

1. Create a data directory:
```bash
mkdir -p data
```

2. Download and prepare each dataset:

#### 1. Cmnist
The CMNIST dataset is hosted on Hugging Face. To download it:

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
```bash
python scripts/download_cmnist.py
```

The dataset will be downloaded to `data/cmnist/` with the following structure:
```
data/cmnist/
├── test/
│   ├── 0/
│   ├── 1/
│   └── ... (classes 2-9)
└── 5pct/
    ├── align/
    │   ├── 0/
    │   └── 1/
    ├── valid/
    │   ├── 0/
    │   └── 1/
    └── conflict/
        ├── 0/
        └── 1/
```

#### 2. waterbirds
Follow - https://github.com/kohpangwei/group_DRO/tree/master to generate the waterbirds dataset

#### 3. Urbancars variants
Follow - https://github.com/facebookresearch/Whac-A-Mole/tree/main to create the dataset
Urbancars_cooccur, Urbancars_bg, Urbancars_both 

#### 4. Metashift
Follow https://github.com/YyzHarry/SubpopBench/tree/main to download Metashift dataset and generate the metadata.

#### 5. Civilcomments
Follow https://github.com/izmailovpavel/spurious_feature_learning/tree/main to setup the dataset. You will need to install Wilds package for this.

#### 6. Nico_95_spurious
We follow https://github.com/yvsriram/FACTS to set up the dataset. Download the NICO++ dataset as specified by them into './Data/NICO'
data_path= './Data/NICO'

#### 7. MultiNLI
Follow https://github.com/izmailovpavel/spurious_feature_learning/tree/main to setup the dataset

#### 8. CelebAhair
Follow https://github.com/kohpangwei/group_DRO#celeba to download the dataset. Then, to use the hair color as the target attribute, we have provided the metadata file - 

### Label Preparation

To prepare the labels for any dataset:

```bash
python scripts/save_dataset_labels.py <dataset_name>
```

For example:
```bash
# For CMNIST
python scripts/save_dataset_labels.py Cmnist

## Running Experiments

### Sample Characterization Scores

To compute sample characterization scores for a dataset, run the corresponding script in the `scripts` directory:

```bash
# For CMNIST dataset
python scripts/run_cmnist.py

# For Waterbirds dataset
python scripts/run_waterbirds.py

# For CelebA dataset
python scripts/run_celeba.py
```

These scripts will:
1. Train models on the full dataset
2. Compute various sample characterization scores (EL2N, Forgetting, Uncertainty, etc.)
3. Save the scores in the `results` directory

### Training Downstream Models

After computing the sample characterization scores, you can train downstream models on the selected coresets using the corresponding training scripts:

```bash
# For CMNIST dataset
python scripts/run_cmnist_train.py

# For Waterbirds dataset
python scripts/run_waterbirds_train.py

# For CelebA dataset
python scripts/run_celeba_train.py
```

These training scripts will:
1. Load the pre-computed sample characterization scores
2. Select coresets based on the specified selection method
3. Train models on the selected coresets
4. Save the trained models and results

### Example Workflow

Here's a complete example for the CMNIST dataset:

```bash
# Step 1: Compute sample characterization scores
python scripts/run_cmnist.py

# Step 2: Train models on selected coresets
python scripts/run_cmnist_train.py
```

The results will be saved in the `results` directory with appropriate naming conventions for each dataset and selection method.






