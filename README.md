# Bias-Select-DeepCore

This repository contains code for studying bias in coreset selection methods for deep learning. The codebase is built on top of DeepCore, a comprehensive library for coreset selection in deep learning.

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Conda or virtualenv

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bias-select-deepcore.git
cd bias-select-deepcore
```

2. Create and activate a conda environment:
```bash
conda create -n bias-select python=3.8
conda activate bias-select
```

3. Install PyTorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The codebase supports the following datasets:

1. **CMNIST** - Colored MNIST with spurious correlations
2. **Waterbirds** - Bird species classification with background bias
3. **UrbanCars** - Car classification in urban environments
4. **MetaShift** - Dataset for studying distribution shifts
5. **CivilComments** - Toxic comment classification dataset
6. **NICO-spurious** - Natural Image Classification with Context
7. **MultiNLI** - Natural Language Inference dataset
8. **CelebAHair** - CelebA dataset with hair color attributes

### Dataset Download Instructions

1. Create a data directory:
```bash
mkdir -p data
```

2. Download and prepare each dataset:

#### 1. CMNIST
Generate the data - 
```bash
# The dataset will be automatically generated when first used
# No manual download required
```

#### 2. Waterbirds
Follow - https://github.com/kohpangwei/group_DRO/tree/master to generate the waterbirds dataset
data_path = './Data/waterbird/waterbird_complete95_forest2water2'

#### 3. UrbanCars
follow - https://github.com/facebookresearch/Whac-A-Mole/tree/main to create the dataset
data_path = './Data/'
for all Urbancars_cooccur, Urbancars_bg, Urbancars_both its the same data_path

#### 4. MetaShift
Follow https://github.com/YyzHarry/SubpopBench/tree/main to download Metashift dataset and generate the metadata.
data_path = './Data/'

#### 5. CivilComments
Follow https://github.com/izmailovpavel/spurious_feature_learning/tree/main to setup the dataset. You will need to install Wilds package for this.

#### 6. NICO-spurious
We follow https://github.com/yvsriram/FACTS to set up the dataset. Download the NICO++ dataset as specified by them into './Data/NICO'
data_path= './Data/NICO'

#### 7. MultiNLI
Follow https://github.com/izmailovpavel/spurious_feature_learning/tree/main to setup the dataset
data_path= './Data/multinli/'

#### 8. CelebAHair
Follow https://github.com/kohpangwei/group_DRO#celeba to download the dataset. Then, to use the hair color as the target attribute, we have provided the metadata file - 
data_path= 

### Verifying Dataset Setup

To verify that all datasets are properly set up, run:
```bash
python scripts/save_dataset_labels.py
```

This will create a `dataset_labels` directory containing JSON files with class and context labels for each dataset.

## Next Steps

After setting up the environment and datasets, you can proceed to:
1. [Training and Evaluation](docs/training.md)
2. [Coreset Selection Methods](docs/methods.md)
3. [Bias Analysis](docs/bias_analysis.md)


