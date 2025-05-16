import os
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as deepcoredatasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
import torch
import sys
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
import pickle
import csv
import pandas as pd
import json
from torchvision import datasets, transforms
from torch import tensor, long
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import numpy as np

def summarize_dataset(dataloader, dataset_name):
    class_counts = defaultdict(int)
    group_counts = defaultdict(int)
    class_group_counts = defaultdict(lambda: defaultdict(int))

    total_samples = 0

    # Iterate through the dataloader
    for images, class_labels, group_labels, _ in tqdm(dataloader):
        for class_label, group_label in zip(class_labels, group_labels):
            class_label= class_label.item()
            group_label= group_label.item()
            class_counts[class_label] += 1
            group_counts[group_label] += 1
            class_group_counts[class_label][group_label] += 1
            total_samples += 1

    # Convert counts to lists for easy calculation
    samples_per_class = list(class_counts.values())
    samples_per_group = list(group_counts.values())

    avg_samples_per_class = np.mean(samples_per_class)
    avg_samples_per_group = np.mean(samples_per_group)
    avg_samples_per_group_per_class = np.mean([np.mean(list(group_dict.values())) for group_dict in class_group_counts.values()])

    # Create a DataFrame for the class-group table
    class_group_df = pd.DataFrame(class_group_counts).fillna(0).astype(int)
    class_group_df.loc['Total'] = class_group_df.sum(axis=0)
    class_group_df['Total'] = class_group_df.sum(axis=1)

    # Print summary
    print(f"Summary for {dataset_name}:")
    # print(f"Total samples: {total_samples}")
    # print(f"Number of classes: {len(class_counts)}")
    # print(f"Number of groups: {len(group_counts)}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Samples per group: {samples_per_group}")
    # print(f"Average samples per class: {avg_samples_per_class:.2f}")
    # print(f"Average samples per group: {avg_samples_per_group:.2f}")
    # print(f"Average samples per group per class: {avg_samples_per_group_per_class:.2f}")
    print("\nClass-Group Distribution Table:")
    print(class_group_df)
    print("\n")


def save_labels(dataloader, path):
    all_labels=[]
    all_groups=[]
    for images, class_labels, group_labels, _ in tqdm(dataloader):
        for class_label, group_label in zip(class_labels, group_labels):
            class_label= class_label.item()
            group_label= group_label.item()
            all_labels.append(class_label)
            all_groups.append(group_label)
    all_labels= np.array(all_labels)
    all_groups= np.array(all_groups)
    torch.save(all_labels, path+'_labels.pt')
    torch.save(all_groups, path+'_groups.pt')

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, dst_val = deepcoredatasets.MimicCXR('/n/fs/visualai-scr/Data/MIMIC-CXR-JPG/mimic-cxr-jpg/2.1.0')
train_loader = torch.utils.data.DataLoader(dst_train, shuffle=False, batch_size=512)
val_loader = torch.utils.data.DataLoader(dst_val, shuffle=False, batch_size=512)
test_loader = torch.utils.data.DataLoader(dst_test, shuffle=False, batch_size=512)

save_labels(train_loader, '/n/fs/dk-diffusion/repos/DeepCore/data/mimic_cxr_train')
save_labels(val_loader, '/n/fs/dk-diffusion/repos/DeepCore/data/mimic_cxr_val')
save_labels(test_loader, '/n/fs/dk-diffusion/repos/DeepCore/data/mimic_cxr_test')