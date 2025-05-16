import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from deepcore.datasets import (
    Civilcomments, waterbirds, Urbancars_cooccur, Urbancars_bg, Urbancars_both, Nico_95_spurious, MultiNLI,
    Metashift, Cmnist, CelebAhair
)
from deepcore.utils.dataset_paths import dataset_paths

def save_dataset_labels(dataset_name):
    """
    Save class and group labels for a dataset as separate .pt files.
    
    Args:
        dataset_name (str): Name of the dataset to process
    """
    print(f"\nProcessing {dataset_name}...")
    
    # Get dataset path
    data_path = dataset_paths.get_path(dataset_name)
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist")
        return
    
    # Get dataset function
    dataset_funcs = {
        'Cmnist': Cmnist,
        'waterbirds': waterbirds,
        'Urbancars_cooccur': Urbancars_cooccur,
        'Urbancars_bg': Urbancars_bg,
        'Urbancars_both': Urbancars_both,
        'Nico_95_spurious': Nico_95_spurious,
        'MultiNLI': MultiNLI,
        'Metashift': Metashift,
        'CelebAhair': CelebAhair,
        'Civilcomments': Civilcomments
    }
    
    if dataset_name not in dataset_funcs:
        print(f"Error: Dataset {dataset_name} not found")
        print(f"Available datasets: {list(dataset_funcs.keys())}")
        return
    
    # Load dataset
    try:
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, dst_val = dataset_funcs[dataset_name](data_path)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Create output directory
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataloaders
    train_loader = DataLoader(dst_train, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(dst_test, batch_size=1, shuffle=False, num_workers=4)
    val_loader = DataLoader(dst_val, batch_size=1, shuffle=False, num_workers=4)
    
    # Process training set
    print("Processing training set...")
    train_labels = []
    train_groups = []
    
    for batch in tqdm(train_loader):
        # Different datasets return different numbers of items
        # Usually: (image, label, group_label, ...)
        if len(batch) >= 3:  # Has both label and group label
            _, label, group_label = batch[:3]
            train_labels.append(label.item())
            train_groups.append(group_label.item())
        elif len(batch) == 2:  # Only has label
            _, label = batch
            train_labels.append(label.item())
            train_groups.append(-1)  # No group label
    
    # Convert to tensors and save
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_groups = torch.tensor(train_groups, dtype=torch.long)
    
    torch.save(train_labels, os.path.join(output_dir, f'{dataset_name}_train_labels.pt'))
    torch.save(train_groups, os.path.join(output_dir, f'{dataset_name}_train_groups.pt'))
    
    print(f"Saved training labels: {train_labels.shape}")
    print(f"Saved training groups: {train_groups.shape}")
    
    # Process validation set
    print("\nProcessing validation set...")
    val_labels = []
    val_groups = []
    
    for batch in tqdm(val_loader):
        if len(batch) >= 3:  # Has both label and group label
            _, label, group_label = batch[:3]
            val_labels.append(label.item())
            val_groups.append(group_label.item())
        elif len(batch) == 2:  # Only has label
            _, label = batch
            val_labels.append(label.item())
            val_groups.append(-1)  # No group label
    
    # Convert to tensors and save
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    val_groups = torch.tensor(val_groups, dtype=torch.long)
    
    torch.save(val_labels, os.path.join(output_dir, f'{dataset_name}_val_labels.pt'))
    torch.save(val_groups, os.path.join(output_dir, f'{dataset_name}_val_groups.pt'))
    
    print(f"Saved validation labels: {val_labels.shape}")
    print(f"Saved validation groups: {val_groups.shape}")
    
    # Process test set
    print("\nProcessing test set...")
    test_labels = []
    test_groups = []
    
    for batch in tqdm(test_loader):
        if len(batch) >= 3:  # Has both label and group label
            _, label, group_label = batch[:3]
            test_labels.append(label.item())
            test_groups.append(group_label.item())
        elif len(batch) == 2:  # Only has label
            _, label = batch
            test_labels.append(label.item())
            test_groups.append(-1)  # No group label
    
    # Convert to tensors and save
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    test_groups = torch.tensor(test_groups, dtype=torch.long)
    
    torch.save(test_labels, os.path.join(output_dir, f'{dataset_name}_test_labels.pt'))
    torch.save(test_groups, os.path.join(output_dir, f'{dataset_name}_test_groups.pt'))
    
    print(f"Saved test labels: {test_labels.shape}")
    print(f"Saved test groups: {test_groups.shape}")
    
    # Print statistics
    print("\nLabel statistics:")
    print("Training set:")
    print(f"  Total samples: {len(train_labels)}")
    print(f"  Label distribution: {torch.bincount(train_labels).tolist()}")
    if (train_groups != -1).any():
        print(f"  Group distribution: {torch.bincount(train_groups[train_groups != -1]).tolist()}")
    
    print("\nValidation set:")
    print(f"  Total samples: {len(val_labels)}")
    print(f"  Label distribution: {torch.bincount(val_labels).tolist()}")
    if (val_groups != -1).any():
        print(f"  Group distribution: {torch.bincount(val_groups[val_groups != -1]).tolist()}")
    
    print("\nTest set:")
    print(f"  Total samples: {len(test_labels)}")
    print(f"  Label distribution: {torch.bincount(test_labels).tolist()}")
    if (test_groups != -1).any():
        print(f"  Group distribution: {torch.bincount(test_groups[test_groups != -1]).tolist()}")
    
    print(f"\nLabels saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Save dataset labels and group labels as .pt files')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset to process')
    args = parser.parse_args()
    
    save_dataset_labels(args.dataset_name)

if __name__ == '__main__':
    main() 