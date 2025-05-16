import os
import json
import torch
from deepcore.datasets import (
    civilcomments, waterbirds, urbancars, nico_plusplus, multinli,
    metashift, imagenetbg, imagenet, cxr, cmnist, celebahair
)

def save_labels(dataset_name, dataset_func, data_path, output_dir):
    """Save class and context labels for a dataset."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = dataset_func(data_path)
        
        # Get context labels if available
        context_labels = None
        if hasattr(dst_train, 'context_labels'):
            context_labels = dst_train.context_labels
        
        # Prepare data to save
        data = {
            'dataset_name': dataset_name,
            'num_classes': num_classes,
            'class_names': class_names,
            'context_labels': context_labels if context_labels is not None else [],
            'train_size': len(dst_train),
            'test_size': len(dst_test)
        }
        
        # Save to JSON file
        output_file = os.path.join(output_dir, f'{dataset_name}_labels.json')
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Saved labels for {dataset_name} to {output_file}")
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")

def main():
    # Configuration
    data_path = 'data'  # Update this to your data path
    output_dir = 'dataset_labels'
    
    # Dataset functions
    datasets = {
        'civilcomments': civilcomments,
        'waterbirds': waterbirds,
        'urbancars': urbancars,
        'nico_plusplus': nico_plusplus,
        'multinli': multinli,
        'metashift': metashift,
        'imagenetbg': imagenetbg,
        'imagenet': imagenet,
        'cxr': cxr,
        'cmnist': cmnist,
        'celebahair': celebahair
    }
    
    # Process each dataset
    for name, func in datasets.items():
        print(f"\nProcessing {name}...")
        save_labels(name, func, data_path, output_dir)

if __name__ == '__main__':
    main() 