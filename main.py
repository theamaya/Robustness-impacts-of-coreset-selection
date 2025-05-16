import os
import torch
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torch.utils.data import DataLoader
from utils import *
from datetime import datetime
import wandb
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Coreset Selection for Robustness')
    
    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Cmnist, waterbirds)')
    parser.add_argument('--model', type=str, default='ResNet18', help='Model architecture')
    parser.add_argument('--selection', type=str, default="uniform", help="Selection method")
    parser.add_argument('--fraction', default=0.1, type=float, help='Fraction of data to select')
    parser.add_argument('--data_path', type=str, default='data', help='Dataset path')
    
    # Training arguments
    parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--nesterov', default=True, type=str_to_bool, help='Use Nesterov momentum')
    
    # Selection arguments
    parser.add_argument('--selection_epochs', default=40, type=int, help='Epochs for selection')
    parser.add_argument('--selection_lr', type=float, default=0.001, help='Learning rate for selection')
    parser.add_argument('--uncertainty', default="Entropy", help='Uncertainty scoring method')
    parser.add_argument('--submodular', default="GraphCut", help='Submodular function')
    parser.add_argument('--submodular_greedy', default="LazyGreedy", help='Greedy algorithm for submodular optimization')
    parser.add_argument('--balance', default=True, type=str_to_bool, help='Balance selection per class')
    
    # Experiment arguments
    parser.add_argument('--num_exp', type=int, default=5, help='Number of experiments')
    parser.add_argument('--seed', default=None, type=int, help='Random seed')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id(s) to use')
    parser.add_argument('--save_path', type=str, default='', help='Path to save results')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint')
    
    # Optional arguments
    parser.add_argument('--pretrain', type=str_to_bool, default=False, help='Use pretrained model')
    parser.add_argument('--save_model', type=str_to_bool, default=False, help='Save model checkpoints')
    parser.add_argument('--linear_probe', type=str_to_bool, default=False, help='Use linear probing')
    parser.add_argument('--subset_path', type=str, default='', help='Path to pre-selected subset')
    parser.add_argument('--precalcfeatures_path', type=str, default='', help='Path to precalculated features')
    
    return parser.parse_args()

def init_wandb(args, exp_num):
    """Initialize Weights & Biases logging"""
    checkpoint_name = f"{args.dataset}_{args.model}_{args.selection}_exp{exp_num}_{args.fraction}"
    if args.uncertainty != "Entropy":
        checkpoint_name = f"{checkpoint_name}_{args.uncertainty}"
    
    wandb.init(
        project="bias_in_selection-selection",
        config=vars(args),
        name=checkpoint_name
    )
    return checkpoint_name

def load_checkpoint(args):
    """Load checkpoint if specified"""
    if not args.resume:
        return {}, 0, 0
    
    try:
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=args.device)
        
        if {"exp", "epoch", "state_dict", "opt_dict", "best_acc1", "rec", "subset", "sel_args"} <= set(checkpoint.keys()):
            return checkpoint, checkpoint['exp'], checkpoint["epoch"]
        elif {"exp", "subset", "sel_args"} <= set(checkpoint.keys()):
            print("=> The checkpoint only contains the subset, training will start from the beginning")
            return checkpoint, checkpoint['exp'], 0
        else:
            print("=> Failed to load the checkpoint, an empty one will be created")
            return {}, 0, 0
    except Exception as e:
        print(f"=> Error loading checkpoint: {e}")
        return {}, 0, 0

def save_selection(args, exp, subset, selection_args, checkpoint_name):
    """Save the selected subset and selection arguments"""
    if not args.save_path:
        return
    
    # Save the checkpoint with only the subset
    save_checkpoint({
        "exp": exp,
        "subset": subset,
        "sel_args": selection_args
    }, os.path.join(args.save_path, f"{checkpoint_name}_unknown.ckpt"), 0, 0.)

def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create directories if needed
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    
    # Load checkpoint if specified
    checkpoint, start_exp, start_epoch = load_checkpoint(args)
    
    for exp in range(start_exp, args.num_exp):
        # Set random seed
        args.seed = exp * 10000 + 1 if args.seed is None else args.seed
        torch.manual_seed(args.seed)
        
        # Initialize wandb
        checkpoint_name = init_wandb(args, exp)
        
        print(f'\n================== Exp {exp} ==================\n')
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Selection: {args.selection}")
        print(f"Fraction: {args.fraction}")
        print(f"Seed: {args.seed}")
        print(f"Device: {args.device}")
        
        # Load dataset
        if args.subset_path:
            data_subset = torch.load(args.subset_path)
            channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = datasets.__dict__[args.dataset](
                args.data_path, subset=data_subset)
        else:
            channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, _ = datasets.__dict__[args.dataset](
                args.data_path)
        
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
        
        # Load labels
        class_labels = torch.load(f'data/{args.dataset}_train_labels.pt')
        context_labels = torch.load(f'data/{args.dataset}_train_groups.pt')
        
        # Load precalculated features if specified
        precalcfeature = None
        if args.precalcfeatures_path:
            precalcfeature = (np.array(torch.load(args.precalcfeatures_path), dtype=object), class_labels)
        
        # Perform selection
        if "subset" in checkpoint:
            subset = checkpoint['subset']
            selection_args = checkpoint["sel_args"]
        else:
            selection_args = dict(
                epochs=args.selection_epochs,
                selection_method=args.uncertainty,
                balance=args.balance,
                greedy=args.submodular_greedy,
                function=args.submodular,
                torchvision_pretrain=args.pretrain,
                dst_test=dst_test,
                save_model=args.save_model,
                linear_probe=args.linear_probe,
                precalcfeature=precalcfeature
            )
            method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
            subset = method.select()
        
        print(f"Selected subset size: {len(subset['indices'])}")
        
        # Save selection if path is specified
        if args.save_path and not args.resume:
            save_selection(args, exp, subset, selection_args, checkpoint_name)
        
        # TODO: Add training loop here
        # This will be implemented in the next step
        
        wandb.finish()

if __name__ == '__main__':
    main()
