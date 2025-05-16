import os
import torch.nn as nn
import argparse
import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
import numpy as np
import wandb
import torchvision
import numpy as np
from collections import defaultdict
import torch
import transformers
import numpy as np
from collections import defaultdict

import torch

class CoresetSelection(object):
    @staticmethod
    def score_monotonic_selection(data_score, key, ratio, descending, class_balanced):
        score = data_score[key]
        score_sorted_index = score.argsort(descending=descending)
        total_num = ratio * data_score['targets'].shape[0]

        if class_balanced:
            print('Class balance mode.')
            all_index = torch.arange(data_score['targets'].shape[0])
            #Permutation
            targets_list = data_score['targets'][score_sorted_index]
            targets_unique = torch.unique(targets_list)
            for target in targets_unique:
                target_index_mask = (targets_list == target)
                targets_num = target_index_mask.sum()

            #Guarantee the class ratio doesn't change
            selected_index = []
            for target in targets_unique:
                target_index_mask = (targets_list == target)
                target_index = all_index[target_index_mask]
                target_coreset_num = targets_num * ratio
                selected_index = selected_index + list(target_index[:int(target_coreset_num)])
            selected_index = torch.tensor(selected_index)
            print(f'High priority {key}: {score[score_sorted_index[selected_index][:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[selected_index][-15:]]}')

            return score_sorted_index[selected_index]

        else:
            print(f'High priority {key}: {score[score_sorted_index[:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[-15:]]}')
            return score_sorted_index[:int(total_num)]

    @staticmethod
    def mislabel_mask(data_score, mis_key, mis_num, mis_descending, coreset_key):
        mis_score = data_score[mis_key]
        mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
        hard_index = mis_score_sorted_index[:mis_num]
        print(f'Bad data -> High priority {mis_key}: {data_score[mis_key][hard_index][:15]}')
        print(f'Prune {hard_index.shape[0]} samples.')

        easy_index = mis_score_sorted_index[mis_num:]
        data_score[coreset_key] = data_score[coreset_key][easy_index]

        return data_score, easy_index


    @staticmethod
    def stratified_sampling(data_score, coreset_key, coreset_num):
        stratas = 50
        # print('Using stratified sampling...')
        score = torch.from_numpy(data_score)
        total_num = coreset_num

        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)

        strata_num = torch.tensor(strata_num)

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]

            num_bin = bins.shape[0]

            rest_exp_num = num
            budgets = []
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                budgets.append(cur_num)
                rest_exp_num -= cur_num


            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)

            return rst

        budgets = bin_allocate(total_num, strata_num)

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(data_score.shape[0])

        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            rand_index = torch.randperm(pool.shape[0])
            selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]

        return selected_index, None

    @staticmethod
    def random_selection(total_num, num):
        print('Random selection.')
        score_random_index = torch.randperm(total_num)

        return score_random_index[:int(num)]


def class_balanced_stratified_sampling(scores, class_labels, n_percent, random_state=None):
    # Set the random seed for reproducibility
    rng = np.random.default_rng(random_state)
    
    # Calculate the total number of samples to select
    total_samples = len(scores)
    num_samples_to_select = int(total_samples * n_percent)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)
    num_classes = len(unique_classes)
    
    # Calculate the ideal number of samples to select per class (uniform distribution)
    ideal_samples_per_class = num_samples_to_select // num_classes
    
    # Initialize selection variables
    selected_indices = []
    shortfall = 0
    class_samples_to_select = np.zeros(num_classes, dtype=int)
    maxed_out_classes = set()
    
    # Sort the indices of scores in descending order
    sorted_indices = np.argsort(-scores)
    
    # Assign the ideal number of samples to each class if possible, otherwise track the shortfall
    for i, class_label in enumerate(unique_classes):
        class_indices = sorted_indices[class_labels[sorted_indices] == class_label]
        if len(class_indices) >= ideal_samples_per_class:
            class_samples_to_select[i] = ideal_samples_per_class
        else:
            class_samples_to_select[i] = len(class_indices)
            shortfall += (ideal_samples_per_class - len(class_indices))
            maxed_out_classes.add(i)
    
    # Distribute the shortfall among the remaining non-maxed-out classes
    while shortfall > 0:
        remaining_classes = [i for i in range(num_classes) if i not in maxed_out_classes]
        if not remaining_classes:
            break
        equal_share = shortfall // len(remaining_classes)
        remainder = shortfall % len(remaining_classes)
        
        for i in remaining_classes:
            class_indices = sorted_indices[class_labels[sorted_indices] == unique_classes[i]]
            max_possible = len(class_indices) - class_samples_to_select[i]
            additional = min(max_possible, equal_share + (1 if remainder > 0 else 0))
            class_samples_to_select[i] += additional
            shortfall -= additional
            remainder -= 1
            if class_samples_to_select[i] == len(class_indices):
                maxed_out_classes.add(i)
    
    # print(class_samples_to_select)
    # Now, select the top samples from each class based on the scores
    scores= torch.from_numpy(scores)
    for i, class_label in enumerate(unique_classes):
        class_indices = np.where(class_labels == class_label)[0]
        selected_class_indices, _ = CoresetSelection.stratified_sampling(data_score=scores[class_indices], coreset_key=None, coreset_num=class_samples_to_select[i])
        # print(class_samples_to_select[i], len(selected_class_indices))
        selected_indices.extend(class_indices[selected_class_indices])
    
    # Return the selected indices
    return np.array(selected_indices)


def select_top_n_percent_with_class_balance(scores, class_labels, n_percent, random_state=None):
    # Set the random seed for reproducibility
    rng = np.random.default_rng(random_state)
    
    # Calculate the total number of samples to select
    total_samples = len(scores)
    num_samples_to_select = int(total_samples * n_percent)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)
    num_classes = len(unique_classes)
    
    # Calculate the ideal number of samples to select per class (uniform distribution)
    ideal_samples_per_class = num_samples_to_select // num_classes
    
    # Initialize selection variables
    selected_indices = []
    shortfall = 0
    class_samples_to_select = np.zeros(num_classes, dtype=int)
    maxed_out_classes = set()
    
    # Sort the indices of scores in descending order
    sorted_indices = np.argsort(-scores)
    
    # Assign the ideal number of samples to each class if possible, otherwise track the shortfall
    for i, class_label in enumerate(unique_classes):
        class_indices = sorted_indices[class_labels[sorted_indices] == class_label]
        if len(class_indices) >= ideal_samples_per_class:
            class_samples_to_select[i] = ideal_samples_per_class
        else:
            class_samples_to_select[i] = len(class_indices)
            shortfall += (ideal_samples_per_class - len(class_indices))
            maxed_out_classes.add(i)
    
    # Distribute the shortfall among the remaining non-maxed-out classes
    while shortfall > 0:
        remaining_classes = [i for i in range(num_classes) if i not in maxed_out_classes]
        if not remaining_classes:
            break
        equal_share = shortfall // len(remaining_classes)
        remainder = shortfall % len(remaining_classes)
        
        for i in remaining_classes:
            class_indices = sorted_indices[class_labels[sorted_indices] == unique_classes[i]]
            max_possible = len(class_indices) - class_samples_to_select[i]
            additional = min(max_possible, equal_share + (1 if remainder > 0 else 0))
            class_samples_to_select[i] += additional
            shortfall -= additional
            remainder -= 1
            if class_samples_to_select[i] == len(class_indices):
                maxed_out_classes.add(i)
    
    # print(class_samples_to_select)
    # Now, select the top samples from each class based on the scores
    for i, class_label in enumerate(unique_classes):
        class_indices = sorted_indices[class_labels[sorted_indices] == class_label]
        selected_indices.extend(class_indices[:class_samples_to_select[i]])
    
    # Return the selected indices
    return np.array(selected_indices)

def select_uniform_balanced_subset(class_labels, context_labels, fraction, scores=None, hard_majority=None, random_state=None):
    rng = np.random.default_rng(random_state)
    
    # Calculate the total number of samples to select
    total_samples = len(class_labels)
    sample_indices= np.arange(total_samples)
    num_samples_to_select = int(total_samples * fraction)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)
    num_classes = len(unique_classes)
    
    # Calculate the ideal number of samples to select per class (uniform distribution)
    ideal_samples_per_class = num_samples_to_select // num_classes
    
    # Initialize selection variables
    selected_indices = []
    shortfall = 0
    class_samples_to_select = np.zeros(num_classes, dtype=int)
    maxed_out_classes = set()
    
    # Assign the ideal number of samples to each class if possible, otherwise track the shortfall
    for i, class_label in enumerate(unique_classes):
        class_indices = sample_indices[class_labels == class_label]
        if len(class_indices) >= ideal_samples_per_class:
            class_samples_to_select[i] = ideal_samples_per_class
        else:
            class_samples_to_select[i] = len(class_indices)
            shortfall += (ideal_samples_per_class - len(class_indices))
            maxed_out_classes.add(i)
    
    # Distribute the shortfall among the remaining non-maxed-out classes
    while shortfall > 0:
        remaining_classes = [i for i in range(num_classes) if i not in maxed_out_classes]
        if not remaining_classes:
            break
        equal_share = shortfall // len(remaining_classes)
        remainder = shortfall % len(remaining_classes)
        
        for i in remaining_classes:
            class_indices = sample_indices[class_labels == i]
            max_possible = len(class_indices) - class_samples_to_select[i]
            additional = min(max_possible, equal_share + (1 if remainder > 0 else 0))
            class_samples_to_select[i] += additional
            shortfall -= additional
            remainder -= 1
            if class_samples_to_select[i] == len(class_indices):
                maxed_out_classes.add(i)
 
    print('class_samples_to_select', class_samples_to_select)

    groups = list(zip(class_labels, context_labels))
    
    # Get the unique classes and groups
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)
    unique_groups, group_counts = np.unique(groups, return_counts=True, axis=0)
    selected_indices = []
    
    for cls, total_cls_count in zip(unique_classes, class_samples_to_select):
        class_indices = np.where(class_labels == cls)[0]
        class_group_indices = [i for i, (c, _) in enumerate(unique_groups) if c == cls]

        ideal_samples_per_group= int(total_cls_count/len(class_group_indices))

        group_samples_to_select = np.zeros(len(class_group_indices), dtype=int)

        maxed_out_groups=set()
        shortfall=0
        for i, group_index in enumerate(class_group_indices):
            group_indices = [j for j, (g1,g2) in enumerate(groups) if g1 == unique_groups[group_index][0] and g2 == unique_groups[group_index][1]]
            if len(group_indices) >= ideal_samples_per_group:
                group_samples_to_select[i] = ideal_samples_per_group
            else:
                group_samples_to_select[i] = len(group_indices)
                shortfall += (ideal_samples_per_group - len(group_indices))
                maxed_out_groups.add(i)
        
        while shortfall > 0:
            remaining_groups = [i for i in range(len(class_group_indices)) if i not in maxed_out_groups]
            if not remaining_groups:
                break
            equal_share = shortfall // len(remaining_groups)
            remainder = shortfall % len(remaining_groups)
            
            for i in remaining_groups:
                group_indices = [j for j, (g1,g2) in enumerate(groups) if g1 == unique_groups[class_group_indices[i]][0] and g2 == unique_groups[class_group_indices[i]][1]]
                max_possible = len(group_indices) - group_samples_to_select[i]
                additional = min(max_possible, equal_share + (1 if remainder > 0 else 0))
                group_samples_to_select[i] += additional
                shortfall -= additional
                remainder -= 1
                if group_samples_to_select[i] == len(group_indices):
                    maxed_out_groups.add(i)

        print("group_samples_to_select: ",group_samples_to_select)
        
        for i, group_index in enumerate(class_group_indices):
            count= group_samples_to_select[i]
            group_indices = np.array([j for j, (g1,g2) in enumerate(groups) if g1 == unique_groups[group_index][0] and g2 == unique_groups[group_index][1]])
            if scores is not None:
                group_scores= scores[group_indices]
                current_group_maj = False
                if unique_groups[group_index][0] == unique_groups[group_index][1]:
                    current_group_maj = True
                if hard_majority == None:
                    sorted_group_indices= group_indices[(np.argsort(-group_scores).astype(int))]
                    selected_indices.extend(sorted_group_indices[:count])
                elif hard_majority== True:
                    if current_group_maj == True:
                        sorted_group_indices= group_indices[(np.argsort(-group_scores).astype(int))]
                        selected_indices.extend(sorted_group_indices[:count])
                    else:
                        sorted_group_indices= group_indices[(np.argsort(group_scores).astype(int))]
                        selected_indices.extend(sorted_group_indices[:count])
                elif hard_majority== False:
                    if current_group_maj == True:
                        sorted_group_indices= group_indices[(np.argsort(group_scores).astype(int))]
                        selected_indices.extend(sorted_group_indices[:count])
                    else:
                        sorted_group_indices= group_indices[(np.argsort(-group_scores).astype(int))]
                        selected_indices.extend(sorted_group_indices[:count])

            else:
                selected_indices.extend(np.random.choice(group_indices, count, replace=False))

    return np.array(selected_indices)

def select_uniform_balanced_stratified_subset(class_labels, context_labels, fraction, scores=None, hard_majority=None, random_state=None):
    rng = np.random.default_rng(random_state)
    
    # Calculate the total number of samples to select
    total_samples = len(class_labels)
    sample_indices= np.arange(total_samples)
    num_samples_to_select = int(total_samples * fraction)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)
    num_classes = len(unique_classes)
    
    # Calculate the ideal number of samples to select per class (uniform distribution)
    ideal_samples_per_class = num_samples_to_select // num_classes
    
    # Initialize selection variables
    selected_indices = []
    shortfall = 0
    class_samples_to_select = np.zeros(num_classes, dtype=int)
    maxed_out_classes = set()
    
    # Assign the ideal number of samples to each class if possible, otherwise track the shortfall
    for i, class_label in enumerate(unique_classes):
        class_indices = sample_indices[class_labels == class_label]
        if len(class_indices) >= ideal_samples_per_class:
            class_samples_to_select[i] = ideal_samples_per_class
        else:
            class_samples_to_select[i] = len(class_indices)
            shortfall += (ideal_samples_per_class - len(class_indices))
            maxed_out_classes.add(i)
    
    # Distribute the shortfall among the remaining non-maxed-out classes
    while shortfall > 0:
        remaining_classes = [i for i in range(num_classes) if i not in maxed_out_classes]
        if not remaining_classes:
            break
        equal_share = shortfall // len(remaining_classes)
        remainder = shortfall % len(remaining_classes)
        
        for i in remaining_classes:
            class_indices = sample_indices[class_labels == i]
            max_possible = len(class_indices) - class_samples_to_select[i]
            additional = min(max_possible, equal_share + (1 if remainder > 0 else 0))
            class_samples_to_select[i] += additional
            shortfall -= additional
            remainder -= 1
            if class_samples_to_select[i] == len(class_indices):
                maxed_out_classes.add(i)
 
    print('class_samples_to_select', class_samples_to_select)

    groups = list(zip(class_labels, context_labels))
    
    # Get the unique classes and groups
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)
    unique_groups, group_counts = np.unique(groups, return_counts=True, axis=0)
    selected_indices = []
    
    for cls, total_cls_count in zip(unique_classes, class_samples_to_select):
        class_indices = np.where(class_labels == cls)[0]
        class_group_indices = [i for i, (c, _) in enumerate(unique_groups) if c == cls]

        ideal_samples_per_group= int(total_cls_count/len(class_group_indices))

        group_samples_to_select = np.zeros(len(class_group_indices), dtype=int)

        maxed_out_groups=set()
        shortfall=0
        for i, group_index in enumerate(class_group_indices):
            group_indices = [j for j, (g1,g2) in enumerate(groups) if g1 == unique_groups[group_index][0] and g2 == unique_groups[group_index][1]]
            if len(group_indices) >= ideal_samples_per_group:
                group_samples_to_select[i] = ideal_samples_per_group
            else:
                group_samples_to_select[i] = len(group_indices)
                shortfall += (ideal_samples_per_group - len(group_indices))
                maxed_out_groups.add(i)
        
        while shortfall > 0:
            remaining_groups = [i for i in range(len(class_group_indices)) if i not in maxed_out_groups]
            if not remaining_groups:
                break
            equal_share = shortfall // len(remaining_groups)
            remainder = shortfall % len(remaining_groups)
            
            for i in remaining_groups:
                group_indices = [j for j, (g1,g2) in enumerate(groups) if g1 == unique_groups[class_group_indices[i]][0] and g2 == unique_groups[class_group_indices[i]][1]]
                max_possible = len(group_indices) - group_samples_to_select[i]
                additional = min(max_possible, equal_share + (1 if remainder > 0 else 0))
                group_samples_to_select[i] += additional
                shortfall -= additional
                remainder -= 1
                if group_samples_to_select[i] == len(group_indices):
                    maxed_out_groups.add(i)

        print("group_samples_to_select: ",group_samples_to_select)
        
        for i, group_index in enumerate(class_group_indices):
            count= group_samples_to_select[i]
            group_indices = np.array([j for j, (g1,g2) in enumerate(groups) if g1 == unique_groups[group_index][0] and g2 == unique_groups[group_index][1]])
            # scores= torch.from_numpy(scores)
            selected_group_indices, _ = CoresetSelection.stratified_sampling(data_score=scores[group_indices], coreset_key=None, coreset_num=group_samples_to_select[i])
            # print(class_samples_to_select[i], len(selected_class_indices))
            selected_indices.extend(group_indices[selected_group_indices])
        

    return np.array(selected_indices)
    

def get_subset(subset_path=None, score_path=None, selection=None, fraction=None, policy=None, class_balance= False, class_equal= False, class_labels= None, context_labels=None, drop_percent=0):

    x = torch.load(score_path)
    if selection == 'Loss':
        scores= x['subset']['scores']['loss'].tolist()
    elif selection == 'Accuracy':
        scores= -1*np.array(x['subset']['scores']['acc'].tolist())
    else:
        scores= x['subset']['scores'].tolist()
        if selection== 'Areaum' or selection== 'DeepFool':
            scores= -1*np.array(scores)

    n_samples = len(scores)
    n_select = int(n_samples * fraction)

    if subset_path== '':
        subset_indices= np.arange(n_samples)
    else:
        subset_indices= torch.load(subset_path)

    if class_balance:
        scores= np.array(scores)
        unique_classes, class_counts = np.unique(class_labels, return_counts=True)

        if policy == 'difficult':
            # selected_indices = np.argsort(scores)[-n_select:]
            samples_to_select = (class_counts * fraction).astype(int)
            selected_indices = []
            for cls, count in zip(unique_classes, samples_to_select):
                class_indices = np.where(class_labels == cls)[0].astype(int)
                sorted_indices = class_indices[np.argsort(-scores[class_indices])]
                if count > 0:
                    selected_indices.extend(sorted_indices[:count])
            selected_indices = np.array(selected_indices)
        elif policy == 'easy':
            # selected_indices = np.argsort(scores)[:n_select]
            samples_to_select = (class_counts * fraction).astype(int)
            selected_indices = []
            for cls, count in zip(unique_classes, samples_to_select):
                class_indices = np.where(class_labels == cls)[0].astype(int)
                sorted_indices = class_indices[np.argsort(-scores[class_indices])]
                if count > 0:
                    selected_indices.extend(sorted_indices[-count:])
            selected_indices = np.array(selected_indices)
        elif policy == 'random':
            selected_indices = np.random.choice(n_samples, n_select, replace=False)
            samples_to_select = (class_counts * fraction).astype(int)
            selected_indices = []
            for cls, count in zip(unique_classes, samples_to_select):
                class_indices = np.where(class_labels == cls)[0].astype(int)
                random_indices = np.random.choice(class_indices, count, replace=False)
                if count > 0:
                    selected_indices.extend(random_indices)
            selected_indices = np.array(selected_indices)
        elif policy == 'median':
            median_score = np.median(scores)
            distances = np.abs(scores - median_score)
            # selected_indices = np.argsort(distances)[:n_select]
            samples_to_select = (class_counts * fraction).astype(int)
            selected_indices = []
            for cls, count in zip(unique_classes, samples_to_select):
                class_indices = np.where(class_labels == cls)[0].astype(int)
                sorted_indices = class_indices[np.argsort(-distances[class_indices])]
                if count > 0:
                    selected_indices.extend(sorted_indices[-count:])
            selected_indices = np.array(selected_indices)
        else:
            selected_indices = select_classlevel_groupbalanced_subset(class_labels, context_labels, fraction)

    elif class_equal:
        scores= np.array(scores)
        unique_classes, class_counts = np.unique(class_labels, return_counts=True)
        if policy == 'half-difficult-easy':
            selected_indices1 = select_top_n_percent_with_class_balance(scores, class_labels, fraction/2)#, random_state=42)
            selected_indices2 = select_top_n_percent_with_class_balance(-scores, class_labels, fraction/2)#, random_state=42)
            selected_indices= np.union1d(selected_indices1, selected_indices2)
        elif policy[:9] == 'difficult':
            if policy == 'difficult-filtered':
                drop_num= int(n_samples*drop_percent)
                drop_indices = np.argsort(scores)[-drop_num:]  # Get indices of the 100 highest values
                # Replace those values with 0
                scores[drop_indices] = 0
                selected_indices = select_top_n_percent_with_class_balance(scores, class_labels, fraction)#, random_state=42)
            elif policy == 'difficult':
                # selected_indices = np.argsort(scores)[-n_select:]
                selected_indices = select_top_n_percent_with_class_balance(scores, class_labels, fraction)#, random_state=42)
            elif policy == 'difficult-filtered-groupbal':
                drop_num= int(n_samples*drop_percent)
                drop_indices = np.argsort(scores)[-drop_num:] 
                scores[drop_indices] = 0
                selected_indices = select_top_n_percent_with_class_balance(scores, class_labels, fraction)#, random_state=42)
            else:
                selected_indices = select_uniform_balanced_subset(class_labels, context_labels, fraction, scores=scores)#, random_state=42)
        elif policy[:4] == 'easy':
            if policy == 'easy':
                # selected_indices = np.argsort(scores)[:n_select]
                selected_indices = select_top_n_percent_with_class_balance(-scores, class_labels, fraction)#, random_state=42)
            else:
                selected_indices = select_uniform_balanced_subset(class_labels, context_labels, fraction, scores=-scores)#, random_state=42)
        elif policy[:6] == 'random':
            rng = np.random.default_rng()
            random_scores = rng.random(len(class_labels))
            if policy == 'random':
                selected_indices = select_top_n_percent_with_class_balance(random_scores, class_labels, fraction)#, random_state=42)
            else:
                selected_indices = select_uniform_balanced_subset(class_labels, context_labels, fraction, scores=random_scores)#, random_state=42)
        elif policy[:6] == 'median':
            median_score = np.median(scores)
            distances = np.abs(scores - median_score)
            if policy =='median':
                # selected_indices = np.argsort(distances)[:n_select]
                selected_indices = select_top_n_percent_with_class_balance(-distances, class_labels, fraction)#, random_state=42)
            else:
                selected_indices = select_uniform_balanced_subset(class_labels, context_labels, fraction, scores=-distances)#, random_state=42)
        elif policy == 'stratified':
            selected_indices = class_balanced_stratified_sampling(scores, class_labels, fraction, random_state=None)
        elif policy == 'stratified-groupbal':
            selected_indices =  select_uniform_balanced_stratified_subset(class_labels, context_labels, fraction, scores=scores)
        else:
            if 'hard_majority' in policy:
                selected_indices = select_uniform_balanced_subset(class_labels, context_labels, fraction, scores=scores, hard_majority=True)
            elif 'easy_majority' in policy:
                selected_indices = select_uniform_balanced_subset(class_labels, context_labels, fraction, scores=scores, hard_majority=False)
            else:
                selected_indices = select_uniform_balanced_subset(class_labels, context_labels, fraction)#, random_state=42)

    else:

        if policy == 'difficult':
            selected_indices = np.argsort(scores)[-n_select:]
        elif policy == 'easy':
            selected_indices = np.argsort(scores)[:n_select]
        elif policy == 'random':
            selected_indices = np.random.choice(n_samples, n_select, replace=False)
        elif policy == 'median':
            median_score = np.median(scores)
            distances = np.abs(scores - median_score)
            selected_indices = np.argsort(distances)[:n_select]
        
    return subset_indices[selected_indices]

def bert_adamw_optimizer(model, lr, momentum, weight_decay):
    # Adapted from https://github.com/facebookresearch/BalancingGroups/blob/main/models.py
    del momentum
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer

def bert_lr_scheduler(optimizer, num_steps):
    return transformers.get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=num_steps)

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=1, type=int, help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--cross", type=str, nargs="+", default=None, help="models for cross-architecture experiments")

    # Optimizer and scheduler
    parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for updating network parameters')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument("--nesterov", default=True, type=str_to_bool, help="if set nesterov")
    parser.add_argument("--scheduler", default="CosineAnnealingLR", type=str, help=
    "Learning rate scheduler")
    parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
    parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")

    # Training
    parser.add_argument('--batch', '--batch-size', "-b", default=128, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                     help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                     help="batch size for selection, if not specified, it will equal to batch size in argument --batch")

    # Testing
    parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
    parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                        help="proportion of test dataset used for evaluating the model (default: 1.)")

    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="SGD",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=str_to_bool,
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.01, help='learning rate for selection')  #previous was 0.1
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
             help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")

    # Algorithm
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")

    # Checkpoint and resumption
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
    parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")
    parser.add_argument('--pretrain', type= str_to_bool, default= False)
    parser.add_argument('--imagenet_pretrain', type= str_to_bool, default= False)
    parser.add_argument('--save_model', type= str_to_bool, default= False)
    parser.add_argument('--linear_probe', type= str_to_bool, default= False)
    parser.add_argument('--subset_path', type= str, default= '')
    parser.add_argument('--score_path', type= str, default= '')
    parser.add_argument('--policy', type= str, default= '')
    parser.add_argument('--level', type= int, default= 0)    
    parser.add_argument('--score_pretrain', type= str_to_bool, default= False)
    parser.add_argument('--class_balance', type= str_to_bool, default= False)
    parser.add_argument('--class_equal', type= str_to_bool, default= False)
    parser.add_argument('--features', type= str, default= '')
    parser.add_argument('--drop_percent', type=float, default=0.00)


    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    checkpoint = {}
    start_exp = 0
    start_epoch = 0

    for exp in range(start_exp, args.num_exp):
        args.seed= exp*10000 +1

        if args.save_path != "":
            if args.score_pretrain:
                if args.linear_probe:
                    checkpoint_name = "{dst}_level{lvl}_{net}_pretrainedlp_{mtd}_frac{fr}_{policy}_exp{exp}_pretrainedScores".format(dst=args.dataset,
                                                                        lvl=args.level,
                                                                        net=args.model,
                                                                        mtd=args.selection,
                                                                        exp=exp,
                                                                        fr=args.fraction,
                                                                        policy=args.policy)
                elif args.pretrain:
                    checkpoint_name = "{dst}_level{lvl}_{net}_pretrained_{mtd}_frac{fr}_{policy}_exp{exp}_pretrainedScores{features}".format(dst=args.dataset,
                                                                        lvl=args.level,
                                                                        net=args.model,
                                                                        mtd=args.selection,
                                                                        exp=exp,
                                                                        fr=args.fraction,
                                                                        policy=args.policy,
                                                                        features= args.features)
                else:
                    checkpoint_name = "{dst}_level{lvl}_{net}_{mtd}_frac{fr}_{policy}_exp{exp}_pretrainedScores".format(dst=args.dataset,
                                                                        lvl=args.level,
                                                                        net=args.model,
                                                                        mtd=args.selection,
                                                                        exp=exp,
                                                                        fr=args.fraction,
                                                                        policy=args.policy)
            else:
                if args.linear_probe:
                    checkpoint_name = "{dst}_level{lvl}_{net}_pretrainedlp_{mtd}_frac{fr}_{policy}_exp{exp}_".format(dst=args.dataset,
                                                                        lvl=args.level,
                                                                        net=args.model,
                                                                        mtd=args.selection,
                                                                        exp=exp,
                                                                        fr=args.fraction,
                                                                        policy=args.policy)
                elif args.pretrain:
                    checkpoint_name = "{dst}_level{lvl}_{net}_pretrained_{mtd}_frac{fr}_{policy}_exp{exp}_".format(dst=args.dataset,
                                                                        lvl=args.level,
                                                                        net=args.model,
                                                                        mtd=args.selection,
                                                                        exp=exp,
                                                                        fr=args.fraction,
                                                                        policy=args.policy)
                else:
                    checkpoint_name = "{dst}_level{lvl}_{net}_{mtd}_frac{fr}_{policy}_exp{exp}_".format(dst=args.dataset,
                                                                        lvl=args.level,
                                                                        net=args.model,
                                                                        mtd=args.selection,
                                                                        exp=exp,
                                                                        fr=args.fraction,
                                                                        policy=args.policy)

        print('\n================== Exp %d ==================\n' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", selection: ", args.selection, ", num_ex: ",
              args.num_exp, ", epochs: ", args.epochs, ", fraction: ", args.fraction, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", resume: ", args.resume, ", device: ", args.device,
              ", checkpoint_name: " + checkpoint_name if args.save_path != "" else "", "\n", sep="")

        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, dst_val = datasets.__dict__[args.dataset] \
        (args.data_path)
        
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

        class_labels= torch.load(f'./data/{args.dataset}_train_labels.pt')
        context_labels= torch.load(f'./data/{args.dataset}_train_groups.pt')

        torch.random.manual_seed(args.seed)
        print(checkpoint_name)
        print(vars(args))

        ###########
        subset = get_subset(args.subset_path, args.score_path, args.selection, args.fraction, args.policy, args.class_balance, args.class_equal, class_labels, context_labels, args.drop_percent)
        ###########


        if_weighted = False
        if args.fraction!= 1.0:
            dst_subset = torch.utils.data.Subset(dst_train, subset)
        else:
            dst_subset = dst_train

        # BackgroundGenerator for ImageNet to speed up dataloaders
        if args.dataset == "ImageNet":
            train_loader = DataLoaderX(dst_subset, batch_size=args.train_batch, shuffle=True,
                                       num_workers=args.workers, pin_memory=True)
            test_loader = DataLoaderX(dst_test, batch_size=args.train_batch, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                       num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(dst_val, batch_size=args.train_batch, shuffle=False,
                                                       num_workers=args.workers, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                      num_workers=args.workers, pin_memory=True)

        # Listing cross-architecture experiment settings if specified.
        models = [args.model]
        if isinstance(args.cross, list):
            for model in args.cross:
                if model != args.model:
                    models.append(model)

        for model in models:
        #     if len(models) > 1:
        #         print("| Training on model %s" % model)

            network = nets.__dict__[model](channel, num_classes, im_size, pretrained=args.pretrain,linear_probe= args.linear_probe).to(args.device)

            if args.imagenet_pretrain:
                network= torchvision.models.resnet50(pretrained=True)
                d = network.fc.in_features
                network.fc = torch.nn.Linear(d, num_classes)
                print("Using pretrained imagenet")

            if args.device == "cpu":
                print("Using CPU.")
            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu[0])
                network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
            elif torch.cuda.device_count() > 1:
                network = nets.nets_utils.MyDataParallel(network).cuda()

            if "state_dict" in checkpoint.keys():
                # Loading model state_dict
                network.load_state_dict(checkpoint["state_dict"])

            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

            # Optimizer
            if args.optimizer == "SGD":
                optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer == "Adam":
                optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == "bert_adamw_optimizer":
                optimizer = bert_adamw_optimizer(network, args.lr, args.momentum, args.weight_decay)
            else:
                optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                                 weight_decay=args.weight_decay, nesterov=args.nesterov)

            # LR scheduler
            if args.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.epochs,
                                                                       eta_min=args.min_lr)
                scheduler.last_epoch = (start_epoch - 1) * len(train_loader)
            elif args.scheduler == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) * args.step_size,
                                                            gamma=args.gamma)
                scheduler.last_epoch = (start_epoch - 1) * len(train_loader)
            elif args.scheduler == "bert_lr_scheduler":
                scheduler= bert_lr_scheduler(optimizer, args.epochs)
            else:
                # scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)
                # scheduler.last_epoch = (start_epoch - 1) * len(train_loader)
                scheduler = None

            if "opt_dict" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["opt_dict"])

            # Log recorder
            if "rec" in checkpoint.keys():
                rec = checkpoint["rec"]
            else:
                rec = init_recorder()

            best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0

            args.test_interval= int(args.epochs/20)
            for epoch in range(start_epoch, args.epochs):
                # train for one epoch
                if dst_train.data_type == "image":
                    train_img(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)
                else:
                    train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)
                # wandb.log({"acc": acc, "loss": loss})

                # evaluate on validation set
                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    prec1 = val_group(val_loader, network, criterion, epoch, args, rec)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1

                    if is_best:
                        best_prec1 = prec1
                        if args.save_path != "":
                            rec = record_ckpt(rec, epoch)
                            save_checkpoint({"exp": exp,
                                             "epoch": epoch + 1,
                                             "state_dict": network.state_dict(),
                                             "opt_dict": optimizer.state_dict(),
                                             "best_acc1": best_prec1,
                                             "rec": rec,
                                             "subset": subset},
                                            os.path.join(args.save_path, checkpoint_name + (
                                                "" if model == args.model else model + "_") + "unknown.ckpt"),
                                            epoch=epoch, prec=best_prec1)

                    test_prec1 = test_group(test_loader, network, criterion, epoch, args, rec)
                    
            test_prec1 = test_group(test_loader, network, criterion, epoch, args, rec)
            val_prec1 = val_group(test_loader, network, criterion, epoch, args, rec)
                # Prepare for the next checkpoint
            if args.save_path != "":
                try:
                    os.rename(
                        os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model + "_") +
                                    "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name +
                                    ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1))
                except:
                    save_checkpoint({"exp": exp,
                                    "epoch": args.epochs,
                                    "state_dict": network.state_dict(),
                                    "opt_dict": optimizer.state_dict(),
                                    "best_acc1": best_prec1,
                                    "rec": rec,
                                    "subset": subset},
                                    os.path.join(args.save_path, checkpoint_name +
                                                ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
                                    epoch=args.epochs - 1,
                                    prec=best_prec1)

            print('| Best accuracy: ', best_prec1, ", on model " + model if len(models) > 1 else "", end="\n\n")
            start_epoch = 0
            checkpoint = {}
            sleep(2)

        # wandb.finish()


if __name__ == '__main__':
    main()
