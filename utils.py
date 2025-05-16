import time, torch
from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
import wandb
from collections import defaultdict

from sklearn.metrics import roc_auc_score
from collections import defaultdict
import time
import torch
import wandb

def val_group_auroc(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    # Data storage for AUROC calculations
    true_labels = []
    predicted_scores = []

    class_scores = defaultdict(list)
    class_labels = defaultdict(list)

    group_scores = defaultdict(list)
    group_labels = defaultdict(list)

    subgroup_scores = defaultdict(list)
    subgroup_labels = defaultdict(list)

    end = time.time()
    for i, contents in enumerate(test_loader):
        input = contents[0].to(args.device)
        target = contents[1].to(args.device)
        group = contents[2].to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)
            loss = criterion(output, target).mean()

        # Convert output to probabilities
        if output.shape[1] > 1:  # Multi-class case
            probabilities = torch.softmax(output, dim=1)[:, 1]  # Assuming class 1 is positive
        else:  # Binary classification case
            probabilities = torch.sigmoid(output).squeeze()

        # Collect labels and scores for AUROC
        true_labels.extend(target.cpu().numpy())
        predicted_scores.extend(probabilities.cpu().numpy())

        for lbl, score, grp in zip(target.cpu().numpy(), probabilities.cpu().numpy(), group.cpu().numpy()):
            class_scores[lbl].append(score)
            class_labels[lbl].append(lbl)

            group_scores[grp].append(score)
            group_labels[grp].append(lbl)

            subgroup = (lbl, grp)
            subgroup_scores[subgroup].append(score)
            subgroup_labels[subgroup].append(lbl)

        # Measure loss
        losses.update(loss.data.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(test_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

    # Compute AUROC scores
    overall_auroc = roc_auc_score(true_labels, predicted_scores)

    class_aurocs = {
        cls: roc_auc_score(class_labels[cls], class_scores[cls]) for cls in class_labels 
    }

    group_aurocs = {
        grp: roc_auc_score(group_labels[grp], group_scores[grp]) for grp in group_labels #if len(set(group_labels[grp])) > 1
    }

    subgroup_aurocs = {
        subgrp: roc_auc_score(subgroup_labels[subgrp], subgroup_scores[subgrp]) 
        for subgrp in subgroup_labels #if len(set(subgroup_labels[subgrp])) > 1
    }

    # Log results to wandb
    wandb.log({
        'val_overall_auroc': overall_auroc,
        'val_class_aurocs': {str(cls): auc for cls, auc in class_aurocs.items()},
        'val_group_aurocs': {str(grp): auc for grp, auc in group_aurocs.items()},
        'val_subgroup_aurocs': {str(subgrp): auc for subgrp, auc in subgroup_aurocs.items()}
    })

    print(f' * Overall AUROC: {overall_auroc:.4f}')
    print(f' * Class-level AUROC: {class_aurocs}')
    print(f' * Group-level AUROC: {group_aurocs}')
    print(f' * Subgroup-level AUROC: {subgroup_aurocs}')

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, overall_auroc)
    return overall_auroc
    


def test_group_auroc(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    # Data storage for AUROC calculations
    true_labels = []
    predicted_scores = []

    class_scores = defaultdict(list)
    class_labels = defaultdict(list)

    group_scores = defaultdict(list)
    group_labels = defaultdict(list)

    subgroup_scores = defaultdict(list)
    subgroup_labels = defaultdict(list)

    end = time.time()
    for i, contents in enumerate(test_loader):
        input = contents[0].to(args.device)
        target = contents[1].to(args.device)
        group = contents[2].to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)
            loss = criterion(output, target).mean()

        # Convert output to probabilities
        if output.shape[1] > 1:  # Multi-class case
            probabilities = torch.softmax(output, dim=1)[:, 1]  # Assuming class 1 is positive
        else:  # Binary classification case
            probabilities = torch.sigmoid(output).squeeze()

        # Collect labels and scores for AUROC
        true_labels.extend(target.cpu().numpy())
        predicted_scores.extend(probabilities.cpu().numpy())

        for lbl, score, grp in zip(target.cpu().numpy(), probabilities.cpu().numpy(), group.cpu().numpy()):
            class_scores[lbl].append(score)
            class_labels[lbl].append(lbl)

            group_scores[grp].append(score)
            group_labels[grp].append(lbl)

            subgroup = (lbl, grp)
            subgroup_scores[subgroup].append(score)
            subgroup_labels[subgroup].append(lbl)

        # Measure loss
        losses.update(loss.data.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(test_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

    # Compute AUROC scores
    overall_auroc = roc_auc_score(true_labels, predicted_scores)

    class_aurocs = {
        cls: roc_auc_score(class_labels[cls], class_scores[cls]) for cls in class_labels if len(set(class_labels[cls])) > 1
    }

    group_aurocs = {
        grp: roc_auc_score(group_labels[grp], group_scores[grp]) for grp in group_labels if len(set(group_labels[grp])) > 1
    }

    subgroup_aurocs = {
        subgrp: roc_auc_score(subgroup_labels[subgrp], subgroup_scores[subgrp]) 
        for subgrp in subgroup_labels if len(set(subgroup_labels[subgrp])) > 1
    }

    # Log results to wandb
    wandb.log({
        'overall_auroc': overall_auroc,
        'class_aurocs': {str(cls): auc for cls, auc in class_aurocs.items()},
        'group_aurocs': {str(grp): auc for grp, auc in group_aurocs.items()},
        'subgroup_aurocs': {str(subgrp): auc for subgrp, auc in subgroup_aurocs.items()}
    })

    print(f' * Overall AUROC: {overall_auroc:.4f}')
    print(f' * Class-level AUROC: {class_aurocs}')
    print(f' * Group-level AUROC: {group_aurocs}')
    print(f' * Subgroup-level AUROC: {subgroup_aurocs}')

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, overall_auroc)
    return overall_auroc



class WeightedSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        assert len(indices) == len(weights)
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]], self.weights[[i for i in idx]]
        return self.dataset[self.indices[idx]], self.weights[idx]

# def train_mixed_prec():
#     scaler = torch.cuda.amp.GradScaler()
#     for i, (inputs, targets, _, _) in enumerate(tqdm(train_loader)):
#         inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
#         inputs = inputs.contiguous(memory_format=torch.channels_last)
#         # Forward propagation, compute loss, get predictions
#         with torch.autocast(device_type='cuda', dtype=torch.float16):
#             self.model_optimizer.zero_grad()
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, targets)
#         self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)
        
#         # Update loss, backward propagate, update optimizer
#         loss = loss.mean()
        
#         total = targets.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         correct = predicted.eq(targets.data).cpu().sum()
#         wandb.log({'Epoch': epoch,
#                 'Train_Loss': loss.item(),
#                 'Train_Acc@1': 100. * correct.item() / total})
#         self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)
#         scaler.scale(loss).backward()
#         scaler.step(self.model_optimizer)#.step()
#         scaler.update()


def train_img(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted: bool = False): ## training with mixed precision
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            target = contents[0][1].to(args.device)
            input = contents[0][0].to(args.device)

            # Compute output
            output = network(input)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = contents[1].to(args.device)
            input = contents[0].to(args.device)
            input = input.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                optimizer.zero_grad()
                output = network(input)
                loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Compute gradient and do SGD step
        # loss.backward()
        # optimizer.step()
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)#.step()
        scaler.update()
        if scheduler != None:
            scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
            
    wandb.log({'Epoch': epoch,
                  'Train_Loss': losses.avg,
                  'Train_Prec@1': top1.avg})

    record_train_stats(rec, epoch, losses.avg, top1.avg, optimizer.state_dict()['param_groups'][0]['lr'])


def train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted: bool = False):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    network.train()

    end = time.time()
    for i, contents in enumerate(train_loader):
        optimizer.zero_grad()
        if if_weighted:
            target = contents[0][1].to(args.device)
            input = contents[0][0].to(args.device)

            # Compute output
            output = network(input)
            weights = contents[1].to(args.device).requires_grad_(False)
            loss = torch.sum(criterion(output, target) * weights) / torch.sum(weights)
        else:
            target = contents[1].to(args.device)
            input = contents[0].to(args.device)

            # Compute output
            output = network(input)
            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
            
    wandb.log({'Epoch': epoch,
                  'Train_Loss': losses.avg,
                  'Train_Prec@1': top1.avg})

    record_train_stats(rec, epoch, losses.avg, top1.avg, optimizer.state_dict()['param_groups'][0]['lr'])


def test(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    # for i, (input, target) in enumerate(test_loader):
    for i, contents in enumerate(test_loader):
        input= contents[0].to(args.device)
        target= contents[1].to(args.device)
        # target = target.to(args.device)
        # input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)
            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))
            
    wandb.log({"test_acc": top1.avg, "test_loss": losses.avg})

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    return top1.avg

def val_group(val_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    total_samples = 0
    correct_predictions = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    group_correct = defaultdict(int)
    group_total = defaultdict(int)

    end = time.time()
    # for i, (input, target) in enumerate(test_loader):
    for i, contents in enumerate(val_loader):
        input= contents[0].to(args.device)
        target= contents[1].to(args.device)
        group = contents[2].to(args.device)
        # target = target.to(args.device)
        # input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)
            loss = criterion(output, target).mean()

        _, predicted = torch.max(output.data, 1)
        total_samples += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        for label, prediction, group in zip(target, predicted, group):
            if label == prediction:
                class_correct[label.item()] += 1
                group_correct[group.item()] += 1
            class_total[label.item()] += 1
            group_total[group.item()] += 1

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))
            
        overall_accuracy = correct_predictions / total_samples

    class_accuracies = {cls: class_correct[cls] / class_total[cls] for cls in class_total}
    group_accuracies = {grp: group_correct[grp] / group_total[grp] for grp in group_total}

    # Log the results to wandb
    wandb.log({
        'Validation_accuracy': overall_accuracy,
        'class_val_accuracies': {str(cls): acc for cls, acc in class_accuracies.items()},
        'group_val_accuracies': {str(grp): acc for grp, acc in group_accuracies.items()}
    })
            
    # wandb.log({"test_acc": top1.avg, "test_loss": losses.avg})

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    return top1.avg


def test_group(test_loader, network, criterion, epoch, args, rec):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    total_samples = 0
    correct_predictions = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    group_correct = defaultdict(int)
    group_total = defaultdict(int)

    subgroup_correct = defaultdict(int)
    subgroup_total = defaultdict(int)

    end = time.time()
    # for i, (input, target) in enumerate(test_loader):
    for i, contents in enumerate(test_loader):
        input= contents[0].to(args.device)
        target= contents[1].to(args.device)
        group = contents[2].to(args.device)
        # target = target.to(args.device)
        # input = input.to(args.device)

        # Compute output
        with torch.no_grad():
            output = network(input)
            loss = criterion(output, target).mean()

        _, predicted = torch.max(output.data, 1)
        total_samples += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        for label, prediction, group in zip(target, predicted, group):
            subgroup = (label.item(), group.item())
            if label == prediction:
                subgroup_correct[subgroup] += 1
            subgroup_total[subgroup] += 1

            if label == prediction:
                class_correct[label.item()] += 1
                group_correct[group.item()] += 1
            class_total[label.item()] += 1
            group_total[group.item()] += 1

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))
            
    overall_accuracy = correct_predictions / total_samples

    class_accuracies = {cls: class_correct[cls] / class_total[cls] for cls in class_total}
    group_accuracies = {grp: group_correct[grp] / group_total[grp] for grp in group_total}
    subgroup_accuracies = {subgrp: subgroup_correct[subgrp] / subgroup_total[subgrp] 
                           for subgrp in subgroup_total}

    # Log the results to wandb
    wandb.log({
        'overall_accuracy': overall_accuracy,
        'class_accuracies': {str(cls): acc for cls, acc in class_accuracies.items()},
        'group_accuracies': {str(grp): acc for grp, acc in group_accuracies.items()},
        'subgroup_accuracies': {str(subgrp): acc for subgrp, acc in subgroup_accuracies.items()}
    })

            
    # wandb.log({"test_acc": top1.avg, "test_loss": losses.avg})

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    network.no_grad = False

    record_test_stats(rec, epoch, losses.avg, top1.avg)
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def save_checkpoint(state, path, epoch, prec):
    print("=> Saving checkpoint for epoch %d, with Prec@1 %f." % (epoch, prec))
    torch.save(state, path)


def init_recorder():
    from types import SimpleNamespace
    rec = SimpleNamespace()
    rec.train_step = []
    rec.train_loss = []
    rec.train_acc = []
    rec.lr = []
    rec.test_step = []
    rec.test_loss = []
    rec.test_acc = []
    rec.ckpts = []
    return rec


def record_train_stats(rec, step, loss, acc, lr):
    rec.train_step.append(step)
    rec.train_loss.append(loss)
    rec.train_acc.append(acc)
    rec.lr.append(lr)
    return rec


def record_test_stats(rec, step, loss, acc):
    rec.test_step.append(step)
    rec.test_loss.append(loss)
    rec.test_acc.append(acc)
    return rec


def record_ckpt(rec, step):
    rec.ckpts.append(step)
    return rec


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
