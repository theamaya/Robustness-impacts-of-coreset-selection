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

import torchvision

# def count_label_group_pairs(labels, groups, subset_indices):
#     """
#     Count the number of samples belonging to each label and group pair.

#     Parameters:
#     - labels (array-like): Array of class labels for the entire dataset.
#     - groups (array-like): Array of group labels for the entire dataset.
#     - subset_indices (array-like): List of indices indicating the selected subset.

#     Returns:
#     - pair_counts (dict): Dictionary with (label, group) pairs as keys and their counts as values.
#     """
#     pair_counts = defaultdict(int)

#     for idx in subset_indices:
#         label = labels[idx]
#         group = groups[idx]
#         pair_counts[(label, group)] += 1

#     return pair_counts

def get_subset(subset_path=None, score_path=None, selection=None, fraction=None, policy=None):
    
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



def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--selection', type=str, default="uniform", help="selection method")
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
    parser.add_argument('--print_freq', '-p', default=20, type=int, help='print frequency (default: 20)')
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
                    checkpoint_name = "{dst}_level{lvl}_{net}_pretrained_{mtd}_frac{fr}_{policy}_exp{exp}_pretrainedScores".format(dst=args.dataset,
                                                                        lvl=args.level,
                                                                        net=args.model,
                                                                        mtd=args.selection,
                                                                        exp=exp,
                                                                        fr=args.fraction,
                                                                        policy=args.policy)
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

        torch.random.manual_seed(args.seed)
        print(checkpoint_name)
        print(vars(args))
        wandb.init(
        # set the wandb project where this run will be logged
        project="Bias_in_selection-debug",
        # track hyperparameters and run metadata
        config= vars(args),
        name= checkpoint_name)

        ###########
        subset = get_subset(args.subset_path, args.score_path, args.selection, args.fraction, args.policy)
        ###########

        # print(len(subset["indices"]))

        # # Augmentation
        if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
            dst_train.transform = transforms.Compose(
                [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
                 transforms.RandomHorizontalFlip(), dst_train.transform])
        elif args.dataset == "ImageNet":
            dst_train.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        # # Handle weighted subset
        # if_weighted = "weights" in subset.keys()
        # if if_weighted:
        #     dst_subset = WeightedSubset(dst_train, subset["indices"], subset["weights"])
        # else:
        #     dst_subset = torch.utils.data.Subset(dst_train, subset["indices"])


        if_weighted = False
        dst_subset = torch.utils.data.Subset(dst_train, subset)

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
                network= torchvision.models.resnet50(pretrained=True).cuda()
                d = network.fc.in_features
                network.fc = torch.nn.Linear(d, num_classes).cuda()

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

            # Save the checkpont with only the susbet.
            # if args.save_path != "" and args.resume == "":
            #     save_checkpoint({"exp": exp,
            #                      "subset": subset,
            #                      "sel_args": selection_args},
            #                     os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model
            #                                  + "_") + "unknown.ckpt"), 0, 0.)

            for epoch in range(start_epoch, args.epochs):
                # train for one epoch
                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)
                # wandb.log({"acc": acc, "loss": loss})

                # evaluate on validation set
                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    prec1 = val_group(val_loader, network, criterion, epoch, args, rec)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1

                    # if is_best:
                    #     best_prec1 = prec1
                    #     if args.save_path != "":
                    #         rec = record_ckpt(rec, epoch)
                    #         save_checkpoint({"exp": exp,
                    #                          "epoch": epoch + 1,
                    #                          "state_dict": network.state_dict(),
                    #                          "opt_dict": optimizer.state_dict(),
                    #                          "best_acc1": best_prec1,
                    #                          "rec": rec,
                    #                          "subset": subset},
                    #                         os.path.join(args.save_path, checkpoint_name + (
                    #                             "" if model == args.model else model + "_") + "unknown.ckpt"),
                    #                         epoch=epoch, prec=best_prec1)

                    test_prec1 = test_group(test_loader, network, criterion, epoch, args, rec)

            # Prepare for the next checkpoint
            # if args.save_path != "":
            #     try:
            #         os.rename(
            #             os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model + "_") +
            #                          "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name +
            #                          ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1))
            #     except:
            #         save_checkpoint({"exp": exp,
            #                          "epoch": args.epochs,
            #                          "state_dict": network.state_dict(),
            #                          "opt_dict": optimizer.state_dict(),
            #                          "best_acc1": best_prec1,
            #                          "rec": rec,
            #                          "subset": subset},
            #                         os.path.join(args.save_path, checkpoint_name +
            #                                      ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
            #                         epoch=args.epochs - 1,
            #                         prec=best_prec1)

            print('| Best accuracy: ', best_prec1, ", on model " + model if len(models) > 1 else "", end="\n\n")
            start_epoch = 0
            checkpoint = {}
            sleep(2)

        wandb.finish()


if __name__ == '__main__':
    main()
