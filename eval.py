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
    parser.add_argument('--test_only', type= str_to_bool, default= True)
    parser.add_argument('--test_split', type= str, default= 'test')
    # parser.add_argument('--only_fg', type= str_to_bool, default= False)


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
        (args.data_path, args.test_split)
        
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

        torch.random.manual_seed(args.seed)
        print(checkpoint_name)
        print(vars(args))
        wandb.init(
        # set the wandb project where this run will be logged
        project="Bias_in_selection-training-classequal(imagenet)",
        # track hyperparameters and run metadata
        config= vars(args),
        name= checkpoint_name)

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

        # BackgroundGenerator for ImageNet to speed up dataloaders
        if args.dataset == "ImageNet":
            test_loader = DataLoaderX(dst_test, batch_size=args.train_batch, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)
        else:
            val_loader = torch.utils.data.DataLoader(dst_val, batch_size=args.train_batch, shuffle=False,
                                                       num_workers=args.workers, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
                                                      num_workers=args.workers, pin_memory=True)
        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

        # Listing cross-architecture experiment settings if specified.
        models = [args.model]
        if isinstance(args.cross, list):
            for model in args.cross:
                if model != args.model:
                    models.append(model)

        epoch= 200
        rec = init_recorder()
        for model in models:
            network = nets.__dict__[model](channel, num_classes, im_size, pretrained=args.pretrain,linear_probe= args.linear_probe).to(args.device)

            if args.device == "cpu":
                print("Using CPU.")
            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu[0])
                network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
            elif torch.cuda.device_count() > 1:
                network = nets.nets_utils.MyDataParallel(network).cuda()

            available_checkpoints= os.listdir(args.save_path)
            for checkpoint_file in available_checkpoints:
                if checkpoint_name in checkpoint_file:
                    print("checkpoint file:", checkpoint_file)
                    checkpoint= torch.load(os.path.join(args.save_path, checkpoint_file), map_location= args.device)
                # else:
            # print("checkpoint file found:", checkpoint_file)

            if "state_dict" in checkpoint.keys():
                # Loading model state_dict
                network.load_state_dict(checkpoint["state_dict"])


            # test_prec1 = test_group(test_loader, network, criterion, epoch, args, rec)

            # test_auroc = test_group_auroc(test_loader, network, criterion, epoch, args, rec)
            val_auroc = val_group_auroc(val_loader, network, criterion, epoch, args, rec)

            sleep(2)

        wandb.finish()


if __name__ == '__main__':
    main()
