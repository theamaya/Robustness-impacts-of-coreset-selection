from .earlytrain import EarlyTrain
import torch, time
from torch import nn
import numpy as np
from .. import nets


# derived from https://github.com/tmllab/2023_ICLR_Moderate-DS/blob/main/selection.py
def get_median(features, targets):
    # get the median feature vector of each class
    num_classes = len(np.unique(targets, axis=0))
    prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)
    
    for i in range(num_classes):
        prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
    return prot

def get_distance(features, labels):
    print('features shape: ', features.shape)
    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))
    
    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels==i).nonzero()[0], :] = prots[i]
    print('features shape: ', features.shape)
    print('prots_for_each_examples shape: ', prots_for_each_example.shape)
    features= np.array(features, dtype=np.float64)
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)
    
    return distance


class supProto(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=True,
                 dst_test=None, precalcfeature= None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model=specific_model,
                         dst_test=dst_test, **kwargs)

        self.balance = balance
        self.precalcfeature= precalcfeature
        self.train_indx = np.arange(self.n_train)

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        assert self.torchvision_pretrain != False, "moderate coreset requires pre-trained models!"
        # Setup model and loss
        self.model = nets.__dict__[self.args.model if self.specific_model is None else self.specific_model](
            self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
            pretrained=self.torchvision_pretrain,
            im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size, linear_probe= self.linear_probe).to(self.args.device)

        if self.args.device == "cpu":
            print("Using CPU.")
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu[0])
            self.model = nets.nets_utils.MyDataParallel(self.model, device_ids=self.args.gpu)
        elif torch.cuda.device_count() > 1:
            self.model = nets.nets_utils.MyDataParallel(self.model).cuda()
        self.emb_dim = self.model.get_last_layer().in_features
        return self.construct_matrix()

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).to(self.args.device)
                labels = []

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                            torch.utils.data.Subset(self.dst_train, index),
                                            batch_size=self.args.selection_batch,
                                            num_workers=self.args.workers,
                                            shuffle=False)

                for i, (inputs, target, _, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = self.model.embedding_recorder.embedding
                    labels += target.tolist()

        self.model.no_grad = False
        return matrix.detach().cpu().numpy(), np.array(labels)

    def select(self, **kwargs):
        if self.precalcfeature is not None:
            features, targets = self.precalcfeature
        else:
            features, targets = self.run()

        scores = get_distance(features, targets)

        if not self.balance:
            top_examples = self.train_indx[np.argsort(scores)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples,
                                    c_indx[np.argsort(scores[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": scores}

class Moderate2(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=True,
                 dst_test=None, precalcfeature= None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model=specific_model,
                         dst_test=dst_test, **kwargs)

        self.balance = balance
        self.precalcfeature= precalcfeature
        self.train_indx = np.arange(self.n_train)

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        assert self.torchvision_pretrain != False, "moderate coreset requires pre-trained models!"
        # Setup model and loss
        self.model = nets.__dict__[self.args.model if self.specific_model is None else self.specific_model](
            self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
            pretrained=self.torchvision_pretrain,
            im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size, linear_probe= self.linear_probe).to(self.args.device)

        if self.args.device == "cpu":
            print("Using CPU.")
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu[0])
            self.model = nets.nets_utils.MyDataParallel(self.model, device_ids=self.args.gpu)
        elif torch.cuda.device_count() > 1:
            self.model = nets.nets_utils.MyDataParallel(self.model).cuda()
        self.emb_dim = self.model.get_last_layer().in_features
        return self.construct_matrix()

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).to(self.args.device)
                labels = []

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                            torch.utils.data.Subset(self.dst_train, index),
                                            batch_size=self.args.selection_batch,
                                            num_workers=self.args.workers,
                                            shuffle=False)

                for i, (inputs, target, _, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = self.model.embedding_recorder.embedding
                    labels += target.tolist()

        self.model.no_grad = False
        return matrix.detach().cpu().numpy(), np.array(labels)

    def select(self, **kwargs):
        if self.precalcfeature is not None:
            features, targets = self.precalcfeature
        else:
            features, targets = self.run()

        scores = get_distance(features, targets)

        ##########
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]

        # Step 2: Find the center index of the sorted array
        center_index = len(sorted_scores) // 2

        # Step 3: Calculate new scores based on proximity to the center index
        rescored_values = np.abs(np.arange(len(sorted_scores)) - center_index)

        # Step 4: Map rescored values back to the original order
        new_scores = np.empty_like(rescored_values)
        new_scores[sorted_indices] = rescored_values  # Reorder to match original indices
        ###########

        if not self.balance:
            top_examples = self.train_indx[np.argsort(scores)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples,
                                    c_indx[np.argsort(scores[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": new_scores}