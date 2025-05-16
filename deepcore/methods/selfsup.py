from .earlytrain import EarlyTrain
import torch, time
from torch import nn
import numpy as np
from .. import nets
from sklearn.cluster import KMeans


# derived from https://github.com/naotoo1/Beyond-Neural-Scaling/blob/main/dataprune.py
def ssl_kmeans(embeddings, num_classes, init='k-means++', n_init='auto'):
    self_supervised_learning_model = KMeans(
        n_clusters=num_classes,
        init=init,
        n_init=n_init
    )
    self_supervised_learning_model.fit(embeddings)

    prototype_responsibilities = self_supervised_learning_model.fit_transform(
        embeddings
    )

    cluster_labels = self_supervised_learning_model.labels_
    cluster_centers = self_supervised_learning_model.cluster_centers_

    return prototype_responsibilities, cluster_labels, cluster_centers



class SelfSup(EarlyTrain):
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
        self.train_indx = np.arange(self.n_train)

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
        distance_matrix, _, _ = ssl_kmeans(features, len(np.unique(targets)))
        scores = np.min(distance_matrix, axis=1)

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
