from .earlytrain import EarlyTrain
import torch, time
from torch import nn
import numpy as np
from tqdm import tqdm


# Acknowledgement to
# https://github.com/mtoneva/example_forgetting

class EL2N(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=True,
                 dst_test=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model=specific_model,
                         dst_test=dst_test, **kwargs)

        self.balance = balance
        self.predictions = None
        self.labels = None
        self.n_train = len(dst_train)
        self.score_tracker= np.zeros((self.epochs, self.n_train))

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        probs = torch.nn.functional.softmax(outputs.detach(), dim=1)
        targets= torch.nn.functional.one_hot(
                        targets.detach(),
                        num_classes=probs.shape[1],
                    ).float()
        scores = torch.sum((probs - targets) ** 2, dim=1).cpu().numpy()
        self.score_tracker[epoch, batch_inds]= scores

    def finish_run(self):
        train_loader = torch.utils.data.DataLoader(self.dst_pretrain_dict['dst_train'] if self.if_dst_pretrain
                                                   else self.dst_train, shuffle=False,
                                                   num_workers=self.args.workers, pin_memory=True)
        predictions = []
        labels = []
        with torch.no_grad():
            for i, (inputs, targets, _, _) in tqdm(enumerate(train_loader)):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.model(inputs)
                predictions.append(outputs.detach().cpu())
                labels.append(targets.detach().cpu())
        predictions = torch.vstack(predictions)
        labels = torch.hstack(labels)
        self.predictions = predictions
        self.labels = labels

    def select(self, **kwargs):
        self.run()
        probs = torch.nn.functional.softmax(self.predictions, dim=1)
        targets = torch.nn.functional.one_hot(
                        self.labels,
                        num_classes=probs.shape[1],
                    ).float()
        scores = torch.sum((probs - targets) ** 2, dim=1)
        if not self.balance:
            top_examples = self.train_indx[np.argsort(scores.numpy())][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples,
                                    c_indx[np.argsort(scores[c_indx].numpy())[::-1][:budget]])

        return {"indices": top_examples, "scores": scores, "score_tracker": self.score_tracker}
