from transformers import BertModel, DistilBertModel, GPT2Model, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder

from transformers import AlbertForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import DebertaV2ForSequenceClassification
import types
import torch

class BertFeatureWrapper(torch.nn.Module):

    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self.n_outputs = model.config.hidden_size
        classifier_dropout = (
            hparams['last_layer_dropout'] if hparams['last_layer_dropout'] != 0. else model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        kwargs = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1]
        }
        if x.shape[-1] == 3:
            kwargs['token_type_ids'] = x[:, :, 2]
        output = self.model(**kwargs)
        if hasattr(output, 'pooler_output'):
            return self.dropout(output.pooler_output)
        else:
            return self.dropout(output.last_hidden_state[:, 0, :])

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

def Featurizer(data_type, input_shape, hparams):
    text_model = BertModel.from_pretrained(hparams['text_arch'])
    return BertFeatureWrapper(text_model, hparams)





def _bert_replace_fc(model, record_embedding):
    model.fc = model.classifier
    delattr(model, "classifier")
    model.embedding_recorder = EmbeddingRecorder(record_embedding)

    def classifier(self, x):
        out = self.embedding_recorder(x)
        out = self.fc(x)
        return out
    
    model.classifier = types.MethodType(classifier, model)

    model.base_forward = model.forward

    def forward(self, x):
        return self.base_forward(input_ids=x[:, :, 0],attention_mask=x[:, :, 1],token_type_ids=x[:, :, 2]).logits

    model.forward = types.MethodType(forward, model)

    def get_last_layer(self):
        return self.fc

    model.get_last_layer = types.MethodType(get_last_layer, model)

    return model


def Bert(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False, linear_probe: bool = False):
	return _bert_replace_fc(BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes), record_embedding)


