""" This script intends to create a simple bert model
"""
import torch.nn as nn
from transformers import BertModel


class BertForSequenceClassification(nn.Module):

    def __init__(self, config, num_labels=2):
        """
        params:
            num_labels(int): binary classification problem
        """
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # init weight
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids,
                                     attention_mask, 
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
