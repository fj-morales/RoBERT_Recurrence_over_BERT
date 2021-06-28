import torch.nn as nn
import transformers


class DisBert_Classification_Model(nn.Module):
    """ A Model for bert fine tuning """

    def __init__(self):
        super(DisBert_Classification_Model, self).__init__()
        # self.bert_path = 'bert-base-uncased'
        self.bert_path = 'distilbert-base-uncased'
        # self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert = transformers.DistilBertModel.from_pretrained(self.bert_path)
        # self.bert_drop=nn.Dropout(0.2)
        # self.fc=nn.Linear(768,256)
        # self.out=nn.Linear(256,10)
        self.out = nn.Linear(768, 3)
        # self.relu=nn.ReLU()

    def forward(self, ids, mask, token_type_ids):
        """ Define how to perfom each call

        Parameters
        __________
        ids: array
            -
        mask: array
            -
        token_type_ids: array
            -

        Returns
        _______
            -
        """

        # _, pooled_out = self.bert(
        #     ids, attention_mask=mask, return_dict=False)
        hidden_state = self.bert(
            ids, attention_mask=mask)[0]
        pooled_out = hidden_state[:, 0]
        # rh=self.bert_drop(pooled_out)
        # rh=self.fc(rh)
        # rh=self.relu(rh)
        return self.out(pooled_out)
