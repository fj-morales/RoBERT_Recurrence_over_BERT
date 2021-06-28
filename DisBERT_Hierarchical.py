import torch
import torch.nn as nn
import transformers


class DisBERT_Hierarchical_Model(nn.Module):

    def __init__(self, pooling_method="mean"):
        super(DisBERT_Hierarchical_Model, self).__init__()

        self.pooling_method = pooling_method

        # self.bert_path = 'bert-base-uncased'
        self.bert_path = 'distilbert-base-uncased'
        self.bert = transformers.DistilBertModel.from_pretrained(self.bert_path)
        self.out = nn.Linear(768, 3)

    def forward(self, ids, mask, token_type_ids, lengt):

        # _, pooled_out = self.bert(
        #     ids, attention_mask=mask)

        # pooled_out = self.bert(
        #     ids, attention_mask=mask)
        hidden_state = self.bert(
            ids, attention_mask=mask)[0]
        pooled_out = hidden_state[:, 0]

        chunks_emb = pooled_out.split_with_sizes(lengt)

        if self.pooling_method == "mean":
            emb_pool = torch.stack([torch.mean(x, 0) for x in chunks_emb])
        elif self.pooling_method == "max":
            emb_pool = torch.stack([torch.max(x, 0)[0] for x in chunks_emb])

        return self.out(emb_pool)
