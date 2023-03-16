import torch
import torch.nn as nn
from transformers import PreTrainedModel,BertModel
from ESGBertReddit_model.configuration_ESGBertReddit import ESGRedditConfig
class ClassificationModel(PreTrainedModel):
    config_class = ESGRedditConfig

    def __init__(self,config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained('yiyanghkust/finbert-esg',output_attentions=True)
        self.W = nn.Linear(self.bert.config.hidden_size, config.num_classes)
        self.num_classes = config.num_classes
        
    def forward(self,input_ids,attention_mask,token_type_ids,**kw):
        h, _, attn = self.bert(input_ids=input_ids, 
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids).values()
        h_cls = h[:,0,:]
        output = self.W(h_cls)
        return output, attn

class BertModelForESGClassification(PreTrainedModel):
    config_class = ESGRedditConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = ClassificationModel(config)

    def forward(self,**inputs):
        logits,_ = self.model(**inputs)
        if "labels" in inputs:
            loss = torch.nn.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}