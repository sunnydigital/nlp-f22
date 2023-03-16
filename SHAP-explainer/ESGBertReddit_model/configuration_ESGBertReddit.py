from transformers import PretrainedConfig
class ESGRedditConfig(PretrainedConfig):
    model_type = "ESGBertReddit"

    def __init__(
        self,
        architectures = ["BertForSequenceClassification"],
        num_classes: int = 4,
        **kwargs
    ):  
        self.architectures = architectures
        self.num_classes = num_classes
        super().__init__(**kwargs)