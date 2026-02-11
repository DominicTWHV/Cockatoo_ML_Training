import torch

from transformers import AutoTokenizer

from cockatoo_ml.registry import ModelConfig
from cockatoo_ml.logger.context import data_processing_logger as logger


def get_tokenizer(model_name=None):
    # load tokenizer from model
    if model_name is None:
        model_name = ModelConfig.BASE_MODEL_NAME
    return AutoTokenizer.from_pretrained(model_name)


def create_preprocess_function(tokenizer, max_length=None):
    # use the tokenizer to preprocess the text for training
    if max_length is None:
        max_length = ModelConfig.MAX_TOKEN_LENGTH
        
    def preprocess(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        tokenized['labels'] = torch.tensor(examples['labels'], dtype=torch.float)
        return tokenized
    
    return preprocess


def tokenize_dataset(dataset, tokenizer, max_length=None):
    # tokenize the dataset using the provided tokenizer and max length (wrapper)
    if max_length is None:
        max_length = ModelConfig.MAX_TOKEN_LENGTH
        
    preprocess = create_preprocess_function(tokenizer, max_length)
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=['text'])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return tokenized_dataset
