import torch

from transformers import AutoTokenizer, CLIPProcessor
from PIL import Image

from cockatoo_ml.registry import ModelConfig, ModelType
from cockatoo_ml.logger.context import data_processing_logger as logger


def get_tokenizer(model_name=None, model_type=None):
    # load tokenizer or processor based on ModelConfig.MODEL_TYPE
    # For CLIP: returns CLIPProcessor
    # For DeBERTa: returns AutoTokenizer

    if model_type is None:
        model_type = ModelConfig.MODEL_TYPE
        
    if model_name is None:
        model_name = ModelConfig.get_base_model_name()
    
    if model_type == ModelType.CLIP_VIT:
        logger.info(f"Loading CLIP processor for {model_name}")
        return CLIPProcessor.from_pretrained(model_name)
    
    elif model_type == ModelType.DEBERTA:
        logger.info(f"Loading tokenizer for {model_name}")
        return AutoTokenizer.from_pretrained(model_name)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_preprocess_function(tokenizer, max_length=None, model_type=None):
    # create preprocessing function based on model type.
    # CLIP: processes both text and optional images
    # DeBERTa: processes text only

    if model_type is None:
        model_type = ModelConfig.MODEL_TYPE
        
    if max_length is None:
        max_length = ModelConfig.get_max_token_length()
    
    if model_type == ModelType.CLIP_VIT:
        # CLIP preprocessing (text + optional images)
        def preprocess_clip(examples):
            # process text
            text_inputs = tokenizer(
                text=examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            result = {
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask'],
                'labels': torch.tensor(examples['labels'], dtype=torch.float)
            }
            
            # process images if available
            if 'image' in examples and examples['image'][0] is not None:
                # handle images if dataset contains them
                images = []
                for img in examples['image']:
                    if img is not None:
                        if not isinstance(img, Image.Image):
                            # if needed, convert to pil image
                            img = Image.fromarray(img) if hasattr(img, 'shape') else img

                        images.append(img)

                    else:
                        # create blank image if None
                        images.append(Image.new('RGB', (ModelConfig.CLIP_IMAGE_SIZE, ModelConfig.CLIP_IMAGE_SIZE)))
                
                image_inputs = tokenizer(images=images, return_tensors='pt')
                result['pixel_values'] = image_inputs['pixel_values']
            
            return result
        
        return preprocess_clip
        
    elif model_type == ModelType.DEBERTA:
        # DeBERTa preprocessing (text only)
        def preprocess_deberta(examples):
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length
            )
            tokenized['labels'] = torch.tensor(examples['labels'], dtype=torch.float)
            return tokenized
        
        return preprocess_deberta
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def tokenize_dataset(dataset, tokenizer, max_length=None, model_type=None):
    # high level orchestrator function to tokenize dataset based on model type
    if model_type is None:
        model_type = ModelConfig.MODEL_TYPE
        
    if max_length is None: # failsafe in case max_length not provided for whatever reason
        max_length = ModelConfig.get_max_token_length()
        
    preprocess = create_preprocess_function(tokenizer, max_length, model_type)
    
    logger.info(f"Tokenizing dataset for {model_type}...")
    
    # columns to keep depend on model type
    if model_type == ModelType.CLIP_VIT:
        # keep image column if it exists
        remove_cols = ['text']

        if 'image' not in dataset['train'].column_names:
            logger.info("No image column found, using text-only mode for CLIP. This is not recommended for accuracy. Please consider DeBERTa for text-only datasets.")

    else:
        remove_cols = ['text']
    
    tokenized_dataset = dataset.map(
        preprocess, 
        batched=True, 
        remove_columns=remove_cols
    )
    
    # Set format based on model type
    if model_type == ModelType.CLIP_VIT:
        format_cols = ['input_ids', 'attention_mask', 'labels']

        if 'pixel_values' in tokenized_dataset['train'].column_names:
            format_cols.append('pixel_values')
            
        tokenized_dataset.set_format(type='torch', columns=format_cols)

    else:
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return tokenized_dataset
