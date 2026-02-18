import torch

from transformers import Trainer


class CustomTrainer(Trainer):
    # custom trainer to handle bce loss with pos_weight (or pre-computed class weights) and optional label masking for multi-label classification
    
    def __init__(self, *args, pos_weight=None, eval_thresholds=None, **kwargs):
        # extract compute_metrics before calling super().__init__
        self.compute_metrics_func = kwargs.get('compute_metrics')
        
        super().__init__(*args, **kwargs)
        # BCEWithLogitsLoss handles the internal conversion to half if needed
        self.pos_weight = pos_weight.to(self.model.device) if pos_weight is not None else None
        self.eval_thresholds = eval_thresholds  # custom thresholds dict for evaluation (label -> threshold)

        if self.pos_weight is not None:
            self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

        else:
            self.loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # compute loss with optional label masking for multi-label classification

        labels = inputs.pop("labels")  # hide from model
        label_mask = inputs.pop("label_mask", None)  # extract mask if present
        
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            # support models that return dicts (e.g. with 'logits' key) or objects with attributes
            logits = outputs.get("logits")

        else:
            logits = outputs.logits
        
        device = logits.device
        dtype = logits.dtype
        
        # create loss function with reduction='none' to apply per-element masking
        if self.pos_weight is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(device=device, dtype=dtype),
                reduction='none'
            )
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        # compute per-element loss
        loss_per_element = loss_fct(logits, labels.to(dtype))
        
        # apply label masking if available
        if label_mask is not None:
            # label_mask: [batch_size, num_labels] with 1 where label should be evaluated, 0 otherwise
            label_mask = label_mask.to(dtype=dtype, device=device)
            
            # zero out loss for masked labels
            loss_per_element = loss_per_element * label_mask
            
            # avg loss only over non-masked labels
            # compute number of valid labels per sample
            num_valid_labels = label_mask.sum(dim=1, keepdim=True)
            num_valid_labels = torch.clamp(num_valid_labels, min=1.0)  # avoid division by zero
            
            # sum loss per sample and divide by number of valid labels
            loss_per_sample = loss_per_element.sum(dim=1) / num_valid_labels.squeeze(1)
            loss = loss_per_sample.mean()

        else:
            # no masking, use standard mean reduction
            loss = loss_per_element.mean()

        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(self, eval_pred):
        # override compute_metrics to pass custom thresholds if set
        if self.compute_metrics_func is not None:
            if self.eval_thresholds is not None:
                return self.compute_metrics_func(eval_pred, thresholds=self.eval_thresholds)
            else:
                # try calling with thresholds parameter, fall back to no parameter if not supported
                try:
                    return self.compute_metrics_func(eval_pred, thresholds=None)
                except TypeError:
                    return self.compute_metrics_func(eval_pred)
        return {}

