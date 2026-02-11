import torch
from transformers import Trainer


class CustomTrainer(Trainer):
    # custom trainer to handle BCEWithLogitsLoss with pos_weight for class imbalance
    
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        # BCEWithLogitsLoss handles the internal conversion to half if needed
        self.pos_weight = pos_weight.to(self.model.device) if pos_weight is not None else None

        if self.pos_weight is not None:
            self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        else:
            self.loss_fct = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # compute loss using BCEWithLogitsLoss
        labels = inputs.pop("labels")  # hide from model
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        device = logits.device
        dtype = logits.dtype
        
        # create loss function
        if self.pos_weight is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(device=device, dtype=dtype)
            )
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        
        loss = loss_fct(logits, labels.to(dtype))
        return (loss, outputs) if return_outputs else loss
