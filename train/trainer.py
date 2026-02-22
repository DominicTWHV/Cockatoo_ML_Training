import torch

from transformers import Trainer

from cockatoo_ml.registry import TrainingConfig
from cockatoo_ml.logger.context import model_training_logger as logger

from train.losses import AsymmetricLoss, BCEWithLogitsLoss


class CustomTrainer(Trainer):
    # custom trainer to handle bce loss with pos_weight (or pre-computed class weights) and optional label masking for multi-label classification
    
    def __init__(self, *args, pos_weight=None, eval_thresholds=None, **kwargs):
        # extract compute_metrics before calling super().__init__
        self.compute_metrics_func = kwargs.get('compute_metrics')
        
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight.to(self.model.device) if pos_weight is not None else None
        self.eval_thresholds = eval_thresholds  # custom thresholds dict for evaluation (label -> threshold)
        self.loss_fn_name = getattr(TrainingConfig, 'LOSS_FUNCTION', 'bce').lower()

        # pre-build a static loss instance
        if self.loss_fn_name == 'asl':
            self.loss_fct = AsymmetricLoss(
                gamma_neg=getattr(TrainingConfig, 'ASL_GAMMA_NEG', 4),
                gamma_pos=getattr(TrainingConfig, 'ASL_GAMMA_POS', 1),
                clip=getattr(TrainingConfig, 'ASL_CLIP', 0.05),
                reduction='none',
            )

        else:
            # default to BCEWithLogitsLoss — pos_weight applied in compute_loss per-device
            self.loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

    def create_optimizer(self):
        # create optimizer according to registry TrainingConfig
        if getattr(self, 'optimizer', None) is not None:
            return self.optimizer

        opt_name = getattr(TrainingConfig, 'OPTIMIZER', 'adamw_8bit')
        lr = getattr(self.args, 'learning_rate', None) or getattr(self.args, 'lr', None) or 1e-5
        weight_decay = getattr(TrainingConfig, 'WEIGHT_DECAY', 0.0)
        betas = getattr(TrainingConfig, 'OPTIMIZER_BETAS', (0.9, 0.999))
        eps = getattr(TrainingConfig, 'OPTIMIZER_EPS', 1e-8)

        # build parameter groups — use LLRD if enabled, otherwise pass all params uniformly
        use_llrd = getattr(TrainingConfig, 'USE_LLRD', False)
        if use_llrd:
            decay_factor = getattr(TrainingConfig, 'LLRD_DECAY_FACTOR', 0.9)
            params = self._build_llrd_param_groups(lr, decay_factor, weight_decay)
            logger.info(f"LLRD enabled: built {len(params)} parameter groups with decay factor {decay_factor}")
        else:
            params = self.model.parameters()

        if opt_name == 'adamw_8bit':
            try:
                from bitsandbytes.optim import Adam8bit

                optimizer = Adam8bit(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            except Exception:
                optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        elif opt_name == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        elif opt_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        elif opt_name == 'sgd':
            momentum = getattr(TrainingConfig, 'OPTIMIZER_MOMENTUM', 0.9)
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

        else:
            raise ValueError(f"Unknown optimizer specified in TrainingConfig.OPTIMIZER: {opt_name}")

        self.optimizer = optimizer
        return optimizer

    def _build_llrd_param_groups(self, base_lr, decay_factor, weight_decay):
        # build optimizer parameter groups with layer-wise learning rate decay
        # the classifier/head receives base_lr; each layer closer to the input is scaled down by decay_facto, bias, LayerNorm, and layer_norm weights skip weight decay within every group (standard practice)
        import re

        no_decay_keywords = ["bias", "layernorm.weight", "layer_norm.weight"]
        # param names containing any of these are treated as the top (head) group
        head_keywords = ["classifier", "head", "pooler"]

        def get_group_idx(name):
            # extract transformer layer index from parameter name matches patterns: .layer.N.  .layers.N.  (covers DeBERTa, ModernBERT, CLIP)
            match = re.search(r'\.layers?\.(\d+)\.', name)
            if match:
                return int(match.group(1)) + 1  # +1 reserves index 0 for embeddings / stem params
            # classifier/head/pooler resolved to max_idx + 1 after scan
            if any(kw in name.lower() for kw in head_keywords):
                return -1
            # everything else (embeddings, unprefixed norms, projections) lowest group
            return 0

        # 1st pass: find the highest layer index present in this model
        max_layer_idx = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            idx = get_group_idx(name)
            if idx > max_layer_idx:
                max_layer_idx = idx

        head_idx = max_layer_idx + 1  # head sits one step above the topmost transformer layer

        # 2nd pass: assign each parameter to its group
        layer_groups = {}  # group_idx -> {'decay': [...], 'no_decay': [...]}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            idx = get_group_idx(name)
            group_idx = head_idx if idx == -1 else idx

            if group_idx not in layer_groups:
                layer_groups[group_idx] = {'decay': [], 'no_decay': []}

            is_no_decay = any(kw in name.lower() for kw in no_decay_keywords)
            if is_no_decay:
                layer_groups[group_idx]['no_decay'].append(param)

            else:
                layer_groups[group_idx]['decay'].append(param)

        # build param group dicts: head (head_idx) -> base_lr; depth increases toward embeddings
        param_groups = []
        for group_idx, group_params in layer_groups.items():
            depth = head_idx - group_idx  # 0 for head, 1 for top transformer layer, ect.
            lr = base_lr * (decay_factor ** depth)
            if group_params['decay']:
                param_groups.append({'params': group_params['decay'], 'lr': lr, 'weight_decay': weight_decay})
                
            if group_params['no_decay']:
                param_groups.append({'params': group_params['no_decay'], 'lr': lr, 'weight_decay': 0.0})

        return param_groups

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

        # build the loss function for this forward pass (device/dtype aware)
        if self.loss_fn_name == 'asl':
            # asl does not use pos_weight; class imbalance is handled via the asymmetric focusing parameters and probability margin instead.
            # so pos_weight is ignored here
            loss_fct = AsymmetricLoss(
                gamma_neg=getattr(TrainingConfig, 'ASL_GAMMA_NEG', 4),
                gamma_pos=getattr(TrainingConfig, 'ASL_GAMMA_POS', 1),
                clip=getattr(TrainingConfig, 'ASL_CLIP', 0.05),
                reduction='none',
            )

        else:
            # imports from losses.py
            # BCEWithLogitsLoss — pos_weight is moved to the correct device/dtype inside the wrapper
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

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

