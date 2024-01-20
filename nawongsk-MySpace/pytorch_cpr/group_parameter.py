import torch.nn as nn
import logging

def cpr_group_named_parameters(model, optim_hps, avoid_keywords=[],
                               embedding_regularization=False,
                               bias_regularization=False,
                               normalization_regularization=False):
    if not avoid_keywords:
        avoid_keywords = []

    apply_decay = set()
    apply_no_decay = set()
    special = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv2d)
    blacklist_weight_modules = ()
    if embedding_regularization:
        whitelist_weight_modules += (nn.Embedding,)
    else:
        blacklist_weight_modules += (nn.Embedding,)

    if normalization_regularization:
        whitelist_weight_modules += (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                     nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                     nn.GroupNorm, nn.SyncBatchNorm,
                                     nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                     nn.LayerNorm, nn.LocalResponseNorm)
    else:
        blacklist_weight_modules += (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                     nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                     nn.GroupNorm, nn.SyncBatchNorm,
                                     nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                     nn.LayerNorm, nn.LocalResponseNorm)


    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            # In case of parameter sharing, some parameters show up here but are not in
            # param_dict.keys()
            if not p.requires_grad or fpn not in param_dict:
                continue  # frozen weights
            if hasattr(p, '_optim'):
                special.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                apply_no_decay.add(fpn)
            elif any([keyword in fpn for keyword in avoid_keywords]):
                apply_no_decay.add(fpn)
            elif not bias_regularization and pn.endswith('bias'):
                apply_no_decay.add(fpn)
            elif isinstance(m, whitelist_weight_modules):
                apply_decay.add(fpn)
            else:
                logging.debug(f"cpr_group_named_parameters: Not using any rule for {fpn} in {type(m)}")

    apply_decay |= (param_dict.keys() - apply_no_decay - special)

    # validate that we considered every parameter
    inter_params = apply_decay & apply_no_decay
    union_params = apply_decay | apply_no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both apply_decay/apply_no_decay sets!"
    assert len(param_dict.keys() - special - union_params) == 0, (f"parameters {str(param_dict.keys() - union_params)} "
                                                                  f" were not separated into either apply_decay/apply_no_decay set!")

    if not apply_no_decay:
        param_groups = [{"params": [param_dict[pn] for pn in sorted(apply_decay)],
                         "names": [pn for pn in sorted(apply_decay)], "apply_decay": True, **optim_hps}]
    else:
        param_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(apply_decay))],
             "names": [pn for pn in sorted(list(apply_decay))], "apply_decay": True, **optim_hps},
            {"params": [param_dict[pn] for pn in sorted(list(apply_no_decay))],
             "names": [pn for pn in sorted(list(apply_no_decay))], "apply_decay": False, **optim_hps},
        ]
    # Add parameters with special hyperparameters
    # Unique dicts
    hps = [dict(s) for s in set(frozenset(param_dict[pn]._optim.items()) for pn in special)]
    for hp in hps:
        params = [param_dict[pn] for pn in sorted(list(special)) if param_dict[pn]._optim == hp]
        param_groups.append({"params": params, **hp})

    return param_groups
