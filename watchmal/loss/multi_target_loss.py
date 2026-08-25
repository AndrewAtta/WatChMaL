"""
Huber loss function that determines loss funciton for each individual target output.
This is to enusre that:
1. The loss for each target is treated according to its appropriate loss calculation (eg
relative huber loss for energy but normal huber loss for )
2. The total loss doesn't overweight any particular target over another

Currently hardcoded to be in the order [neutrino energy, energies, positions, directions]
TODO: Need to eventually make it so it automatically knows which to implement
Author: Andrew Atta
"""

import torch.nn as nn
from watchmal.loss.relative_huber import RelativeHuberLoss

#must match engine.target_key
loss_functions = {"neutrino_energy": RelativeHuberLoss,
                  "energies":        RelativeHuberLoss,
                  "positions":       nn.HuberLoss,
                  "directions":      nn.HuberLoss}

class MultiTargetLoss(nn.Module):
    
    def __init__(self, deltas=[0.1, 0.1, 0.1, 1], target_sizes = [1,1,3,3]):
        super().__init__()
        self.deltas = deltas
        self.target_sizes = target_sizes
        
    def forward(self, input, target):
        assert sum(self.target_sizes) == input.shape[-1], \
        f"targets {list(loss_functions)} with sizes {self.target_sizes} need " \
        f"{sum(self.target_sizes)} channels, got {input.shape[-1]}"

        
        total_loss = 0
        feature_idx = 0
        for loss_class, n, delta in zip(loss_functions.values(), self.target_sizes, self.deltas):
            loss_fn = loss_class(delta=delta)
            total_loss += loss_fn(input[:,feature_idx:feature_idx+n], target[:,feature_idx:feature_idx+n])
            feature_idx += n
            
        return total_loss