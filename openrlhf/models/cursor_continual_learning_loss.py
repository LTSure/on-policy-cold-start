"""
Continual Learning Loss Functions for Baselines
These losses are designed to mitigate catastrophic forgetting during fine-tuning.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import masked_mean


class ReplayLoss(nn.Module):
    """
    Simple Replay Loss - mix old and new data
    This is handled at the data loading level, so this is just standard cross-entropy.
    """
    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation (EWC) Loss
    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017
    
    Loss = L_task + λ * Σ F_i * (θ_i - θ*_i)²
    where F_i is the Fisher Information for parameter i
    """
    def __init__(self, ewc_lambda: float = 1000.0):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.IGNORE_INDEX = -100
        self.task_loss_fn = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        model: nn.Module = None,
        fisher_dict: dict = None,
        old_params_dict: dict = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output logits (B, T, V)
            labels: Target labels (B, T)
            model: The model being trained (needed for EWC regularization)
            fisher_dict: Dictionary of Fisher information for each parameter
            old_params_dict: Dictionary of old parameter values
        """
        # Standard cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        task_loss = self.task_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # EWC regularization term
        ewc_loss = 0.0
        if model is not None and fisher_dict is not None and old_params_dict is not None:
            for name, param in model.named_parameters():
                if name in fisher_dict and param.requires_grad:
                    fisher = fisher_dict[name]
                    old_param = old_params_dict[name]
                    # EWC penalty: F_i * (θ_i - θ*_i)²
                    ewc_loss += (fisher * (param - old_param) ** 2).sum()
        
        total_loss = task_loss + self.ewc_lambda * ewc_loss
        return total_loss


class L2RegularizationLoss(nn.Module):
    """
    L2 Regularization Loss (also known as weight decay regularization)
    
    Loss = L_task + λ * ||θ - θ_old||²
    
    This is a simpler baseline compared to EWC, treating all parameters equally.
    """
    def __init__(self, l2_lambda: float = 0.01):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.IGNORE_INDEX = -100
        self.task_loss_fn = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        model: nn.Module = None,
        old_params_dict: dict = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output logits (B, T, V)
            labels: Target labels (B, T)
            model: The model being trained
            old_params_dict: Dictionary of old parameter values
        """
        # Standard cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        task_loss = self.task_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # L2 regularization term
        l2_loss = 0.0
        if model is not None and old_params_dict is not None:
            for name, param in model.named_parameters():
                if name in old_params_dict and param.requires_grad:
                    old_param = old_params_dict[name]
                    # L2 penalty: ||θ - θ_old||²
                    l2_loss += ((param - old_param) ** 2).sum()
        
        total_loss = task_loss + self.l2_lambda * l2_loss
        return total_loss


class KLRegularizationLoss(nn.Module):
    """
    KL Regularization Loss
    
    Loss = L_task + λ * KL(π_new || π_old)
    
    This constrains the output distribution to not deviate too much from the old policy.
    Similar to the approach in: "Learning without Forgetting" (Li & Hoiem, 2016)
    """
    def __init__(self, kl_lambda: float = 0.1):
        super().__init__()
        self.kl_lambda = kl_lambda
        self.IGNORE_INDEX = -100
        self.task_loss_fn = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        old_logits: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: New model output logits (B, T, V)
            labels: Target labels (B, T)
            old_logits: Old model output logits (B, T, V) - from frozen reference model
            attention_mask: Attention mask (B, T)
        """
        # Standard cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        task_loss = self.task_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # KL divergence regularization
        kl_loss = 0.0
        if old_logits is not None:
            # Compute KL(new || old) at each position
            new_log_probs = F.log_softmax(logits[..., :-1, :], dim=-1)
            old_probs = F.softmax(old_logits[..., :-1, :], dim=-1)
            
            # KL divergence: Σ p_old * log(p_old / p_new)
            kl_div = F.kl_div(new_log_probs, old_probs, reduction='none').sum(dim=-1)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                mask = attention_mask[..., :-1].float()
                kl_loss = (kl_div * mask).sum() / mask.sum()
            else:
                kl_loss = kl_div.mean()
        
        total_loss = task_loss + self.kl_lambda * kl_loss
        return total_loss


class ClippedImportanceSamplingLoss(nn.Module):
    """
    Clipped Importance Sampling for SFT (Your Method)
    
    This is essentially PPO-style clipping applied during supervised fine-tuning.
    Loss = -min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)
    where r(θ) = π_θ(a|s) / π_old(a|s) is the importance ratio
    and A is the advantage (or simply reward for SFT context)
    """
    def __init__(self, clip_eps: float = 0.2):
        super().__init__()
        self.clip_eps = clip_eps
        
    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from reference policy
            advantages: Advantages or rewards
            action_mask: Mask for valid actions
            return_info: Whether to return additional info
        """
        # Importance ratio
        ratio = (log_probs - old_log_probs).exp()
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        
        # Masked mean
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        
        if return_info:
            info = {}
            if action_mask is not None:
                valid_ratios = ratio[action_mask.bool()]
                in_clip_mask = (valid_ratios >= (1 - self.clip_eps)) & (valid_ratios <= (1 + self.clip_eps))
                num_in_clip = in_clip_mask.sum().item()
                total_valid = valid_ratios.numel()
                info["ratio_in_clip"] = num_in_clip / total_valid if total_valid > 0 else 0.0
            return loss, info
        
        return loss


def compute_fisher_information(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_samples: int = 1000,
) -> dict:
    """
    Compute Fisher Information Matrix for EWC.
    
    Args:
        model: The model to compute Fisher information for
        dataloader: Data loader for computing Fisher information
        device: Device to run computation on
        num_samples: Number of samples to use for Fisher computation
        
    Returns:
        fisher_dict: Dictionary mapping parameter names to their Fisher information
    """
    model.eval()
    fisher_dict = {}
    
    # Initialize Fisher information to zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
    
    # Compute Fisher information using empirical samples
    samples_seen = 0
    for batch in dataloader:
        if samples_seen >= num_samples:
            break
            
        # Forward pass
        inputs = batch[1].to(device) if isinstance(batch, (list, tuple)) else batch['input_ids'].to(device)
        attention_mask = batch[2].to(device) if len(batch) > 2 else None
        
        outputs = model(inputs, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Compute log probability
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Sample from the model's distribution
        sampled_tokens = torch.multinomial(
            F.softmax(logits.view(-1, logits.size(-1)), dim=-1), 
            num_samples=1
        ).squeeze(-1)
        
        # Get log probability of sampled tokens
        log_prob = log_probs.view(-1, logits.size(-1)).gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        loss = -log_prob.sum()
        
        # Compute gradients
        model.zero_grad()
        loss.backward()
        
        # Accumulate squared gradients (Fisher information)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        samples_seen += inputs.size(0)
    
    # Normalize by number of samples
    for name in fisher_dict:
        fisher_dict[name] /= samples_seen
    
    return fisher_dict


def save_old_parameters(model: nn.Module) -> dict:
    """
    Save a copy of the current model parameters.
    
    Args:
        model: The model to save parameters from
        
    Returns:
        old_params_dict: Dictionary mapping parameter names to their values
    """
    old_params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            old_params_dict[name] = param.data.clone().detach()
    return old_params_dict

