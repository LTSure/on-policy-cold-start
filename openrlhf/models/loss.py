from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2, use_decay: bool = False, mode: str = "clip", kl_coef: float = 0.02, target_kl: float = 0.03,) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.mode = mode
        self.kl_coef = kl_coef
        self.target_kl = target_kl

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        return_info=False,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()

        # []
        if self.mode == "clip":
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            loss = -torch.min(surr1, surr2)

        # elif self.mode == "kl":
        #     print(f'[kl coef]:{self.kl_coef}')

        #     kl = old_log_probs - log_probs 
        #     clip_eps0 = 1
        #     ratio = torch.clamp(ratio, 1 - clip_eps0, 1 + clip_eps0)
        #     surr = ratio * advantages - self.kl_coef * kl
        #     loss = -surr

        elif self.mode == "kl":
            print("====== KL MODE DEBUG ======")

            # 1. 基本 KL（注意方向）
            kl = old_log_probs - log_probs     # approx KL (first-order)
            print(f"[KL] mean={kl.mean().item():.6f}, max={kl.max().item():.6f}, min={kl.min().item():.6f}")

            # 2. ratio 原始值
            raw_ratio = (log_probs - old_log_probs).exp()
            print(f"[RATIO raw] mean={raw_ratio.mean().item():.6f}, max={raw_ratio.max().item():.6f}, min={raw_ratio.min().item():.6f}")

            # 3. ratio clip 前后对比
            clip_eps0 = 1.0
            ratio = torch.clamp(raw_ratio, 1 - clip_eps0, 1 + clip_eps0)
            print(f"[RATIO clipped] mean={ratio.mean().item():.6f}, max={ratio.max().item():.6f}, min={ratio.min().item():.6f}")

            # clipped 比 raw 小多少？检查是否频繁触发 clip
            clipped_ratio_diff = (raw_ratio - ratio).abs()
            print(f"[RATIO diff after clip] mean={clipped_ratio_diff.mean().item():.6f}, max={clipped_ratio_diff.max().item():.6f}")

            # 4. advantages 是否正常（例如全是 1 或 0 或爆掉）
            print(f"[ADV] mean={advantages.mean().item():.6f}, max={advantages.max().item():.6f}, min={advantages.min().item():.6f}, std={advantages.std().item():.6f}")

            # 5. 主要组成项
            term1 = ratio * advantages
            term2 = self.kl_coef * kl
            print(f"[TERM1 = ratio * adv] mean={term1.mean().item():.6f}, max={term1.max().item():.6f}, min={term1.min().item():.6f}")
            print(f"[TERM2 = kl_coef * kl] coef={self.kl_coef:.6f}, mean={term2.mean().item():.6f}, max={term2.max().item():.6f}, min={term2.min().item():.6f}")

            # 6. 合成目标值
            surr = term1 - term2
            print(f"[SURR] mean={surr.mean().item():.6f}, max={surr.max().item():.6f}, min={surr.min().item():.6f}")

            # 7. loss 之前
            print(f"[LOSS before mean reduction] mean={(-surr).mean().item():.6f}")

            loss = -surr

            # 8. action mask 后真实使用的有效 token
            if action_mask is not None:
                valid_loss = loss[action_mask]
                print(f"[LOSS masked] mean={valid_loss.mean().item():.6f}, size={valid_loss.numel()}")

            # 最终 loss
            loss = masked_mean(loss, action_mask, dim=-1).mean()
            print(f"[LOSS final] {loss.item():.6f}")
            print("====== END DEBUG ======\n")



        loss = masked_mean(loss, action_mask, dim=-1).mean()
        print(f'[kl loss]:{loss}')




        if return_info:
            info = {}
            valid_ratios = ratio[action_mask]
            valid_log_probs = log_probs[action_mask]
            in_clip_mask = (valid_ratios >= (1 - self.clip_eps)) & (valid_ratios <= (1 + self.clip_eps))
            num_in_clip = in_clip_mask.sum().item()
            total_valid = valid_ratios.numel()
            info["ratio_in_clip"] = num_in_clip / total_valid if total_valid > 0 else 0.0
            info["clip_eps"] = self.clip_eps

            over_clip_mask = valid_ratios > (1 + self.clip_eps)
            under_clip_mask = valid_ratios < (1 - self.clip_eps)
            num_over_clip = over_clip_mask.sum().item()
            num_under_clip = under_clip_mask.sum().item()
            info["clip_over_ratio"] = num_over_clip / total_valid if total_valid > 0 else 0.0
            info["clip_under_ratio"] = num_under_clip / total_valid if total_valid > 0 else 0.0

            info["num_tokens"] = total_valid
            info["num_clipped_tokens"] = num_over_clip + num_under_clip
            

            valid_probs = torch.exp(valid_log_probs)
            bins = torch.arange(0.0, 1.01, 0.1, device=valid_probs.device)
            bin_indices = torch.bucketize(valid_probs, bins, right=False) - 1
            logprob_all_bins = {}
            for i in range(len(bins) - 1):
                left = bins[i].item()
                right = bins[i + 1].item()
                count = (bin_indices == i).sum().item()
                logprob_all_bins[f"{left:.1f}-{right:.1f}"] = count
            count = (bin_indices >= len(bins) - 1).sum().item()
            if count > 0:
                logprob_all_bins[f">= {bins[-1].item():.1f}"] = count
            info["all_logprob_bins"] = logprob_all_bins

            if num_over_clip > 0:
                over_log_probs = valid_log_probs[over_clip_mask]
                over_probs = torch.exp(over_log_probs)
                over_bin_indices = torch.bucketize(over_probs, bins, right=False) - 1
                over_logprob_bins = {}
                for i in range(len(bins) - 1):
                    left = bins[i].item()
                    right = bins[i + 1].item()
                    count = (over_bin_indices == i).sum().item()
                    over_logprob_bins[f"{left:.1f}-{right:.1f}"] = count
                count = (over_bin_indices >= len(bins) - 1).sum().item()
                if count > 0:
                    over_logprob_bins[f">= {bins[-1].item():.1f}"] = count
                info["over_logprob_bins"] = over_logprob_bins
            else:
                info["over_logprob_bins"] = {}
            

            if num_under_clip > 0:
                under_log_probs = valid_log_probs[under_clip_mask]
                under_probs = torch.exp(under_log_probs)
                under_bin_indices = torch.bucketize(under_probs, bins, right=False) - 1
                under_logprob_bins = {}
                for i in range(len(bins) - 1):
                    left = bins[i].item()
                    right = bins[i + 1].item()
                    count = (under_bin_indices == i).sum().item()
                    under_logprob_bins[f"{left:.1f}-{right:.1f}"] = count
                count = (under_bin_indices >= len(bins) - 1).sum().item()
                if count > 0:
                    under_logprob_bins[f">= {bins[-1].item():.1f}"] = count
                info["under_logprob_bins"] = under_logprob_bins
            else:
                info["under_logprob_bins"] = {}


            return loss, info


        return loss


class ReinforceLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
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
        loss = -advantages * log_probs
        # ratio = (log_probs - old_log_probs).exp()
        # surr1 = ratio * advantages
        # surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        # loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        if return_info:
            return loss, {"ratio_in_clip": 1.0}
        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
