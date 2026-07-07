import json
import math
import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss, ReinforceLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer, NaiveExperienceMakerORM800K, NaiveExperienceMakerPRM800K, NaiveExperienceMakerPRM800K_BOX, NaiveExperienceMakerSFT, NaiveExperienceMakerSFT_MT

# Import analysis functions
try:
    from openrlhf.utils.prob_gradient_analysis import (
        analyze_prob_gradient_distribution,
        visualize_prob_gradient_distribution,
        save_analysis_results_txt,
    )
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the L2 norm of gradients for all parameters in the model.
    
    Args:
        model: The model to compute gradient norms for
        
    Returns:
        grad_norm: The L2 norm of all gradients
    """
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    return grad_norm ** 0.5


def compute_average_gradient_norm(model: torch.nn.Module, num_params: int = None) -> float:
    """
    Compute the average gradient norm across all parameters.
    
    Args:
        model: The model to compute gradient norms for
        num_params: Number of parameters (if None, will count automatically)
        
    Returns:
        avg_grad_norm: The average gradient norm
    """
    if num_params is None:
        num_params = sum(1 for p in model.parameters() if p.grad is not None)
    
    if num_params == 0:
        return 0.0
        
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item()
    
    return total_grad_norm / num_params


def save_gradient_logs(actor_grad_norms: list, log_file_path: str, global_step: int, save_individual: bool = False):
    """
    Save actor gradient norm and std to a text file.
    
    Args:
        actor_grad_norms: List of actor gradient norm values
        log_file_path: Path to the log file
        global_step: Current global step
        save_individual: If True, save each gradient norm value individually
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    if save_individual:
        # Save each gradient norm value individually
        with open(log_file_path, 'a') as f:
            for i, grad_norm in enumerate(actor_grad_norms):
                f.write(f"Step {global_step}, Batch {i+1}: Actor Gradient Norm = {grad_norm:.6f}\n")
    else:
        # Get the last gradient norm
        actor_grad_norm = actor_grad_norms[-1]
        
        # Calculate std
        if len(actor_grad_norms) > 1:
            grad_std = torch.tensor(actor_grad_norms).std().item()
        else:
            grad_std = 0.0
        
        with open(log_file_path, 'a') as f:
            f.write(f"Step {global_step}: Actor Gradient Norm = {actor_grad_norm:.6f}, Std = {grad_std:.6f}\n")


def compute_actor_gradient_statistics(actor_grad_norms: list) -> dict:
    """
    Compute statistics for actor gradient norms.
    
    Args:
        actor_grad_norms: List of actor gradient norm values
        
    Returns:
        stats: Dictionary containing actor gradient statistics
    """
    if not actor_grad_norms:
        return {}
    
    stats = {
        'actor_avg_gradient_norm': sum(actor_grad_norms) / len(actor_grad_norms),
        'actor_max_gradient_norm': max(actor_grad_norms),
        'actor_min_gradient_norm': min(actor_grad_norms),
        'actor_gradient_norm_std': torch.tensor(actor_grad_norms).std().item() if len(actor_grad_norms) > 1 else 0.0
    }
    return stats


def save_step_prob_logs(step_probabilities: list, log_file_path: str, global_step: int, save_individual: bool = False):
    """
    Save step probabilities to a text file.
    
    Args:
        step_probabilities: List of step probability values for each batch
        log_file_path: Path to the log file
        global_step: Current global step
        save_individual: If True, save each step probability value individually
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    if save_individual:
        # Save each step probability value individually
        with open(log_file_path, 'a') as f:
            for batch_idx, batch_probs in enumerate(step_probabilities):
                for step_idx, prob_list in enumerate(batch_probs):
                    for token_idx, prob in enumerate(prob_list):
                        f.write(f"Step {global_step}, Batch {batch_idx+1}, Step {step_idx+1}, Token {token_idx+1}: Probability = {prob:.6f}\n")
    else:
        # Save summary statistics
        all_probs = []
        for batch_probs in step_probabilities:
            for prob_list in batch_probs:
                if isinstance(prob_list, list):
                    all_probs.extend(prob_list)
                else:
                    all_probs.append(prob_list)
        
        if all_probs:
            # Ensure all elements are numbers, not lists
            flat_probs = []
            for prob in all_probs:
                if isinstance(prob, list):
                    flat_probs.extend(prob)
                else:
                    flat_probs.append(prob)
            
            if flat_probs:
                avg_prob = sum(flat_probs) / len(flat_probs)
                max_prob = max(flat_probs)
                min_prob = min(flat_probs)
                prob_std = torch.tensor(flat_probs).std().item() if len(flat_probs) > 1 else 0.0
                
                with open(log_file_path, 'a') as f:
                    f.write(f"Step {global_step}: Avg Prob = {avg_prob:.6f}, Max = {max_prob:.6f}, Min = {min_prob:.6f}, Std = {prob_std:.6f}\n")


def save_prob_gradient_pairs_logs(prob_gradient_pairs: list, log_file_path: str, global_step: int):
    """
    Save probability-gradient pairs to a text file in JSON format.
    
    Args:
        prob_gradient_pairs: List of (probability, gradient) tuples
        log_file_path: Path to the log file
        global_step: Current global step
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    with open(log_file_path, 'a') as f:
        for prob, grad in prob_gradient_pairs:
            f.write(json.dumps({"prob": float(prob), "grad": float(grad), "step": global_step}) + "\n")


def compute_step_prob_statistics(step_probabilities: list) -> dict:
    """
    Compute statistics for step probabilities.
    
    Args:
        step_probabilities: List of step probability values for each batch
        
    Returns:
        stats: Dictionary containing step probability statistics
    """
    if not step_probabilities:
        return {}
    
    # Flatten all probabilities from all batches
    all_probs = []
    for batch_probs in step_probabilities:
        for prob_list in batch_probs:
            if isinstance(prob_list, list):
                all_probs.extend(prob_list)
            else:
                all_probs.append(prob_list)
    
    # Ensure all elements are numbers, not lists
    flat_probs = []
    for prob in all_probs:
        if isinstance(prob, list):
            flat_probs.extend(prob)
        else:
            flat_probs.append(prob)
    
    all_probs = flat_probs
    
    if not all_probs:
        return {}
    
    stats = {
        'step_avg_probability': sum(all_probs) / len(all_probs),
        'step_max_probability': max(all_probs),
        'step_min_probability': min(all_probs),
        'step_probability_std': torch.tensor(all_probs).std().item() if len(all_probs) > 1 else 0.0,
        'total_steps': len(all_probs)
    }
    return stats


class SFTPPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        entropy_coef: float = 0.0,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 2048,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        without_ppo: bool = False,
        use_decay: bool = False,
        loss_mode: bool = "clip",
        use_muti_turn: bool = False,
        dft_baseline: bool = False,  # Enable DFT baseline mode for confidence-based loss reweighting
        neftune_alpha: float = None,  # NEFT alpha parameter for embedding noise injection
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.entropy_coef=entropy_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn
        self.dft_baseline = dft_baseline
        self.neftune_alpha = neftune_alpha

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
        # [add]------------------------[add]
        self.count = 0
        self.over_logprob_bins={}
        self.under_logprob_bins={}
        self.all_logprob_bins={}
        self.use_muti_turn=use_muti_turn
        
        # Actor gradient tracking variables
        self.actor_gradient_norms = []
        self.gradient_log_path = None
        # Step probability tracking variables
        self.step_probabilities = []
        self.step_prob_log_path = None
        # Token probability-gradient pairs tracking
        self.prob_gradient_pairs = []
        self.prob_gradient_log_path = None
        self.save_prob_gradient_pairs = getattr(self.args, "save_prob_gradient_pairs", False)
        
        # [NEW] Token clip statistics tracking variables
        self.token_clip_log_path = None
        self.rollout_batch_counter = 0
        # [add]------------------------[add]

        if without_ppo:
            self.actor_loss_fn = ReinforceLoss(eps_clip)
        else:
            # [replace]------------------------[replace]
            # self.actor_loss_fn = PolicyLoss(eps_clip)
            self.actor_loss_fn = PolicyLoss(eps_clip, use_decay=use_decay, mode=loss_mode, kl_coef=init_kl_coef, target_kl=kl_target if kl_target else 0.01)
            self.actor_loss_fn_sft = ReinforceLoss(eps_clip)
            # if not self.use_muti_turn:
            #     self.actor_loss_fn = PolicyLoss(eps_clip, use_decay=use_decay)
            #     self.actor_loss_fn_sft = ReinforceLoss(eps_clip)
            # else:
            #     self.actor_loss_fn = PolicyLoss_MT(eps_clip, use_decay=use_decay)
            #     self.actor_loss_fn_sft = ReinforceLoss(eps_clip)
            # [replace]------------------------[replace]


        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        # [lhy replace]
        if self.use_muti_turn:
            self.experience_maker = NaiveExperienceMakerSFT_MT(
                actor,
                critic,
                reward_model,
                initial_model,
                tokenizer,
                prompt_max_len,
                self.kl_ctl,
                strategy,
                remote_rm_url,
                reward_fn,
                without_ppo=without_ppo,
            )
        else:
            self.experience_maker = NaiveExperienceMakerSFT(
                actor,
                critic,
                reward_model,
                initial_model,
                tokenizer,
                prompt_max_len,
                self.kl_ctl,
                strategy,
                remote_rm_url,
                reward_fn,
                without_ppo=without_ppo,
            )
        # [lhy replace]



        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples, self.use_muti_turn
        )
        self.without_ppo = without_ppo
        self.dft_baseline = dft_baseline
        self.neftune_alpha = neftune_alpha

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )
        # [lhy add]
        self.ckpt_path=args.ckpt_path
        self.save_gradient_norms = getattr(args, 'save_gradient_norms', False)
        self.save_step_probs = getattr(args, 'save_step_probs', False)
        # Initialize gradient log file path
        if self.strategy.is_rank_0():
            self.gradient_log_path = os.path.join(args.ckpt_path, "gradient", "gradient_logs.txt")
            self.step_prob_log_path = os.path.join(args.ckpt_path, "step_probs", "step_prob_logs.txt")
            # [NEW] Initialize token clip log file path
            self.token_clip_log_path = os.path.join(args.ckpt_path, "token.txt")
            os.makedirs(os.path.dirname(self.gradient_log_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.step_prob_log_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.token_clip_log_path), exist_ok=True)
        
        # Set up probability-gradient pairs logging
        self.prob_gradient_log_path = None
        self.prob_gradient_output_dir = None
        if self.save_prob_gradient_pairs:
            if self.strategy.is_rank_0():
                self.prob_gradient_output_dir = os.path.join(args.ckpt_path, "prob_gradient")
                self.prob_gradient_log_path = os.path.join(
                    self.prob_gradient_output_dir,
                    "prob_gradient_pairs.txt",
                )
                os.makedirs(self.prob_gradient_output_dir, exist_ok=True)
                self.strategy.print(f"Probability-gradient pairs will be saved to: {self.prob_gradient_log_path}")
        # [lhy add]




        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        print(f'{consumed_samples} === {args.rollout_batch_size }==== {num_rollouts_per_episodes}')
        # assert 0
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}",
                disable=not self.strategy.is_rank_0(),
            )


            for rand_prompts in self.prompts_dataloader:
                # print(f'qweqeqw:{len(self.prompts_dataloader)},dfsd:{len(rand_prompts)},{len(rand_prompts["input"])}')
                rand_targets = rand_prompts["target"]
                rand_answer = rand_prompts["answer"]
                rand_responses = rand_prompts["response"]
                # rand_responses = None

                rand_prompts = rand_prompts["input"]

                
                #print("rand_prompts", rand_prompts)
                #print("rabd_answers", rand_answer)
                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(rand_prompts, rand_answer, rand_responses, **self.generate_kwargs)
                ):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)
                    # assert (experience.action_mask.sum(-1)==0).any().item==False

                torch.cuda.empty_cache()
                # self.replay_buffer.normalize("advantages", self.strategy)  # FIXED: Re-enabled advantage normalization
                status = self.ppo_train(steps)
                self.replay_buffer.clear()
                torch.cuda.empty_cache()

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()

  

                steps = steps + 1


            if episode >= args.num_episodes - 5:
                self._save_checkpoint(args, tag=str(episode+1), global_step=steps)

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        
        # [NEW] Track token clip statistics for this rollout batch
        total_tokens_in_rollout = 0
        total_clipped_tokens_in_rollout = 0

        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            
            for experience in pbar:
                action_mask=experience.action_mask
                mask_sum = action_mask.sum(-1)
                if (mask_sum == 0).any().item():
                    print("[DEBUG] sum(dim):", mask_sum)
                    assert 0
                experience.to_device(device)
                status = self.training_step(experience, global_steps)
                
                # [NEW] Accumulate token statistics
                if "num_tokens" in status and "num_clipped_tokens" in status:
                    total_tokens_in_rollout += status["num_tokens"]
                    total_clipped_tokens_in_rollout += status["num_clipped_tokens"]

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                        "rc": status["ratio_in_clip"]
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)
            
        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        
        # [NEW] Save token clip statistics to file
        if self.strategy.is_rank_0() and self.token_clip_log_path is not None:
            self.rollout_batch_counter += 1
            with open(self.token_clip_log_path, 'a') as f:
                f.write(f"rollout batch {self.rollout_batch_counter}: token cliped: {total_clipped_tokens_in_rollout} / all token: {total_tokens_in_rollout}\n")
        
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience, global_steps)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        return status

    def training_step_actor(self, experience: Experience, global_steps: int = 0) -> Dict[str, float]:
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            action_mask = experience.action_mask
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        

        # actor loss       
        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        # Gradient norm analysis - compute per probability bin
        # Run only on the first step, then exit
        run_grad_analysis = False
        # if self.strategy.is_rank_0() and global_steps == 1:
        #     run_grad_analysis = True
        
        if run_grad_analysis:
            try:
                def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                    log_probs = F.log_softmax(logits, dim=-1)  
                    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  

                def get_response_log_probs(output_logits: torch.Tensor, labels: torch.Tensor, action_mask: torch.Tensor) -> tuple:
                    shift_logits = output_logits[:, :-1, :].contiguous()  # (B, T-1, V)
                    shift_labels = labels[:, 1:].contiguous()  # (B, T-1)
                    
                    all_log_probs = log_probs_from_logits(shift_logits, shift_labels)  # (B, T-1)
                    
                    # Use action_mask to identify valid response tokens
                    # action_mask might have shape (B, T) or (B, T-1), need to match all_log_probs shape
                    if action_mask.size(1) == all_log_probs.size(1):
                        # action_mask already matches shifted dimensions
                        full_valid_mask = action_mask.contiguous().bool()
                    elif action_mask.size(1) == all_log_probs.size(1) + 1:
                        # action_mask has one more dimension, remove the last one
                        full_valid_mask = action_mask[:, :-1].contiguous().bool()
                    else:
                        # Fallback: create mask matching all_log_probs shape
                        full_valid_mask = torch.ones_like(all_log_probs, dtype=torch.bool)
                    
                    # Filter to only first turn tokens (for multi-turn data)
                    # Find the first contiguous block of response tokens (first turn)
                    first_turn_mask = torch.zeros_like(full_valid_mask, dtype=torch.bool)
                    for b in range(full_valid_mask.size(0)):
                        batch_mask = full_valid_mask[b]
                        # Find first True position (start of first response)
                        true_indices = batch_mask.nonzero(as_tuple=True)[0]
                        if len(true_indices) > 0:
                            first_true = true_indices[0].item()
                            # Find where first turn ends (first False after first True, or end of sequence)
                            false_after_first = (batch_mask[first_true:] == False).nonzero(as_tuple=True)[0]
                            if len(false_after_first) > 0:
                                first_turn_end = first_true + false_after_first[0].item()
                            else:
                                first_turn_end = batch_mask.size(0)
                            # Mark first turn tokens
                            first_turn_mask[b, first_true:first_turn_end] = True
                    
                    # Use only first turn tokens
                    valid_mask = first_turn_mask
                    response_log_probs = all_log_probs[valid_mask]
                    
                    return response_log_probs, valid_mask

                def analyze_response_probs(response_log_probs: torch.Tensor, valid_mask: torch.Tensor, bins: torch.Tensor) -> tuple:
                    """
                    Analyze response probabilities and create group masks for each bin.
                    """
                    if len(response_log_probs) == 0:
                        # Return empty results if no valid tokens
                        group_masks = [torch.zeros_like(valid_mask, dtype=torch.bool) for _ in range(len(bins) - 1)]
                        counts = torch.zeros(len(bins) - 1, dtype=torch.float32)
                        return torch.tensor([]), counts, group_masks
                    
                    probs = torch.exp(response_log_probs)
                    
                    # Create a full-size mask tensor matching shift_labels shape
                    # Use the same dtype as probs to avoid dtype mismatch (probs might be bfloat16)
                    full_probs = torch.zeros_like(valid_mask, dtype=probs.dtype)
                    full_probs[valid_mask] = probs
                    
                    # Create group masks for each bin
                    group_masks = []
                    counts = []
                    for i in range(len(bins) - 1):
                        bin_start = bins[i]
                        bin_end = bins[i + 1]
                        if i == len(bins) - 2:  # Last bin includes the right endpoint
                            mask = (full_probs >= bin_start) & (full_probs <= bin_end) & valid_mask
                        else:
                            mask = (full_probs >= bin_start) & (full_probs < bin_end) & valid_mask
                        group_masks.append(mask)
                        counts.append(mask.sum().item())
                    
                    counts = torch.tensor(counts, dtype=torch.float32)
                    return probs, counts, group_masks

                def compute_group_gradient_norms(
                    model: torch.nn.Module,
                    inputs: torch.Tensor,
                    labels: torch.Tensor,
                    action_mask: torch.Tensor,
                    group_masks: list,
                    max_samples_per_bin: int = 20,
                    ignore_index: int = -100
                ) -> dict:
                    """
                    Compute per-token gradient norms for each probability bin and return mean/std statistics.
                    Filters to first turn only for multi-turn data.
                    """
                    model.train()
                    results = {}
                    
                    # Debug: Check original data statistics
                    print(f"[DEBUG] Original data: inputs shape={inputs.shape}, labels shape={labels.shape}, action_mask shape={action_mask.shape}")
                    print(f"[DEBUG] Original action_mask: {action_mask.sum().item()} True tokens out of {action_mask.numel()}")
                    
                    # Filter to first turn only (for multi-turn data)
                    first_turn_mask = torch.zeros_like(action_mask, dtype=torch.bool)
                    for b in range(action_mask.size(0)):
                        batch_mask = action_mask[b]
                        true_count = batch_mask.sum().item()
                        print(f"[DEBUG] Batch {b}: action_mask has {true_count} True tokens out of {len(batch_mask)} total")
                        
                        # Find first True position (start of first response)
                        true_indices = batch_mask.nonzero(as_tuple=True)[0]
                        if len(true_indices) > 0:
                            first_true = true_indices[0].item()
                            print(f"[DEBUG] Batch {b}: First response starts at position {first_true}")
                            
                            # Strategy: Find ALL True tokens that belong to the first turn
                            first_turn_end = first_true
                            i = first_true
                            
                            # Track the last True position in the first turn
                            last_true_pos = first_true
                            
                            # Count total True tokens in first turn
                            first_turn_true_count = 0
                            
                            while i < len(batch_mask):
                                if batch_mask[i]:
                                    # Found True token - this is part of first turn
                                    last_true_pos = i
                                    first_turn_end = i
                                    first_turn_true_count += 1
                                    i += 1
                                else:
                                    # Found False - check if it's a gap or end of first turn
                                    # Count how many consecutive False we have
                                    false_start = i
                                    false_count = 0
                                    while i < len(batch_mask) and not batch_mask[i]:
                                        false_count += 1
                                        i += 1
                                    
                                    # If we have a very long False sequence (>100), this is likely the next turn's prompt
                                    # Stop here and use the last True position as the end
                                    if false_count > 100:
                                        first_turn_end = last_true_pos
                                        print(f"[DEBUG] Batch {b}: Found long False sequence ({false_count} tokens) at position {false_start}, ending first turn at {last_true_pos}")
                                        break
                                    # Otherwise, it's a small gap - continue (we've already advanced i)
                            
                            # If we reached the end, use the last True position
                            if i >= len(batch_mask):
                                first_turn_end = last_true_pos
                            
                            print(f"[DEBUG] Batch {b}: First turn has {first_turn_true_count} True tokens (response tokens)")
                            
                            # Include the entire first turn (prompt + first response)
                            # Mark from start to first_turn_end (inclusive)
                            first_turn_mask[b, :first_turn_end+1] = True
                            first_turn_length = first_turn_end + 1
                            first_response_length = first_turn_end - first_true + 1
                            
                            # Debug: show action_mask around first turn
                            start_debug = max(0, first_true - 5)
                            end_debug = min(len(batch_mask), first_turn_end + 10)
                            mask_snippet = batch_mask[start_debug:end_debug].cpu().numpy().astype(int).tolist()
                            print(f"[DEBUG] Batch {b}: First turn ends at position {first_turn_end} (total length: {batch_mask.size(0)}, first turn length: {first_turn_length}, first response length: {first_response_length})")
                            print(f"[DEBUG] Batch {b}: action_mask snippet around first turn [{start_debug}:{end_debug}]: {mask_snippet}")
                        else:
                            print(f"[WARNING] Batch {b}: No True tokens in action_mask!")
                            # If no True tokens, include all tokens (might be single-turn or prompt-only)
                            first_turn_mask[b, :] = True
                    
                    print(f"[DEBUG] First turn mask: {first_turn_mask.sum().item()} tokens marked as first turn (out of {first_turn_mask.numel()} total)")
                    
                    # Filter inputs, labels, and action_mask to first turn only
                    # For each batch, keep only tokens up to first_turn_end
                    filtered_inputs = []
                    filtered_labels = []
                    filtered_action_mask = []
                    
                    for b in range(inputs.size(0)):
                        batch_first_turn_end = first_turn_mask[b].nonzero(as_tuple=True)[0]
                        if len(batch_first_turn_end) > 0:
                            end_idx = batch_first_turn_end[-1].item() + 1
                            filtered_inputs.append(inputs[b, :end_idx])
                            filtered_labels.append(labels[b, :end_idx])
                            filtered_action_mask.append(action_mask[b, :end_idx])
                    
                    # Pad to same length for batching
                    if len(filtered_inputs) > 0:
                        max_len = max([x.size(0) for x in filtered_inputs])
                        padded_inputs = []
                        padded_labels = []
                        padded_action_mask = []
                        
                        for i in range(len(filtered_inputs)):
                            pad_len = max_len - filtered_inputs[i].size(0)
                            if pad_len > 0:
                                # Pad inputs: (seq_len, ) -> pad to (max_len, )
                                padded_inputs.append(torch.cat([filtered_inputs[i], torch.zeros(pad_len, dtype=filtered_inputs[i].dtype, device=filtered_inputs[i].device)]))
                                # Pad labels: (seq_len, ) -> pad to (max_len, )
                                padded_labels.append(torch.cat([filtered_labels[i], torch.full((pad_len,), ignore_index, dtype=filtered_labels[i].dtype, device=filtered_labels[i].device)]))
                                # Pad action_mask: (seq_len, ) -> pad to (max_len, )
                                padded_action_mask.append(torch.cat([filtered_action_mask[i], torch.zeros(pad_len, dtype=filtered_action_mask[i].dtype, device=filtered_action_mask[i].device)]))
                            else:
                                padded_inputs.append(filtered_inputs[i])
                                padded_labels.append(filtered_labels[i])
                                padded_action_mask.append(filtered_action_mask[i])
                        
                        filtered_inputs = torch.stack(padded_inputs)
                        filtered_labels = torch.stack(padded_labels)
                        filtered_action_mask = torch.stack(padded_action_mask)
                    else:
                        # No valid first turn data, return empty results
                        return {i: {'mean': 0.0, 'std': 0.0, 'count': 0} for i in range(10)}
                    
                    # Re-run forward pass with filtered first-turn data
                    print(f"[DEBUG] Re-running forward pass with first-turn only data (shape: {filtered_inputs.shape})")
                    print(f"[DEBUG] Original sequences shape: {inputs.shape}, filtered shape: {filtered_inputs.shape}")
                    output = model(filtered_inputs, attention_mask=filtered_action_mask, return_output=True)
                    output_logits = output.logits
                    
                    # Calculate shift_logits and shift_labels (aligned with loss computation)
                    shift_logits = output_logits[:, :-1, :].contiguous()
                    shift_labels = filtered_labels[:, 1:].contiguous()
                    
                    # Update group_masks to match filtered dimensions
                    # Recompute masks based on filtered data
                    filtered_action_mask_shifted = filtered_action_mask[:, :-1].contiguous().bool()
                    print(f"[DEBUG] Filtered action_mask_shifted: {filtered_action_mask_shifted.sum().item()} valid tokens out of {filtered_action_mask_shifted.numel()}")
                    
                    # Recompute group masks for filtered data
                    # We need to recompute probabilities and masks for the filtered sequence
                    shift_log_probs = F.log_softmax(shift_logits, dim=-1)
                    
                    # Ensure shift_labels are valid (within vocab size and not ignore_index)
                    vocab_size = shift_logits.size(-1)
                    # Check for valid labels
                    valid_label_mask = (shift_labels >= 0) & (shift_labels < vocab_size) & (shift_labels != ignore_index)
                    print(f"[DEBUG] Valid labels: {valid_label_mask.sum().item()} out of {valid_label_mask.numel()}")
                    
                    # Clamp labels to valid range to avoid index errors during gather
                    shift_labels_clamped = torch.clamp(shift_labels, 0, vocab_size - 1)
                    
                    # Gather log probs (using clamped labels to avoid index errors)
                    filtered_all_log_probs = shift_log_probs.gather(dim=-1, index=shift_labels_clamped.unsqueeze(-1)).squeeze(-1)
                    
                    # Mask out invalid labels (set to very negative value so exp becomes ~0)
                    filtered_all_log_probs = torch.where(
                        valid_label_mask, 
                        filtered_all_log_probs, 
                        torch.tensor(-1e10, device=filtered_all_log_probs.device, dtype=filtered_all_log_probs.dtype)
                    )
                    
                    # Only consider tokens that are both in action_mask and have valid labels
                    filtered_valid_mask = filtered_action_mask_shifted & valid_label_mask
                    print(f"[DEBUG] Final valid_mask: {filtered_valid_mask.sum().item()} tokens")
                    
                    # Extract probabilities for valid tokens only (matching reference code pattern)
                    if filtered_valid_mask.any():
                        filtered_response_log_probs = filtered_all_log_probs[filtered_valid_mask]
                        filtered_probs = torch.exp(filtered_response_log_probs)  # log probability to actual probability
                        print(f"[DEBUG] Filtered probs: {len(filtered_probs)} tokens, min={filtered_probs.min().item():.6f}, max={filtered_probs.max().item():.6f}, mean={filtered_probs.mean().item():.6f}")
                        print(f"[DEBUG] Filtered probs distribution: <0.1: {(filtered_probs < 0.1).sum().item()}, 0.1-0.5: {((filtered_probs >= 0.1) & (filtered_probs < 0.5)).sum().item()}, 0.5-0.9: {((filtered_probs >= 0.5) & (filtered_probs < 0.9)).sum().item()}, >=0.9: {(filtered_probs >= 0.9).sum().item()}")
                    else:
                        filtered_probs = torch.tensor([], device=filtered_all_log_probs.device, dtype=filtered_all_log_probs.dtype)
                        filtered_response_log_probs = torch.tensor([], device=filtered_all_log_probs.device, dtype=filtered_all_log_probs.dtype)
                        print(f"[WARNING] No valid tokens after filtering!")
                    
                    # Recreate bins and group masks using the same approach as reference
                    bins = torch.linspace(0.0, 1.0, 11).to(output_logits.device)
                    
                    # Use histc to count tokens in each bin (like reference code)
                    # histc requires float32, so convert if needed
                    if len(filtered_probs) > 0:
                        filtered_probs_f32 = filtered_probs.float()  # Convert to float32 for histc
                        counts = torch.histc(filtered_probs_f32, bins=len(bins)-1, min=bins[0].item(), max=bins[-1].item())
                        print(f"[DEBUG] Filtered probabilities range: min={filtered_probs.min().item():.4f}, max={filtered_probs.max().item():.4f}, mean={filtered_probs.mean().item():.4f}")
                        print(f"[DEBUG] Token counts per bin: {counts.tolist()}")
                    else:
                        counts = torch.zeros(len(bins) - 1, dtype=torch.float32)
                    
                    # Create group masks: map each token back to its bin
                    # First, create a full-size probability tensor matching shift_labels shape
                    filtered_full_probs = torch.zeros_like(filtered_valid_mask, dtype=filtered_probs.dtype if len(filtered_probs) > 0 else torch.float32)
                    if filtered_valid_mask.any() and len(filtered_probs) > 0:
                        filtered_full_probs[filtered_valid_mask] = filtered_probs
                        print(f"[DEBUG] Assigned probabilities to full tensor: {filtered_valid_mask.sum().item()} positions filled")
                        print(f"[DEBUG] Full probs tensor: shape={filtered_full_probs.shape}, non-zero count={(filtered_full_probs > 0).sum().item()}")
                        print(f"[DEBUG] Full probs range: min={filtered_full_probs[filtered_valid_mask].min().item():.6f}, max={filtered_full_probs[filtered_valid_mask].max().item():.6f}")
                    
                    # Create group masks for each bin
                    filtered_group_masks = []
                    print(f"[DEBUG] Creating group masks with bins: {bins.tolist()}")
                    for i in range(len(bins) - 1):
                        bin_start = bins[i]
                        bin_end = bins[i + 1]
                        if i == len(bins) - 2:  # Last bin includes the right endpoint
                            mask = (filtered_full_probs >= bin_start) & (filtered_full_probs <= bin_end) & filtered_valid_mask
                        else:
                            mask = (filtered_full_probs >= bin_start) & (filtered_full_probs < bin_end) & filtered_valid_mask
                        filtered_group_masks.append(mask)
                        token_count = mask.sum().item()
                        if token_count > 0:
                            # Debug: show actual probability values in this bin
                            bin_probs = filtered_full_probs[mask]
                            print(f"[DEBUG] Bin [{bin_start:.1f}, {bin_end:.1f}] has {token_count} tokens, prob range: [{bin_probs.min().item():.6f}, {bin_probs.max().item():.6f}]")
                        else:
                            print(f"[DEBUG] Bin [{bin_start:.1f}, {bin_end:.1f}] has 0 tokens")
                    
                    # Flatten for easier indexing
                    batch_size, seq_len_minus_one, vocab_size = shift_logits.shape
                    shift_logits_flat = shift_logits.view(-1, vocab_size)
                    shift_labels_flat = shift_labels.view(-1)
                    
                    print(f"[DEBUG] Computing gradient norms for every bin!!!")
                    # Process each bin - compute gradient norm for the bin's loss
                    # IMPORTANT: DeepSpeed ZeRO-3 doesn't work well with retain_graph=True
                    # Solution: Accumulate gradients and compute difference for each bin
                    
                    # Initialize variable to track previous gradient norm squared
                    prev_grad_norm_sq = 0.0
                    
                    for bin_idx, mask in enumerate(filtered_group_masks):
                        if not mask.any():
                            print(f"[DEBUG] Bin {bin_idx} has no tokens")
                            results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': 0}
                            continue
                        
                        # Extract logits and labels for tokens in this bin from pre-computed logits
                        # mask is (batch_size, seq_len-1), flatten to index into shift_logits
                        mask_flat = mask.view(-1)  # Flatten to (batch_size * (seq_len-1),)
                        
                        group_logits = shift_logits_flat[mask_flat]  # (num_group_tokens, vocab_size)
                        group_labels = shift_labels_flat[mask_flat]  # (num_group_tokens,)
                        
                        if len(group_logits) == 0:
                            print(f"[DEBUG] Bin {bin_idx} has empty group_logits")
                            results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': 0}
                            continue
                        
                        # Sample tokens if too many
                        num_tokens = len(group_logits)
                        print(f"[DEBUG] Bin {bin_idx} has {num_tokens} tokens")
                        if num_tokens > max_samples_per_bin:
                            sampled_indices = np.random.choice(num_tokens, max_samples_per_bin, replace=False)
                            sampled_logits = group_logits[sampled_indices]
                            sampled_labels = group_labels[sampled_indices]
                        else:
                            sampled_logits = group_logits
                            sampled_labels = group_labels
                        
                        try:
                            # Check if all labels are ignore_index
                            valid_labels = (sampled_labels != ignore_index)
                            if not valid_labels.any():
                                print(f"[WARNING] Bin {bin_idx} has no valid labels (all ignore_index)")
                                results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': int(num_tokens)}
                                continue
                            
                            # Compute group loss
                            group_loss = F.cross_entropy(
                                sampled_logits,
                                sampled_labels,
                                ignore_index=ignore_index
                            )
                            
                            if group_loss.item() == 0.0 or not torch.isfinite(group_loss):
                                print(f"[WARNING] Bin {bin_idx} has invalid loss: {group_loss.item()}")
                                results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': int(num_tokens)}
                                continue
                            
                            print(f"[DEBUG] Computing gradient norms before backward for bin {bin_idx} with {len(sampled_logits)} tokens, loss={group_loss.item():.4f}")
                            
                            # CRITICAL: For DeepSpeed ZeRO-3, we cannot use retain_graph=True with multiple backward passes
                            # The issue is that DeepSpeed manages gradients internally and expects them to be None or valid
                            # Solution: Don't zero grad between bins, but accumulate gradients instead
                            # However, we need separate gradient norms per bin, so we need a different approach
                            
                            # Alternative: Store the gradient state before backward, then restore it
                            # Actually, let's try a simpler approach: only zero grad for the FIRST bin
                            if bin_idx == 0:
                                model.zero_grad()
                            
                            # Use retain_graph=True for ALL bins to keep computation graph
                            group_loss.backward(retain_graph=True)
                            
                            print(f"[DEBUG] Computing gradient norms after backward for bin {bin_idx}!")
                            
                            # For bins after the first, we need to compute the DIFFERENCE in gradient norm
                            # because gradients are accumulating
                            # Strategy: Save gradient state BEFORE backward, then compute difference AFTER backward
                            
                            # Save current gradient state (before backward for this bin)
                            prev_grads = {}
                            if bin_idx > 0:
                                for name, param in model.named_parameters():
                                    if param.grad is not None:
                                        prev_grads[name] = param.grad.data.clone()
                            
                            # Compute gradient norm AFTER backward
                            current_grad_norm_sq = 0.0
                            num_params_with_grad = 0
                            for param in model.parameters():
                                if param.grad is not None:
                                    try:
                                        param_norm = param.grad.data.norm(2).item() ** 2
                                        if torch.isfinite(torch.tensor(param_norm)):
                                            current_grad_norm_sq += param_norm
                                            num_params_with_grad += 1
                                    except Exception:
                                        pass
                            
                            if num_params_with_grad == 0:
                                print(f"[WARNING] Bin {bin_idx} has no parameters with valid gradients!")
                                results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': int(num_tokens)}
                                continue
                            
                            if bin_idx == 0:
                                # First bin: gradient norm is just the current norm
                                grad_norm = current_grad_norm_sq ** 0.5
                                prev_grad_norm_sq = current_grad_norm_sq
                            else:
                                # Subsequent bins: compute difference
                                # We saved prev_grads before backward, so compute prev norm squared
                                prev_grad_norm_sq_from_saved = 0.0
                                for name, param in model.named_parameters():
                                    if name in prev_grads:
                                        try:
                                            prev_norm = prev_grads[name].norm(2).item() ** 2
                                            if torch.isfinite(torch.tensor(prev_norm)):
                                                prev_grad_norm_sq_from_saved += prev_norm
                                        except Exception:
                                            pass
                                
                                # Compute this bin's gradient norm: sqrt(current^2 - prev^2)
                                bin_grad_norm_sq = max(0.0, current_grad_norm_sq - prev_grad_norm_sq_from_saved)
                                grad_norm = bin_grad_norm_sq ** 0.5
                                
                                # Update for next iteration
                                prev_grad_norm_sq = current_grad_norm_sq
                                
                                # Clean up saved gradients
                                del prev_grads
                            
                            if grad_norm == 0.0:
                                print(f"[WARNING] Bin {bin_idx} has zero gradient norm!")
                            
                            print(f"[DEBUG] Bin {bin_idx} gradient norm: {grad_norm:.4f}, params with grad: {num_params_with_grad}")
                            
                            # Store result
                            results[bin_idx] = {
                                'mean': float(grad_norm),
                                'std': 0.0,
                                'count': int(num_tokens)
                            }
                            
                        except Exception as e:
                            print(f"[WARNING] Failed to compute gradient for bin {bin_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                            results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': int(num_tokens)}
                            model.zero_grad()  # Clear gradients on error
                    
                    # Final cleanup - zero grad after all bins are processed
                    model.zero_grad()
                    
                    return results

                # Define bin boundaries (0.1 intervals, 10 bins)
                bins = torch.linspace(0.0, 1.0, 11).to(output.logits.device)

                # Step 1: Analyze probability distribution and create group masks
                response_log_probs, valid_mask = get_response_log_probs(output.logits, sequences, experience.action_mask)
                shift_labels = sequences[:, 1:].contiguous()
                probs, counts, group_masks = analyze_response_probs(response_log_probs, valid_mask, bins)
                
                # Debug: Show original probability distribution BEFORE filtering
                print(f"[DEBUG] ORIGINAL (before first-turn filter) probability distribution:")
                print(f"[DEBUG] Total valid tokens: {valid_mask.sum().item()}")
                if len(probs) > 0:
                    print(f"[DEBUG] Original probs: min={probs.min().item():.6f}, max={probs.max().item():.6f}, mean={probs.mean().item():.6f}")
                    print(f"[DEBUG] Original probs distribution: <0.1: {(probs < 0.1).sum().item()}, 0.1-0.5: {((probs >= 0.1) & (probs < 0.5)).sum().item()}, 0.5-0.9: {((probs >= 0.5) & (probs < 0.9)).sum().item()}, >=0.9: {(probs >= 0.9).sum().item()}")
                    print(f"[DEBUG] Original token counts per bin: {counts.tolist()}")

                # Step 2: Compute per-token gradient norms and aggregate by bins
                # Pass inputs, labels, and action_mask - function will filter to first turn and re-run forward
                print(f"[DEBUG] Computing gradient norms at global_step {global_steps}")
                
                grad_stats = compute_group_gradient_norms(
                    model=self.actor,
                    inputs=sequences,
                    labels=sequences,  # Labels are same as inputs in this case
                    action_mask=experience.action_mask,
                    group_masks=group_masks,  # Will be recomputed inside function
                    max_samples_per_bin=20,
                    ignore_index=-100
                )
                
                # Prepare results for saving
                results_dict = {
                    'global_step': int(global_steps),
                    'bins': []
                }
                
                # Print and collect results
                print(f"\n=== Gradient Norm Statistics by Probability Bin (Global Step {global_steps}) ===")
                for bin_idx in range(len(bins) - 1):
                    bin_start = bins[bin_idx].item()
                    bin_end = bins[bin_idx + 1].item()
                    stats = grad_stats[bin_idx]
                    print(f"Bin [{bin_start:.1f}, {bin_end:.1f}): "
                          f"Mean={stats['mean']:.6f}, Std={stats['std']:.6f}, Count={stats['count']}")
                    
                    results_dict['bins'].append({
                        'bin_start': float(bin_start),
                        'bin_end': float(bin_end),
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'count': stats['count']
                    })
                print("=" * 60 + "\n")
                
                # Save results to file
                if hasattr(self, 'ckpt_path') and self.ckpt_path:
                    output_dir = self.ckpt_path
                elif hasattr(self.args, 'ckpt_path') and self.args.ckpt_path:
                    output_dir = self.args.ckpt_path
                elif hasattr(self.args, 'save_path') and self.args.save_path:
                    output_dir = self.args.save_path
                else:
                    output_dir = "./grad_norm_results"
                
                print(f"[DEBUG] Saving to directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"grad_norm_stats_step_{global_steps}.json")
                
                with open(output_file, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                
                print(f"Gradient norm statistics saved to: {output_file}\n")
                print("Gradient analysis completed. Exiting program...")
                
                # Exit the program after saving gradient analysis
                import sys
                sys.exit(0)
                
            except Exception as e:
                print(f"[ERROR] Failed to compute/save gradient norm statistics: {e}")
                import traceback
                traceback.print_exc()
                # Exit even on error
                import sys
                sys.exit(1)

        # [add]------------------------[add]
        # def compute_entropy_from_output(output, response_length):
        #     logits = output.logits
        #     probs = torch.softmax(logits, dim=-1)
        #     log_probs = torch.log_softmax(logits, dim=-1)
        #     entropy = -torch.sum(probs * log_probs, dim=-1)  

        #     all_entropy = []
        #     all_mask = []
        #     for i in range(entropy.size(0)):
        #         entropy_slice = entropy[i, -int(response_length[i].item()) - 1:-1] 
        #         mask_slice = attention_mask[i, -int(entropy_slice.size(0)):]
        #         all_entropy.append(entropy_slice)
        #         all_mask.append(mask_slice)

        #     all_entropy = torch.cat(all_entropy) 
        #     all_mask = torch.cat(all_mask)      
        #     entropy_loss = torch.sum(all_entropy * all_mask) / torch.sum(all_mask)
        #     return entropy_loss

        def compute_entropy_from_output(output, action_mask):
            logits = output.logits[:, :-1]  # 去掉最后一个预测 (B, T-1, V)
            probs = torch.softmax(logits, dim=-1)         # (B, T-1, V)
            log_probs = torch.log_softmax(logits, dim=-1) # (B, T-1, V)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, T-1)

            mask = action_mask.to(dtype=entropy.dtype, device=entropy.device)
            entropy_loss = torch.sum(entropy * mask) / torch.clamp(torch.sum(mask), min=1.0)
            return entropy_loss


        # entropy_loss = compute_entropy_from_output(output, action_mask)
        # metrics =  {"entropy_loss": entropy_loss.item()}

        
        # if self._wandb is not None and self.strategy.is_rank_0() and self.count % 20 == 0:
        #     # print(f'[log count]: {self.count}')
        #     for k, v in metrics.items():
        #         self._wandb.log({f"train/{k}": v}, step=self.count)
        # self.count += 1



        if not self.without_ppo:
            sft_loss, sft_info = self.actor_loss_fn_sft(
                action_log_probs,
                old_action_log_probs,
                advantages,
                action_mask=experience.action_mask,
                return_info=True
            )
            
            # [add] DFT Baseline mode: reweight loss based on model confidence
            if self.dft_baseline:
                # Log DFT baseline activation

                logits = output.logits
                
                # Shift logits and labels for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
                shift_labels = sequences[:, 1:].contiguous()   # (B, T-1)
                
                # Reshape for loss calculation
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                
                # Calculate probabilities for reweighting
                probs = torch.softmax(shift_logits, dim=-1)
                prob_coefficients = probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Collect step probabilities for logging
                if self.save_step_probs:
                    # Reshape probabilities back to batch format for step-wise collection
                    batch_size = sequences.size(0)
                    seq_len = sequences.size(1) - 1  # -1 because we shifted
                    
                    # Use action mask that matches the shifted dimensions (T-1 instead of T)
                    # prob_coefficients corresponds to shift_logits which has shape (B, T-1)
                    valid_mask = experience.action_mask[:, :-1].contiguous()  # (B, T-1) to match shift_logits
                    
                    # Reshape prob_coefficients to match the mask shape
                    step_probs = prob_coefficients.view(batch_size, seq_len)
                    
                    # Apply mask to get only valid steps
                    masked_step_probs = step_probs * valid_mask.to(step_probs.device)
                    
                    # Collect probabilities for each batch (only non-zero values)
                    batch_step_probs = []
                    for i in range(batch_size):
                        valid_probs = masked_step_probs[i][valid_mask[i].bool()].cpu().tolist()
                        if valid_probs:  # Only add if there are valid probabilities
                            batch_step_probs.append(valid_probs)
                    
                    if batch_step_probs:  # Only add if we have valid data
                        self.step_probabilities.append(batch_step_probs)
                
                # Create loss mask from action mask
                loss_mask = experience.action_mask[:, :-1].contiguous().view(-1)  # (B*(T-1),)
                
                # Apply reweighting: higher confidence tokens get higher weight
                sft_loss = sft_loss * prob_coefficients.detach()
                sft_loss = sft_loss * loss_mask.to(sft_loss.device)
                
                # Normalize by the sum of weights
                weight_sum = (prob_coefficients.detach() * loss_mask.to(prob_coefficients.device)).sum()
                if weight_sum > 0:
                    sft_loss = sft_loss.sum() / weight_sum
                else:
                    sft_loss = sft_loss.sum()
                

       
        actor_loss, info = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            return_info=True
        )

        # Collect step probabilities for logging (outside DFT baseline block)
        if self.save_step_probs and not self.dft_baseline:
            # Get logits from the actor output
            logits = output.logits
            
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
            shift_labels = sequences[:, 1:].contiguous()   # (B, T-1)
            
            # Reshape for probability calculation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            # Calculate probabilities
            probs = torch.softmax(shift_logits, dim=-1)
            prob_coefficients = probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Reshape probabilities back to batch format for step-wise collection
            batch_size = sequences.size(0)
            seq_len = sequences.size(1) - 1  # -1 because we shifted
            
            # Reshape prob_coefficients to match the mask shape
            step_probs = prob_coefficients.view(batch_size, seq_len)
            
            # Use action mask that matches the shifted dimensions (T-1 instead of T)
            # Check if action_mask has more than 1 element to avoid empty slice
            if experience.action_mask.size(1) > 1:
                valid_mask = experience.action_mask[:, :-1].contiguous()  # (B, T-1) to match shift_logits
            else:
                # If action_mask is (B, 1), we can't slice it, so create a zero mask with matching shape
                # This should not happen in normal cases, but handle it gracefully
                valid_mask = torch.zeros_like(step_probs, dtype=torch.long)
            
            # Ensure dimensions match before applying mask
            if valid_mask.size(1) == step_probs.size(1):
                # Apply mask to get only valid steps
                masked_step_probs = step_probs * valid_mask.to(step_probs.device)
                
                # Collect probabilities for each batch (only non-zero values)
                batch_step_probs = []
                for i in range(batch_size):
                    valid_probs = masked_step_probs[i][valid_mask[i].bool()].cpu().tolist()
                    if valid_probs:  # Only add if there are valid probabilities
                        batch_step_probs.append(valid_probs)
                
                if batch_step_probs:  # Only add if we have valid data
                    self.step_probabilities.append(batch_step_probs)
            # If dimensions don't match, skip probability collection for this batch

            ## openorca
            # # prob_coefficients corresponds to shift_logits which has shape (B, T-1)
            # valid_mask = experience.action_mask[:, :-1].contiguous()  # (B, T-1) to match shift_logits
            

            # ####### openorca instruction finetuning 
            # # step_probs = prob_coefficients.view(valid_mask.shape)
            # # Reshape prob_coefficients to match the mask shape
            # step_probs = prob_coefficients.view(batch_size, seq_len)
            
            # # Apply mask to get only valid steps
            # masked_step_probs = step_probs * valid_mask.to(step_probs.device)
            
            # # Collect probabilities for each batch (only non-zero values)
            # batch_step_probs = []
            # for i in range(batch_size):
            #     valid_probs = masked_step_probs[i][valid_mask[i].bool()].cpu().tolist()
            #     if valid_probs:  # Only add if there are valid probabilities
            #         batch_step_probs.append(valid_probs)
            
            # if batch_step_probs:  # Only add if we have valid data
            #     self.step_probabilities.append(batch_step_probs)

        # [lhy add]
        if not self.without_ppo:

            clip_info = {"clip_over_ratio": info["clip_over_ratio"]   ,"clip_under_ratio":info["clip_under_ratio"], "ratio_in_clip2": info["ratio_in_clip"] }
            if self._wandb is not None and self.strategy.is_rank_0():
                for k, v in clip_info.items():
                    # print("dhakdjas")
                    # print(k,v)
                    self._wandb.log({f"train/{k}": v}, step=self.count)



            for key, count in info['all_logprob_bins'].items():
                if key in self.all_logprob_bins:
                    self.all_logprob_bins[key] += count
                else:
                    self.all_logprob_bins[key] = count
        
            for key, count in info['under_logprob_bins'].items():
                if key in self.under_logprob_bins:
                    self.under_logprob_bins[key] += count
                else:
                    self.under_logprob_bins[key] = count

            for key, count in info['over_logprob_bins'].items():
                if key in self.over_logprob_bins:
                    self.over_logprob_bins[key] += count
                else:
                    self.over_logprob_bins[key] = count
            
            def visualize_clip_distribution(save_dir: str):
                import matplotlib.pyplot as plt
                from datetime import datetime
                os.makedirs(save_dir, exist_ok=True)

                def prepare_bins_and_counts(count_dict):
                    # 确保区间按升序排列
                    sorted_items = sorted(count_dict.items(), key=lambda x: float(x[0].split('-')[0].strip('<>=')))
                    bin_edges = []
                    counts = []
                    for i, (k, v) in enumerate(sorted_items):
                        if '-' in k:
                            left, right = map(float, k.split('-'))
                            if i == 0:
                                bin_edges.append(left)
                            bin_edges.append(right)
                        elif k.startswith('>='):
                            # 为 >= 最后一个 bin 添加一个更大上限
                            left = float(k[2:].strip())
                            if i == 0:
                                bin_edges.append(left)
                            bin_edges.append(left + 0.1)
                        elif k.startswith('<'):
                            right = float(k[1:].strip())
                            bin_edges = [right - 0.1, right] + bin_edges  # 插入在前
                        counts.append(v)
                    # 转为 tensor
                    bins = torch.tensor(bin_edges)
                    counts = torch.tensor(counts)
                    return bins, counts

                def plot_prob_distribution(bins: torch.Tensor, counts: torch.Tensor, save_path: str = None, mode='over'):
                    plt.figure(figsize=(10, 6))
                    plt.bar(
                        x=bins[:-1],  # 区间左边界
                        height=counts.cpu().numpy(),  # 区间样本数量,
                        width=0.1,  # 区间宽度（0.1）
                        align='edge',  # 对齐左边界
                        edgecolor='black',
                        alpha=0.7
                    )
                    plt.xticks(bins)  # 显示所有区间边界
                    plt.xlabel('Probability Interval [left, right)')
                    plt.ylabel('Number of Tokens')
                    plt.title('Distribution of Response Token Probabilities')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"prob_distribution_{timestamp}_{mode}.png"
                    save_path = os.path.join(save_path, filename)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close()


                def plot_overlay_prob_distributions(
                    all_bins_dict: Dict[str, int],
                    over_bins_dict: Dict[str, int],
                    save_path: str = None,
                ):
                    import matplotlib.pyplot as plt
                    import torch
                    import os
                    from datetime import datetime
                    from typing import Dict, Tuple

                    # 准备 bins 和 counts
                    def prepare_bins_and_counts(bins_dict: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
                        bin_keys = sorted(bins_dict.keys(), key=lambda x: float(x.split('-')[0]) if '-' in x else float(x[2:]))
                        left_edges = []
                        counts = []
                        for k in bin_keys:
                            if '-' in k:
                                left = float(k.split('-')[0])
                            else:
                                left = float(k[2:])  # e.g., ">= 1.0"
                            left_edges.append(left)
                            counts.append(bins_dict[k])
                        last_right = left_edges[-1] + 0.1
                        bins_tensor = torch.tensor(left_edges + [last_right])
                        counts_tensor = torch.tensor(counts)
                        return bins_tensor, counts_tensor

                    all_bins, all_counts = prepare_bins_and_counts(all_bins_dict)
                    over_bins, over_counts = prepare_bins_and_counts(over_bins_dict)

                    # 绘图
                    plt.figure(figsize=(10, 6))

                    width = 0.04  # 柱子宽度（设置略小避免重叠）

                    # 红色：all log probs
                    plt.bar(
                        x=all_bins[:-1],
                        height=all_counts.cpu().numpy(),
                        width=width,
                        align='edge',
                        edgecolor='black',
                        alpha=0.6,
                        color='red',
                        label='All LogProbs'
                    )

                    # 蓝色：over clipped log probs
                    plt.bar(
                        x=over_bins[:-1],
                        height=over_counts.cpu().numpy(),
                        width=width,
                        align='edge',
                        edgecolor='black',
                        alpha=0.7,
                        color='blue',
                        label='Over-Clipped LogProbs'
                    )

                    # 设置坐标轴和标题
                    plt.xticks(all_bins, rotation=45)
                    plt.xlabel('Probability Interval [left, right)')
                    plt.ylabel('Number of Tokens')
                    plt.title('Log Probability Distribution: All vs Over-Clipped')
                    plt.legend()
                    plt.grid(axis='y', linestyle='--', alpha=0.7)

                    # 保存图像
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"logprob_overlay_{timestamp}.png"
                    save_file = os.path.join(save_path, filename)
                    plt.savefig(save_file, dpi=300, bbox_inches='tight')
                    plt.close()

        
                # -------- all CLIP --------
                if len(self.all_logprob_bins) > 0 and len(self.over_logprob_bins) > 0:
                    bins_all, counts_all = prepare_bins_and_counts(self.all_logprob_bins)
                    bins_under, counts_under = prepare_bins_and_counts(self.under_logprob_bins)
                    plot_overlay_prob_distributions(all_bins_dict=self.all_logprob_bins,over_bins_dict=self.over_logprob_bins,save_path=save_dir)

                
            if self.count % 20==0:
                print('==============================')
                print(self.over_logprob_bins)
                print(self.under_logprob_bins)
                print(self.all_logprob_bins)
                print('==============================')

                os.makedirs(self.ckpt_path, exist_ok=True)
                log_file = os.path.join(self.ckpt_path, "all_logprob_bins.txt")
                with open(log_file, "a") as f:
                    f.write(f"[step {self.count}]: {self.all_logprob_bins}\n")
            
                visualize_clip_distribution(self.ckpt_path)

                
        self.count += 1
        
        # [lhy add]


        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0

        # [replace]-------------------------[replace]
        # loss = actor_loss + aux_loss * self.args.aux_loss_coef
        loss = actor_loss + aux_loss * self.args.aux_loss_coef #-  self.entropy_coef * entropy_loss
        # [replace]-------------------------[replace]



        # Compute gradient norm for actor using independent loss to avoid interference
        # Create independent loss by re-running forward pass

            # Re-run forward pass to get independent loss
        action_log_probs_indep, output_indep = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        
        # Compute independent actor loss
        actor_loss_indep, _ = self.actor_loss_fn(
            action_log_probs_indep,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            return_info=True
        )
        
        # Add aux loss if needed
        if self.aux_loss:
            aux_loss_indep = output_indep.aux_loss
        else:
            aux_loss_indep = 0
        loss_indep = actor_loss_indep + aux_loss_indep * self.args.aux_loss_coef
        
        # Compute gradients using independent loss
        self.actor.zero_grad()
        loss_indep.backward(retain_graph=True)
        actor_grad_norm = compute_gradient_norm(self.actor)
        self.actor_gradient_norms.append(actor_grad_norm)
        
        # Collect probability-gradient pairs if enabled (after backward to get gradients)
        if self.save_prob_gradient_pairs:
            self._collect_prob_gradient_pairs(
                action_log_probs_indep, 
                old_action_log_probs, 
                advantages, 
                experience.action_mask
            )
            # Save immediately after collection (don't wait for logging_steps)
            if self.prob_gradient_pairs and self.prob_gradient_log_path and self.strategy.is_rank_0():
                self._save_prob_gradient_pairs(0)  # Use step 0 to indicate immediate save
                self.prob_gradient_pairs = []  # Clear after saving
        
        # self.actor.zero_grad()  # Clear gradients after computation

        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        # [replace]---------------------------[replace]
        # status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0], "ratio_in_clip": info["ratio_in_clip"]}
        if self.without_ppo:
            status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0], "ratio_in_clip": info["ratio_in_clip"]}
        else:
            status = {"policy_loss": actor_loss.item(), "sft_loss": sft_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0], "ratio_in_clip": info["ratio_in_clip"]}
        
        # Add gradient norm information to status
        if self.actor_gradient_norms:
            status["actor_gradient_norm"] = self.actor_gradient_norms[-1]  # Most recent actor gradient norm
        # [replace]---------------------------[replace]
        
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        
        # [NEW] Add token clip statistics to status
        if not self.without_ppo and "ratio_in_clip" in info:
            status["num_tokens"] = info["num_tokens"]
            status["num_clipped_tokens"] = info["num_clipped_tokens"]
        
        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # Add actor gradient information to logs
            if self.actor_gradient_norms:
                grad_stats = compute_actor_gradient_statistics(self.actor_gradient_norms)
                logs_dict.update(grad_stats)
                
                # Save actor gradient logs to text file
                if self.strategy.is_rank_0() and self.gradient_log_path:
                    # Save the most recent actor gradient norm and std
                    save_gradient_logs(self.actor_gradient_norms, self.gradient_log_path, global_step, self.save_gradient_norms)
                
                # Reset gradient norms for next logging step
                self.actor_gradient_norms = []
            
            # Add step probability information to logs
            if self.step_probabilities:
                prob_stats = compute_step_prob_statistics(self.step_probabilities)
                logs_dict.update(prob_stats)
                
                # Save step probability logs to text file
                if self.strategy.is_rank_0() and self.step_prob_log_path:
                    # Save the most recent step probabilities
                    save_step_prob_logs(self.step_probabilities, self.step_prob_log_path, global_step, self.save_step_probs)
                
                # Reset step probabilities for next logging step
                self.step_probabilities = []
            
            # Save probability-gradient pairs
            if self.prob_gradient_pairs and self.prob_gradient_log_path:
                if self.strategy.is_rank_0():
                    self._save_prob_gradient_pairs(global_step)
                self.prob_gradient_pairs = []
            
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, global_step, client_states)

    def _collect_prob_gradient_pairs(self, action_log_probs, old_action_log_probs, advantages, action_mask):
        """
        Collect probability pairs only (no gradient calculation).
        
        Args:
            action_log_probs: Current action log probabilities (B, T)
            old_action_log_probs: Old action log probabilities (B, T)
            advantages: Advantage values (B, T)
            action_mask: Mask for valid actions (B, T)
        """
        device = action_log_probs.device
        
        # Calculate probability from log probability: prob = exp(log_prob)
        token_probs = torch.exp(action_log_probs)
        
        # Collect probability pairs (gradient will be calculated later or separately)
        masked_probs = token_probs * action_mask.to(device)
        
        batch_size = action_log_probs.size(0)
        for i in range(batch_size):
            valid_indices = action_mask[i].bool()
            valid_probs = masked_probs[i][valid_indices].cpu()
            
            for prob in valid_probs:
                if prob.item() > 0:  # Only add non-zero probabilities
                    # Use 0.0 as placeholder for gradient (to be calculated separately)
                    self.prob_gradient_pairs.append((prob.item(), 0.0))
    
    def _save_prob_gradient_pairs(self, global_step):
        """Save probability-gradient pairs to log file and analyze."""
        if self.prob_gradient_log_path and self.prob_gradient_pairs:
            # Save raw data
            save_prob_gradient_pairs_logs(
                self.prob_gradient_pairs,
                self.prob_gradient_log_path,
                global_step
            )
            
            # Analyze and visualize if enabled
            if HAS_ANALYSIS and len(self.prob_gradient_pairs) > 0:
                output_dir = self.prob_gradient_output_dir or os.path.join(self.ckpt_path, "prob_gradient")
                os.makedirs(output_dir, exist_ok=True)
                
                # Analyze distribution
                results = analyze_prob_gradient_distribution(self.prob_gradient_pairs, num_bins=10)
                
                # Save TXT
                txt_path = os.path.join(output_dir, "prob_gradient_data.txt")
                save_analysis_results_txt(results, txt_path)
                
                # Save PNG
                fig_path = os.path.join(output_dir, "prob_gradient_distribution.png")
                visualize_prob_gradient_distribution(results, fig_path, title="Token Probability vs Gradient Distribution")
                
                if self.strategy.is_rank_0():
                    self.strategy.print(f"Analysis saved to: {txt_path} and {fig_path}")

    def _save_checkpoint(self, args, tag, global_step, client_states=None):
        save_path = os.path.join(args.ckpt_path, "_actor",  f"global_step_{global_step}")
        os.makedirs(save_path, exist_ok=True)
        self.strategy.save_model(
            self.actor,
            self.tokenizer,
            save_path,
        )

        # Original checkpoint saving for training resumption
        # self.strategy.save_ckpt(
        #     self.actor.model,
        #     os.path.join(args.ckpt_path, "_actor"),
        #     tag,
        #     args.max_ckpt_num,
        #     args.max_ckpt_mem,
        #     client_states,
        # )
        # if self.critic is not None:
        #     self.strategy.save_ckpt(
        #         self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
        #     )

        # Save in Hugging Face format
        # if self.strategy.is_rank_0():  # Only save on main process
        #     save_path = os.path.join(args.ckpt_path, "_actor", "hf_checkpoint",  tag)
        #     os.makedirs(save_path, exist_ok=True)
            
        #     # Save model weights
        #     self.actor.model.save_pretrained(save_path)
            
        #     # Save tokenizer if available
        #     if self.tokenizer is not None:
        #         self.tokenizer.save_pretrained(save_path)
            
        #     # Save training arguments
        #     if hasattr(args, 'to_json_file'):
        #         args.to_json_file(os.path.join(save_path, "training_args.json"))
