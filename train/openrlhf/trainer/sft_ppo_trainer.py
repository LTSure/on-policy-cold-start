import math
import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss, ReinforceLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer, NaiveExperienceMakerORM800K, NaiveExperienceMakerPRM800K, NaiveExperienceMakerPRM800K_BOX, NaiveExperienceMakerSFT, NaiveExperienceMakerSFT_MT


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
        use_muti_turn: bool = False,
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
        # [add]------------------------[add]

        if without_ppo:
            self.actor_loss_fn = ReinforceLoss(eps_clip)
        else:
            # [replace]------------------------[replace]
            # self.actor_loss_fn = PolicyLoss(eps_clip)
            self.actor_loss_fn = PolicyLoss(eps_clip, use_decay=use_decay)
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
        # print(f'{consumed_samples} === {args.rollout_batch_size }==== {num_rollouts_per_episodes}')
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
                # self.replay_buffer.normalize("advantages", self.strategy)
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
                self._save_checkpoint(args, tag=str(episode+1))

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
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        return status

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
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


        entropy_loss = compute_entropy_from_output(output, action_mask)
        metrics =  {"entropy_loss": entropy_loss.item()}

        
        if self._wandb is not None and self.strategy.is_rank_0() and self.count % 20 == 0:
            # print(f'[log count]: {self.count}')
            for k, v in metrics.items():
                self._wandb.log({f"train/{k}": v}, step=self.count)
        # self.count += 1



        if not self.without_ppo:
            sft_loss, sft_info = self.actor_loss_fn_sft(
                action_log_probs,
                old_action_log_probs,
                advantages,
                action_mask=experience.action_mask,
                return_info=True
            )

        # [add]------------------------[add]
             
       
        actor_loss, info = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            return_info=True
        )


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
        loss = actor_loss + aux_loss * self.args.aux_loss_coef -  self.entropy_coef * entropy_loss
        # [replace]-------------------------[replace]



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
        # if global_step % args.save_steps == 0:
        #     tag = f"global_step{global_step}"
        #     self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states=None):
        save_path = os.path.join(args.ckpt_path, "_actor",  tag)
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
