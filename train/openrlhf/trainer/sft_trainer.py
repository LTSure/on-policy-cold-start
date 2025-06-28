import math
import os
from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.models import GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class SFTTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args

        self.loss_fn = GPTLMLoss()

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

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
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for prompts_id_lens, inputs, attention_masks, infos in self.train_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                output = self.model(inputs, attention_mask=attention_mask, return_output=True)

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompts_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompts_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX


#############################################3

                def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                    log_probs = F.log_softmax(logits, dim=-1)  
                    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  

                def get_response_log_probs(output_logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
                    shift_logits = output_logits[:, :-1, :].contiguous()  # 形状 (batch_size, seq_len-1, vocab_size)
                    shift_labels = labels[:, 1:].contiguous()             # 形状 (batch_size, seq_len-1)
                    
                    all_log_probs = log_probs_from_logits(shift_logits, shift_labels)  
                    
                    valid_mask = (shift_labels != self.loss_fn.IGNORE_INDEX)  # 布尔掩码，True表示response位置
                    
                    response_log_probs = all_log_probs[valid_mask]  # 仅保留有效位置的log prob
                    
                    return response_log_probs

                response_log_probs = get_response_log_probs(output.logits, labels)

                def analyze_response_probs(response_log_probs: torch.Tensor, bins: torch.Tensor) -> tuple:
 

                    probs = torch.exp(response_log_probs)  # log概率转实际概率
                    counts = torch.histc(probs, bins=bins, min=bins[0], max=bins[-1])
                    return probs, counts

                def plot_prob_distribution(bins: torch.Tensor, counts: torch.Tensor):
                    """
                    绘制概率区间分布的柱状图
                    
                    Args:
                        bins: 区间边界（如[0.0, 0.1, ..., 1.0]）
                        counts: 各区间的样本数量（与bins长度-1一致）
                    """
                    plt.figure(figsize=(10, 6))
                    plt.bar(
                        x=bins[:-1],  # 区间左边界
                        height=counts,
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
                    plt.show()

                def compute_group_gradient_norms(
                    model: torch.nn.Module,
                    inputs: torch.Tensor,
                    labels: torch.Tensor,
                    group_masks: list,
                    ignore_index: int = -100
                ) -> list:
                    """
                    计算各分组样本的梯度范数
                    
                    Args:
                        model: 训练中的模型
                        inputs: 模型输入（如token_ids）
                        labels: 原始标签（已标记IGNORE_INDEX）
                        group_masks: 各区间的mask列表（与shift_labels同形状）
                        ignore_index: 忽略的标签值
                    
                    Returns:
                        grad_norms: 各区间的梯度范数列表
                    """
                    model.train()  # 确保模型处于训练模式
                    grad_norms = []
                    
                    # 模型前向传播（获取logits）
                    output = model(inputs)
                    output_logits = output.logits  # 假设模型输出包含logits
                    
                    # 计算shift_logits和shift_labels（与损失计算对齐）
                    shift_logits = output_logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    
                    # 遍历每个分组mask
                    for mask in group_masks:
                        if not mask.any():  # 跳过无样本的分组
                            grad_norms.append(0.0)
                            continue
                        
                        # 提取分组内的logits和labels
                        group_logits = shift_logits[mask]
                        group_labels = shift_labels[mask]
                        
                        # 计算分组损失（与原loss_fn逻辑一致）
                        # 原loss_fn是CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                        # 分组损失需排除IGNORE_INDEX（但mask已过滤，无需额外处理）
                        group_loss = F.cross_entropy(
                            group_logits,  # 形状 (num_group_tokens, vocab_size)
                            group_labels,  # 形状 (num_group_tokens,)
                            ignore_index=ignore_index
                        )
                        
                        # 反向传播并计算梯度范数
                        model.zero_grad()
                        group_loss.backward(retain_graph=True)  # 保留计算图以便后续分组
                        
                        # 计算参数梯度的L2范数
                        grad_norm = 0.0
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        grad_norms.append(grad_norm)
                    
                    return grad_norms

                # 定义区间边界（0.1间隔，共10个区间）
                bins = torch.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0]

                # -------------------- 步骤1：分析概率分布并生成mask --------------------
                # 需传入valid_mask（在get_response_log_probs中生成）
                shift_labels = labels[:, 1:].contiguous()  # 与get_response_log_probs中的shift_labels一致
                valid_mask = (shift_labels != ignore_index)  # 有效位置掩码

                probs, counts, group_masks = analyze_response_probs(response_log_probs, bins)

                # -------------------- 步骤2：绘制柱状图 --------------------
                plot_prob_distribution(bins, counts)

                # -------------------- 步骤3：计算分组梯度范数（假设model是当前模型） --------------------
                # 模拟模型输入（假设inputs是token_ids）
                inputs = torch.randint(0, vocab_size, (batch_size, seq_len))  # 示例输入

                # 计算各分组的梯度范数
                grad_norms = compute_group_gradient_norms(
                    model=model,  # 替换为实际模型
                    inputs=inputs,
                    labels=labels,
                    group_masks=group_masks,
                    ignore_index=ignore_index
                )

                # 打印结果
                for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
                    print(f"Bin [{bin_start:.1f}, {bin_end:.1f}): Gradient Norm = {grad_norms[i]:.4f}")
########################################################3

                gpt_loss = self.loss_fn(output.logits, labels)
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_lens, inputs, attention_masks, infos in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                output = self.model(inputs, attention_mask=attention_mask, return_output=True)

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompts_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompts_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
