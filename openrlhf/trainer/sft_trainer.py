import math
import os
import json
from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm
from transformers.trainer import get_scheduler
import numpy as np

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
                settings=wandb.Settings(init_timeout=300),
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
        if num_update_steps_per_epoch is None or num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
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
                # Gradient norm analysis - only run conditionally to avoid overhead
                # Run when: step % accumulated_gradient == 0 (at optimizer step) AND global_step % logging_steps == 0
                # With logging_steps=1, this runs every optimizer step
                run_grad_analysis = False
                if self.strategy.is_rank_0() and step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    if hasattr(self.args, 'logging_steps') and self.args.logging_steps > 0:
                        # Run every logging_steps global steps
                        if global_step % self.args.logging_steps == 0:
                            run_grad_analysis = True
                    else:
                        # If no logging_steps specified, run every accumulated_gradient steps (every optimizer step)
                        run_grad_analysis = True
                
                if run_grad_analysis:
                    try:
                        def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                            log_probs = F.log_softmax(logits, dim=-1)  
                            return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  

                        def get_response_log_probs(output_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                            shift_logits = output_logits[:, :-1, :].contiguous()
                            shift_labels = labels[:, 1:].contiguous()
                            
                            all_log_probs = log_probs_from_logits(shift_logits, shift_labels)  
                            
                            valid_mask = (shift_labels != self.loss_fn.IGNORE_INDEX)
                            
                            response_log_probs = all_log_probs[valid_mask]
                            
                            return response_log_probs, valid_mask

                        def analyze_response_probs(response_log_probs: torch.Tensor, valid_mask: torch.Tensor, bins: torch.Tensor) -> tuple:
                            """
                            Analyze response probabilities and create group masks for each bin.
                            
                            Returns:
                                probs: probabilities for each valid token
                                counts: count of tokens in each bin
                                group_masks: list of masks, one per bin, indicating which tokens belong to each bin
                            """
                            if len(response_log_probs) == 0:
                                # Return empty results if no valid tokens
                                group_masks = [torch.zeros_like(valid_mask, dtype=torch.bool) for _ in range(len(bins) - 1)]
                                counts = torch.zeros(len(bins) - 1, dtype=torch.float32)
                                return torch.tensor([]), counts, group_masks
                            
                            probs = torch.exp(response_log_probs)
                            
                            # Create a full-size mask tensor matching shift_labels shape
                            full_probs = torch.zeros_like(valid_mask, dtype=torch.float32)
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
                            shift_logits: torch.Tensor,
                            shift_labels: torch.Tensor,
                            group_masks: list,
                            max_samples_per_bin: int = 50,
                            ignore_index: int = -100
                        ) -> dict:
                            """
                            Compute per-token gradient norms for each probability bin and return mean/std statistics.
                            """
                            model.train()
                            results = {}
                            
                            # Flatten for easier indexing
                            batch_size, seq_len_minus_one, vocab_size = shift_logits.shape
                            shift_logits_flat = shift_logits.view(-1, vocab_size)
                            shift_labels_flat = shift_labels.view(-1)
                            
                            # Compute per-token losses
                            per_token_losses = F.cross_entropy(
                                shift_logits_flat,
                                shift_labels_flat,
                                reduction='none',
                                ignore_index=ignore_index
                            )
                            
                            # Process each bin
                            for bin_idx, mask in enumerate(group_masks):
                                if not mask.any():
                                    results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': 0}
                                    continue
                                
                                # Get indices of tokens in this bin
                                mask_flat = mask.view(-1)
                                bin_indices = mask_flat.nonzero(as_tuple=True)[0].cpu().numpy()
                                
                                if len(bin_indices) == 0:
                                    results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': 0}
                                    continue
                                
                                # Sample tokens if too many
                                if len(bin_indices) > max_samples_per_bin:
                                    sampled_indices = np.random.choice(bin_indices, max_samples_per_bin, replace=False)
                                else:
                                    sampled_indices = bin_indices
                                
                                # Compute gradient norm for each sampled token
                                grad_norms = []
                                for idx in sampled_indices:
                                    model.zero_grad()
                                    token_loss = per_token_losses[idx]
                                    token_loss.backward(retain_graph=True)
                                    
                                    # Compute gradient norm
                                    grad_norm = 0.0
                                    for param in model.parameters():
                                        if param.grad is not None:
                                            grad_norm += param.grad.data.norm(2).item() ** 2
                                    grad_norm = grad_norm ** 0.5
                                    grad_norms.append(grad_norm)
                                
                                if len(grad_norms) > 0:
                                    results[bin_idx] = {
                                        'mean': float(np.mean(grad_norms)),
                                        'std': float(np.std(grad_norms)),
                                        'count': int(len(bin_indices))
                                    }
                                else:
                                    results[bin_idx] = {'mean': 0.0, 'std': 0.0, 'count': 0}
                            
                            return results

                        # Define bin boundaries (0.1 intervals, 10 bins)
                        bins = torch.linspace(0.0, 1.0, 11).to(output.logits.device)

                        # Step 1: Analyze probability distribution and create group masks
                        response_log_probs, valid_mask = get_response_log_probs(output.logits, labels)
                        shift_labels = labels[:, 1:].contiguous()
                        probs, counts, group_masks = analyze_response_probs(response_log_probs, valid_mask, bins)

                        # Step 2: Compute per-token gradient norms and aggregate by bins
                        global_step = step // self.strategy.accumulated_gradient
                        shift_logits = output.logits[:, :-1, :].contiguous()
                        
                        print(f"[DEBUG] Computing gradient norms at step {step}, global_step {global_step}")
                        
                        grad_stats = compute_group_gradient_norms(
                            model=self.model,
                            shift_logits=shift_logits,
                            shift_labels=shift_labels,
                            group_masks=group_masks,
                            max_samples_per_bin=50,
                            ignore_index=self.loss_fn.IGNORE_INDEX
                        )
                        
                        # Prepare results for saving
                        results_dict = {
                            'step': int(step),
                            'global_step': int(global_step),
                            'bins': []
                        }
                        
                        # Print and collect results
                        print(f"\n=== Gradient Norm Statistics by Probability Bin (Step {step}, Global Step {global_step}) ===")
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
                        if hasattr(self.args, 'ckpt_path') and self.args.ckpt_path:
                            output_dir = self.args.ckpt_path
                        elif hasattr(self.args, 'save_path') and self.args.save_path:
                            output_dir = self.args.save_path
                        else:
                            output_dir = "./grad_norm_results"
                        
                        print(f"[DEBUG] Saving to directory: {output_dir}")
                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, f"grad_norm_stats_step_{global_step}.json")
                        
                        with open(output_file, 'w') as f:
                            json.dump(results_dict, f, indent=2)
                        
                        print(f"Gradient norm statistics saved to: {output_file}\n")
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to compute/save gradient norm statistics: {e}")
                        import traceback
                        traceback.print_exc()
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
