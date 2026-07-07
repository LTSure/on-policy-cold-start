"""
Training script for Continual Learning Baselines
This script trains SFT models with different continual learning methods to compare against the PPO-SFT approach.

Baselines implemented:
1. Vanilla SFT (no forgetting mitigation)
2. Replay (mix old and new data)
3. EWC (Elastic Weight Consolidation)
4. L2 Regularization
5. KL Regularization
"""

import argparse
import math
import os
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset, blending_datasets
from openrlhf.models import Actor
from openrlhf.models.continual_learning_loss import (
    ReplayLoss,
    EWCLoss,
    L2RegularizationLoss,
    KLRegularizationLoss,
    compute_fisher_information,
    save_old_parameters,
)
from openrlhf.utils import get_strategy, get_tokenizer


def train(args):
    # Configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    # Load model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    
    if args.actor_init_on_gpu:
        model = model.to(torch.cuda.current_device())
    
    # Load tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, 
        model.model, 
        "left", 
        strategy, 
        use_fast=not args.disable_fast_tokenizer
    )
    
    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
    
    # Configure optimizer
    optim = strategy.create_optimizer(
        model, 
        lr=args.learning_rate, 
        betas=args.adam_betas, 
        weight_decay=args.l2
    )
    
    # Prepare datasets based on baseline method
    if args.baseline_method == "replay":
        # Mix old (SFT) and new data
        strategy.print("Using Replay baseline - mixing old and new data")
        
        # Load SFT data (old task)
        sft_data = blending_datasets(
            args.sft_data,
            args.sft_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples // 2,  # Use half for old data
            return_eval=False,
            train_split=args.train_split,
        )
        
        # Load new task data
        new_data = blending_datasets(
            args.dataset,
            args.dataset_probs,
            strategy,
            args.seed,
            max_count=args.max_samples // 2,  # Use half for new data
            return_eval=False,
            train_split=args.train_split,
        )
        
        # Concatenate datasets
        from datasets import concatenate_datasets
        train_data = concatenate_datasets([sft_data, new_data])
        
    else:
        # Standard loading for other methods
        train_data = blending_datasets(
            args.dataset,
            args.dataset_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.train_split,
        )
    
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )
    
    # For EWC: compute Fisher information on SFT data
    fisher_dict = None
    old_params_dict = None
    
    if args.baseline_method in ["ewc", "l2", "kl"]:
        strategy.print(f"Computing Fisher information or saving old parameters for {args.baseline_method}...")
        
        # Save old parameters
        old_params_dict = save_old_parameters(model.model)
        
        # For EWC, also compute Fisher information
        if args.baseline_method == "ewc":
            # Load a small subset of SFT data for Fisher computation
            sft_data_fisher = blending_datasets(
                args.sft_data,
                args.sft_data_probs,
                strategy,
                args.seed,
                max_count=1000,  # Use 1000 samples for Fisher computation
                return_eval=False,
                train_split=args.train_split,
            )
            
            sft_dataset_fisher = SFTDataset(
                sft_data_fisher,
                tokenizer,
                args.max_len,
                strategy,
                pretrain_mode=args.pretrain_mode,
            )
            
            fisher_dataloader = strategy.setup_dataloader(
                sft_dataset_fisher,
                args.micro_train_batch_size,
                False,
                False,
                sft_dataset_fisher.collate_fn,
            )
            
            fisher_dict = compute_fisher_information(
                model.model,
                fisher_dataloader,
                torch.cuda.current_device(),
                num_samples=1000,
            )
    
    # For KL regularization: load reference model
    reference_model = None
    if args.baseline_method == "kl":
        strategy.print("Loading reference model for KL regularization...")
        reference_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        reference_model.eval()
    
    # Configure loss function based on baseline method
    if args.baseline_method == "vanilla":
        loss_fn = ReplayLoss()
        strategy.print("Using Vanilla SFT (no continual learning)")
    elif args.baseline_method == "replay":
        loss_fn = ReplayLoss()
        strategy.print(f"Using Replay with mixing ratio: {args.replay_ratio}")
    elif args.baseline_method == "ewc":
        loss_fn = EWCLoss(ewc_lambda=args.ewc_lambda)
        strategy.print(f"Using EWC with lambda={args.ewc_lambda}")
    elif args.baseline_method == "l2":
        loss_fn = L2RegularizationLoss(l2_lambda=args.l2_lambda)
        strategy.print(f"Using L2 Regularization with lambda={args.l2_lambda}")
    elif args.baseline_method == "kl":
        loss_fn = KLRegularizationLoss(kl_lambda=args.kl_lambda)
        strategy.print(f"Using KL Regularization with lambda={args.kl_lambda}")
    else:
        raise ValueError(f"Unknown baseline method: {args.baseline_method}")
    
    # Configure scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    
    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )
    
    # Prepare model/optimizer with strategy
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler), is_rlhf=False)
    
    if reference_model is not None:
        reference_model = strategy.prepare(reference_model, is_rlhf=False)
    
    # Training loop
    strategy.print("Starting training...")
    global_step = 0
    
    for epoch in range(args.max_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Unpack batch
            _, inputs, attention_mask, info = batch
            inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
            attention_mask = attention_mask.to(torch.cuda.current_device()).squeeze(1)
            
            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask, return_output=True)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            
            # Compute loss based on method
            if args.baseline_method == "vanilla" or args.baseline_method == "replay":
                loss = loss_fn(logits, inputs)
                
            elif args.baseline_method == "ewc":
                loss = loss_fn(
                    logits, 
                    inputs,
                    model=model.model,
                    fisher_dict=fisher_dict,
                    old_params_dict=old_params_dict,
                )
                
            elif args.baseline_method == "l2":
                loss = loss_fn(
                    logits,
                    inputs,
                    model=model.model,
                    old_params_dict=old_params_dict,
                )
                
            elif args.baseline_method == "kl":
                # Get reference model outputs
                with torch.no_grad():
                    ref_outputs = reference_model(inputs, attention_mask=attention_mask, return_output=True)
                    ref_logits = ref_outputs["logits"] if isinstance(ref_outputs, dict) else ref_outputs.logits
                
                loss = loss_fn(
                    logits,
                    inputs,
                    old_logits=ref_logits,
                    attention_mask=attention_mask,
                )
            
            # Backward pass
            strategy.backward(loss, model, optim)
            strategy.optimizer_step(optim, model, scheduler)
            
            # Logging
            if global_step % args.logging_steps == 0:
                strategy.print(
                    f"Epoch {epoch}/{args.max_epochs}, "
                    f"Step {global_step}/{max_steps}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
            
            global_step += 1
            
            # Save checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                tag = f"epoch_{epoch}_step_{global_step}"
                save_path = os.path.join(args.save_path, tag)
                strategy.save_model(model, tokenizer, save_path)
    
    # Save final model
    strategy.save_model(model, tokenizer, args.save_path)
    strategy.print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument("--pretrain", type=str, required=True, help="HF model name or path")
    parser.add_argument("--save_path", type=str, default="./ckpt_cl_baseline")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    
    # Baseline method
    parser.add_argument(
        "--baseline_method", 
        type=str, 
        required=True,
        choices=["vanilla", "replay", "ewc", "l2", "kl"],
        help="Continual learning baseline method"
    )
    
    # Hyperparameters for different methods
    parser.add_argument("--ewc_lambda", type=float, default=1000.0, help="EWC regularization strength")
    parser.add_argument("--l2_lambda", type=float, default=0.01, help="L2 regularization strength")
    parser.add_argument("--kl_lambda", type=float, default=0.1, help="KL regularization strength")
    parser.add_argument("--replay_ratio", type=float, default=0.5, help="Ratio of old data in replay")
    
    # Data
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset name or path (new task)")
    parser.add_argument("--sft_data", type=str, default=None, help="SFT data path (old task, for EWC/Replay)")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="Sampling probs for datasets")
    parser.add_argument("--sft_data_probs", type=str, default="1.0", help="Sampling probs for SFT data")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--pretrain_mode", action="store_true", default=False)
    
    # Training
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--max_norm", type=float, default=1.0)
    
    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    
    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    
    args = parser.parse_args()
    train(args)

