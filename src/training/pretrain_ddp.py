# src/training/pretrain_ddp.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from datasets import load_from_disk
from transformers import AutoTokenizer, get_scheduler # Using HF scheduler for convenience
from tokenizers import Tokenizer # For custom tokenizer
import os
import mlflow
import time

from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logger
from src.utils.ddp_utils import setup_ddp, cleanup_ddp, is_main_process, ddp_barrier
from src.model.transformer import DummyTransformer # Replace with actual model

# --- Config & Setup ---
config = load_config()
model_cfg = config['model']
pretrain_cfg = config['pretrain']
tok_cfg = config['tokenizer']

logger = setup_logger("PretrainDDP")

def train():
    # --- DDP Setup ---
    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    if is_main_process():
        logger.info("Starting Pre-training with DDP...")
        logger.info(f"World Size: {world_size}")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(config['mlflow_config']['experiment_name']) # Add mlflow_config section
        mlflow.start_run(run_name="LLM_Pretraining_DDP")
        mlflow.log_params(model_cfg)
        mlflow.log_params(pretrain_cfg)

    # --- Load Tokenizer ---
    tokenizer_path = os.getenv("TOKENIZER_PATH")
    if is_main_process(): logger.info(f"Loading tokenizer from {tokenizer_path}")
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # If saved via HF
    tokenizer = Tokenizer.from_file(tokenizer_path) # If custom
    # Ensure PAD token is set if tokenizer doesn't have one automatically
    if tokenizer.token_to_id("[PAD]") is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model_cfg['vocab_size'] = tokenizer.get_vocab_size() # Update config vocab size

    # --- Load Data ---
    if is_main_process(): logger.info(f"Loading processed dataset from {pretrain_cfg['data_path']}")
    processed_dataset = load_from_disk(pretrain_cfg['data_path'])
    # Assuming dataset has 'input_ids' and 'labels' after preprocessing
    train_dataset = processed_dataset # Add split logic if needed ['train']

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=pretrain_cfg['seed'])
    dataloader = DataLoader(
        train_dataset,
        batch_size=pretrain_cfg['per_device_train_batch_size'],
        sampler=sampler,
        num_workers=4, # Adjust based on system
        pin_memory=True
    )

    # --- Init Model, Optimizer, Scheduler, Scaler ---
    if is_main_process(): logger.info("Initializing model...")
    model = DummyTransformer(model_cfg).to(device) # Replace with actual model
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=pretrain_cfg['learning_rate'], weight_decay=pretrain_cfg['weight_decay'])

    num_training_steps = pretrain_cfg['num_train_epochs'] * len(dataloader) // pretrain_cfg['gradient_accumulation_steps']
    lr_scheduler = get_scheduler(
        name=pretrain_cfg['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=pretrain_cfg['warmup_steps'],
        num_training_steps=num_training_steps
    )
    scaler = GradScaler(enabled=pretrain_cfg['use_amp'])

    # --- Training Loop ---
    if is_main_process(): logger.info("Starting training loop...")
    global_step = 0
    model.train()
    for epoch in range(pretrain_cfg['num_train_epochs']):
        sampler.set_epoch(epoch) # Important for shuffling with DDP
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device) # Assumes labels are present

            with autocast(enabled=pretrain_cfg['use_amp']):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / pretrain_cfg['gradient_accumulation_steps'] # Scale loss

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * pretrain_cfg['gradient_accumulation_steps'] # Unscale for logging
            num_batches += 1

            if (step + 1) % pretrain_cfg['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging (Rank 0 Only)
                if is_main_process() and global_step % pretrain_cfg['logging_steps'] == 0:
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                    logger.info(f"Epoch: {epoch+1}, Step: {global_step}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                    mlflow.log_metric("train_loss", avg_loss, step=global_step)
                    mlflow.log_metric("learning_rate", lr_scheduler.get_last_lr()[0], step=global_step)

                # Checkpointing (Rank 0 Only)
                if is_main_process() and global_step % pretrain_cfg['save_steps'] == 0:
                    save_path = os.path.join(pretrain_cfg['output_dir'], f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    # Save model state dict (unwrap DDP model)
                    torch.save(model.module.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                    # Save tokenizer and config if needed for HF compatibility
                    # tokenizer.save_pretrained(save_path)
                    # model.module.config.save_pretrained(save_path) # If model has HF config
                    logger.info(f"Checkpoint saved to {save_path}")
                    # Log checkpoint path to MLflow maybe? or just rely on output_dir

        # End of Epoch
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if is_main_process():
            logger.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Duration: {epoch_duration:.2f}s")
            mlflow.log_metric("epoch_avg_loss", avg_epoch_loss, step=epoch+1)

    # --- Final Save ---
    if is_main_process():
        final_save_path = os.path.join(pretrain_cfg['output_dir'], "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(final_save_path, "pytorch_model.bin"))
        logger.info(f"Final model saved to {final_save_path}")
        mlflow.log_artifact(final_save_path) # Log final model artifact
        mlflow.end_run()

    # --- Cleanup ---
    cleanup_ddp()

if __name__ == "__main__":
    # Basic checks for environment variables
    if not os.getenv("TOKENIZER_PATH") or not config['pretrain']['data_path']:
         print("Error: TOKENIZER_PATH or pretrain.data_path not configured correctly.")
    else:
        train()