# src/training/finetune_lora.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset # Or load custom data
from transformers import AutoTokenizer, get_scheduler, AutoModelForCausalLM # Can load base via HF if saved that way
from tokenizers import Tokenizer
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
import os
import mlflow
import time

from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logger
# Note: Fine-tuning often done on single node, multi-GPU but not necessarily DDP unless dataset is huge
# For simplicity, this example assumes single GPU or uses accelerate/Trainer later.
# To adapt for DDP: Add DDP setup/cleanup, DistributedSampler, wrap model with DDP.

# --- Config & Setup ---
config = load_config()
model_cfg = config['model'] # Base model config
ft_cfg = config['finetune_lora']
lora_cfg = ft_cfg['lora']
tok_cfg = config['tokenizer']

logger = setup_logger("FinetuneLoRA")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def finetune():
    logger.info("Starting LoRA Fine-tuning...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(config['mlflow_config']['experiment_name']) # Add mlflow_config section
    mlflow.start_run(run_name="LLM_Finetuning_LoRA")
    mlflow.log_params(ft_cfg) # Log fine-tuning specific params
    mlflow.log_params(lora_cfg) # Log LoRA params

    # --- Load Tokenizer ---
    tokenizer_path = os.getenv("TOKENIZER_PATH")
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.token_to_id("[PAD]") is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Resize model embeddings if needed after adding tokens (if loading base model state_dict directly)
    pad_token_id = tokenizer.token_to_id("[PAD]")

    # --- Load Base Model ---
    base_model_path = ft_cfg['base_model_path']
    logger.info(f"Loading base model from {base_model_path}")
    # Option A: Load custom model state dict
    from src.model.transformer import DummyTransformer # Replace with actual
    base_model = DummyTransformer(model_cfg) # Initialize with config used for pretraining
    state_dict_path = os.path.join(base_model_path, "pytorch_model.bin")
    base_model.load_state_dict(torch.load(state_dict_path, map_location="cpu")) # Load to CPU first
    logger.info("Loaded base model state_dict.")
    # Option B: Load using Hugging Face AutoModel if saved in that format
    # base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # --- Apply LoRA ---
    logger.info("Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # Important for Causal LM tasks
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['lora_dropout'],
        bias=lora_cfg['bias']
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    # --- Load Fine-tuning Data ---
    # Example: Loading a jsonl file { "code": "...", "summary": "..." }
    logger.info(f"Loading fine-tuning data from {ft_cfg['data_path']}")
    # Replace with your data loading logic
    try:
        # Using HF datasets to load jsonl easily
        ft_dataset = load_dataset('json', data_files=ft_cfg['data_path'])['train']
        # Add data splitting logic if needed
    except Exception as e:
        logger.error(f"Failed to load fine-tuning data: {e}")
        mlflow.end_run()
        return

    # --- Preprocess Fine-tuning Data ---
    def preprocess_finetune(examples):
        # Example for code summarization: format as "<code> [SEP] <summary>"
        # Adjust based on your task and how the model was pre-trained
        # Ensure consistent tokenization and padding/truncation
        inputs = [f"{code} [SEP] {summary}" for code, summary in zip(examples['code'], examples['summary'])] # Example structure
        # Tokenize combined input
        # Max length should consider prompt + expected output length
        model_inputs = tokenizer(inputs, max_length=model_cfg['seq_len'], padding="max_length", truncation=True, return_tensors="pt")

        # For Causal LM, labels are usually the input_ids shifted
        labels = model_inputs["input_ids"].clone()
        # Replace padding token id's with -100 so it's ignored by loss function
        labels[labels == pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    logger.info("Preprocessing fine-tuning dataset...")
    processed_ft_dataset = ft_dataset.map(
        preprocess_finetune,
        batched=True,
        remove_columns=ft_dataset.column_names
    )
    processed_ft_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(processed_ft_dataset, batch_size=ft_cfg['per_device_train_batch_size'], shuffle=True)

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=ft_cfg['learning_rate']) # Optimizer will only update LoRA params
    num_training_steps = ft_cfg['num_train_epochs'] * len(dataloader) // ft_cfg['gradient_accumulation_steps']
    lr_scheduler = get_scheduler(
        name=ft_cfg['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=ft_cfg['warmup_steps'],
        num_training_steps=num_training_steps
    )

    # --- Fine-tuning Loop ---
    logger.info("Starting fine-tuning loop...")
    global_step = 0
    model.train() # Set model to training mode
    for epoch in range(ft_cfg['num_train_epochs']):
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch) # Pass dict directly
            loss = outputs.loss / ft_cfg['gradient_accumulation_steps'] # Scale loss

            loss.backward()
            epoch_loss += loss.item() * ft_cfg['gradient_accumulation_steps']
            num_batches += 1

            if (step + 1) % ft_cfg['gradient_accumulation_steps'] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % ft_cfg['logging_steps'] == 0:
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                    logger.info(f"Epoch: {epoch+1}, Step: {global_step}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                    mlflow.log_metric("finetune_loss", avg_loss, step=global_step)

                # Save adapter weights periodically
                if global_step % ft_cfg['save_steps'] == 0:
                    save_path = os.path.join(ft_cfg['output_dir'], f"checkpoint-adapter-{global_step}")
                    model.save_pretrained(save_path) # Saves only adapter weights + config
                    logger.info(f"LoRA adapter checkpoint saved to {save_path}")

        # End of Epoch
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Duration: {epoch_duration:.2f}s")
        mlflow.log_metric("epoch_avg_finetune_loss", avg_epoch_loss, step=epoch+1)

        # --- (Optional) Evaluation Step ---
        # model.eval()
        # total_eval_loss = 0
        # with torch.no_grad():
        #     for batch in eval_dataloader: # Create an eval_dataloader
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         outputs = model(**batch)
        #         total_eval_loss += outputs.loss.item()
        # avg_eval_loss = total_eval_loss / len(eval_dataloader)
        # logger.info(f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.4f}")
        # mlflow.log_metric("eval_loss", avg_eval_loss, step=epoch+1)
        # model.train() # Set back to train mode


    # --- Final Save ---
    final_save_path = ft_cfg['output_dir']
    model.save_pretrained(final_save_path) # Saves adapter_model.bin, adapter_config.json etc.
    logger.info(f"Final LoRA adapter weights saved to {final_save_path}")
    mlflow.log_artifact(final_save_path, artifact_path="final_lora_adapter")
    mlflow.end_run()

if __name__ == "__main__":
     # Basic checks for environment variables
    if not os.getenv("TOKENIZER_PATH") or not ft_cfg['base_model_path'] or not ft_cfg['data_path']:
         logger.error("Error: TOKENIZER_PATH, finetune_lora.base_model_path, or finetune_lora.data_path not configured correctly.")
    else:
        finetune()