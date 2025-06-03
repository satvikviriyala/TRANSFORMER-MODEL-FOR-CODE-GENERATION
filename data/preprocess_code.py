# data/preprocess_code.py
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer # Can use HF tokenizer if trained/saved there
from tokenizers import Tokenizer # Or use custom trained tokenizer directly
import glob
import os
from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("DataPreprocessor")
config = load_config()
tok_cfg = config['tokenizer']
model_cfg = config['model']
pretrain_cfg = config['pretrain']

tokenizer_path = os.getenv("TOKENIZER_PATH", "trained_tokenizer/tokenizer.json")
raw_data_dir = os.getenv("RAW_DATA_DIR", "data/raw_corpus/")
processed_data_dir = pretrain_cfg['data_path'] # Get from config
seq_len = model_cfg['seq_len']

def load_and_tokenize_data():
    """Loads raw data, tokenizes, chunks, and saves as HF dataset."""
    logger.info("Starting data preprocessing...")

    # Load the trained tokenizer
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer file not found at {tokenizer_path}. Run tokenizer training first.")
        return
    # Load custom tokenizer
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.info(f"Successfully loaded tokenizer from {tokenizer_path}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}", exc_info=True)
        return
    # Or load from HF Hub if applicable
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # --- Load raw data ---
    # Example: Load python files using HF datasets text loader
    # Adjust glob pattern as needed
    try:
        # Using 'text' loader for simplicity, might need custom loader for specific structures
        raw_dataset = load_dataset('text', data_files=glob.glob(f"{raw_data_dir}/**/*.py", recursive=True), sample_by="document")['train']
        logger.info(f"Loaded raw dataset: {raw_dataset}")
    except Exception as e:
        logger.error(f"Failed to load raw data from {raw_data_dir}: {e}")
        return

    # --- Tokenize ---
    def tokenize_function(examples):
        # Tokenize text; assumes input field is 'text'
        # The tokenizer should handle padding/truncation if configured,
        # but we'll handle chunking separately for Causal LM.
        output = tokenizer.encode_batch(examples['text'])
        # Return list of token IDs for each example
        return {"input_ids": [enc.ids for enc in output]}

    logger.info("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(), # Use multiple processes
        remove_columns=raw_dataset.column_names # Remove original text column
    )
    logger.info(f"Tokenized dataset: {tokenized_dataset}")

    # --- Chunking for Causal LM ---
    # Concatenate all texts and then split into chunks of seq_len
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the small remainder to ensure chunks are full size
        total_length = (total_length // seq_len) * seq_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy() # For Causal LM, labels are shifted input_ids
        return result

    logger.info(f"Chunking dataset into sequences of length {seq_len}...")
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
    )
    logger.info(f"Final processed dataset: {lm_dataset}")

    # --- Save Processed Dataset ---
    logger.info(f"Saving processed dataset to {processed_data_dir}")
    os.makedirs(processed_data_dir, exist_ok=True)
    lm_dataset.save_to_disk(processed_data_dir)
    logger.info("Data preprocessing complete.")

if __name__ == "__main__":
    # Ensure necessary env vars are set
    if not os.getenv("RAW_DATA_DIR") or not os.getenv("TOKENIZER_PATH"):
         logger.error("RAW_DATA_DIR or TOKENIZER_PATH environment variables not set.")
    else:
        load_and_tokenize_data()