# src/tokenizer/train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE # Or WordPiece
from tokenizers.trainers import BpeTrainer # Or WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace # Or ByteLevel
from tokenizers.normalizers import Lowercase, NFD, StripAccents # Example normalizers
from tokenizers import normalizers
import glob
import os
from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logger

logger = setup_logger("TokenizerTrainer")
config = load_config()
tok_cfg = config['tokenizer']
output_path = os.getenv("TOKENIZER_PATH", "trained_tokenizer/tokenizer.json")

def train_tokenizer():
    """Trains a custom BPE tokenizer on the specified corpus."""
    logger.info("Starting tokenizer training...")

    # Initialize a tokenizer model (e.g., BPE)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Customize normalization and pre-tokenization
    # Example: lowercase, unicode normalization, strip accents
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace() # Or ByteLevel() for robustness

    # Setup trainer
    trainer = BpeTrainer(
        vocab_size=tok_cfg['vocab_size'],
        min_frequency=tok_cfg['min_frequency'],
        special_tokens=tok_cfg['special_tokens']
    )

    # Find training files
    files_pattern = tok_cfg['training_files_glob']
    files = glob.glob(files_pattern, recursive=True)
    if not files:
        logger.error(f"No files found matching pattern: {files_pattern}")
        return
    logger.info(f"Found {len(files)} files for training.")

    # Train the tokenizer
    tokenizer.train(files, trainer)
    logger.info("Tokenizer training complete.")

    # Save the tokenizer
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(output_path)
    logger.info(f"Tokenizer saved to {output_path}")

if __name__ == "__main__":
    # Ensure RAW_DATA_DIR is set in .env or environment
    if not os.getenv("RAW_DATA_DIR"):
        logger.error("RAW_DATA_DIR environment variable not set. Cannot find training files.")
    else:
        train_tokenizer()