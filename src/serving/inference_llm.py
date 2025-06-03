# src/serving/inference_llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tokenizers import Tokenizer
from peft import PeftModel, PeftConfig # To load LoRA adapters
import os
import time # Added import, was missing from previous read_files but present in file
from src.utils.config_loader import load_config
from src.utils.logging_utils import setup_logger
from src.model.transformer import CustomTransformerLM # Added import

logger = setup_logger("LLMInferenceService")
config = load_config()
model_cfg = config['model']
serving_cfg = config['serving']

class LLMGenerator:
    def __init__(self, base_model_path, adapter_path=None, tokenizer_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer_path = tokenizer_path or os.getenv("TOKENIZER_PATH")
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        logger.info(f"Loading tokenizer from: {self.tokenizer_path}")
        try:
            # tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            tokenizer = Tokenizer.from_file(self.tokenizer_path)
            if tokenizer.token_to_id("[PAD]") is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Tokenizer loaded successfully.")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _load_model(self):
        logger.info(f"Loading base model from: {self.base_model_path}")
        try:
            # Option A: Load custom model state dict
            # Need to instantiate the model class first
            # from src.model.transformer import DummyTransformer # Replace with actual
            # base_model = DummyTransformer(model_cfg) # Ensure config matches saved model

            # CRITICAL: The `model_cfg` used to instantiate `CustomTransformerLM` must EXACTLY match
            # the configuration of the model saved at `base_model_path`. This includes parameters like
            # `vocab_size`, `embed_dim`, `n_layers`, `n_heads`, etc.
            #
            # Best Practice: Load `model_config.yaml` from the `base_model_path` directory
            # (if saved there during training, e.g., by `pretrain_ddp.py`) and use that specific config.
            # `global_model_cfg = load_config()['model']` might not be the correct one if multiple model
            # versions or configurations exist.
            #
            # The `tokenizer` loaded by `_load_tokenizer` might also have its vocab_size adjusted
            # (e.g., by adding special tokens). If `model_cfg['vocab_size']` doesn't reflect the
            # tokenizer's actual vocab size that the model was trained with, it will cause a mismatch
            # with the embedding layer's dimensions when loading the `state_dict`.
            #
            # For this implementation, we are using the globally loaded `model_cfg`.
            # Ensure this global `model_cfg` is appropriate for the specific checkpoint being loaded.
            logger.info(f"Instantiating CustomTransformerLM with model_cfg (vocab_size: {model_cfg.get('vocab_size')}). Ensure this matches the checkpoint.")
            base_model = CustomTransformerLM(model_cfg) # Instantiate our custom model structure

            state_dict_path = os.path.join(self.base_model_path, "pytorch_model.bin") # Path to the model's weights
            logger.info(f"Loading base model state_dict from: {state_dict_path}")
            base_model.load_state_dict(torch.load(state_dict_path, map_location="cpu")) # Load weights to CPU first
            logger.info("Successfully loaded base model state_dict into CustomTransformerLM.")

            # Option B: Load HF model
            # base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path)

            if self.adapter_path and os.path.exists(self.adapter_path):
                logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
                # Ensure PeftConfig can find the base model type if needed
                model = PeftModel.from_pretrained(base_model, self.adapter_path)
                logger.info("LoRA adapter loaded successfully.")
            else:
                logger.info("No LoRA adapter path provided or found. Using base model.")
                model = base_model

            model.to(self.device)
            model.eval() # Set to evaluation mode
            logger.info("Model loaded and moved to device.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
        logger.info(f"Generating text for prompt (first 50 chars): {prompt[:50]}...")
        start_time = time.time()

        # --- Tokenize Prompt ---
        # Use encode method for custom tokenizer
        inputs = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([inputs.ids]).to(self.device) # Add batch dimension

        # --- Setup Generation Config ---
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True, # Use sampling
            pad_token_id=self.tokenizer.token_to_id("[PAD]"), # Use tokenizer's pad token ID
            eos_token_id=self.tokenizer.token_to_id("[EOS]") # Use tokenizer's EOS token ID
        )

        # --- Generate ---
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )

        # --- Decode ---
        # Use decode method for custom tokenizer
        # outputs[0] contains the full sequence (prompt + generation)
        generated_ids = outputs[0][input_ids.shape[1]:] # Get only generated token IDs
        result = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)

        duration = time.time() - start_time
        logger.info(f"Generation complete in {duration:.2f}s. Output length: {len(result)}")
        return result

# --- Global Instance ---
llm_generator_instance = None

def get_llm_generator():
    global llm_generator_instance
    if llm_generator_instance is None:
        logger.info("Initializing LLMGenerator instance...")
        llm_generator_instance = LLMGenerator(
            base_model_path=os.getenv("SERVING_MODEL_PATH", config['finetune_lora']['base_model_path']), # Default to pretrain output
            adapter_path=os.getenv("SERVING_ADAPTER_PATH", config['finetune_lora']['output_dir']), # Default to finetune output
            tokenizer_path=os.getenv("TOKENIZER_PATH")
        )
        logger.info("LLMGenerator instance initialized.")
    return llm_generator_instance