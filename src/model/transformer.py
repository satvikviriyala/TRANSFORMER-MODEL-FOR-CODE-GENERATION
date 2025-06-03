import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GenerationConfig # For generate method

# Helper Function: Scaled Dot Product Attention
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    """
    Calculates Scaled Dot-Product Attention.

    This function implements the core attention mechanism used in Transformer models.
    It computes the dot products of the query with all keys, divides each by sqrt(head_dim),
    applies a softmax function to obtain the weights on the values, and then outputs
    the weighted sum of the values. An optional mask can be applied to hide certain
    positions from attending to others (e.g., for padding or causal attention).
    An optional dropout can be applied to the attention weights.
    Args:
        q (Tensor): Query tensor; shape (batch_size, num_heads, seq_len_q, head_dim)
        k (Tensor): Key tensor; shape (batch_size, num_heads, seq_len_k, head_dim)
        v (Tensor): Value tensor; shape (batch_size, num_heads, seq_len_v, head_dim)
                    (seq_len_k and seq_len_v must be the same)
        mask (Tensor, optional): Boolean mask for attention.
                                  - For self-attention, this is (batch_size, 1, seq_len_q, seq_len_k)
                                  - For cross-attention, this is (batch_size, 1, seq_len_q, seq_len_k)
                                  Where mask values are True for positions to be masked (ignored).
                                  Defaults to None.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
    Returns:
        output (Tensor): Attention output; shape (batch_size, num_heads, seq_len_q, head_dim)
        attn_weights (Tensor): Attention weights; shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    head_dim = q.size(-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim) # (batch, num_heads, seq_len_q, seq_len_k)

    if mask is not None:
        # The mask should be broadcastable to the shape of attn_scores.
        # Typical attention masks are (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k) for causal.
        # PyTorch's masked_fill expects mask values of True to be filled.
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf')) # Mask value 0 means hide

    attn_weights = F.softmax(attn_scores, dim=-1)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    output = torch.matmul(attn_weights, v) # (batch_size, num_heads, seq_len_q, head_dim)
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention.

    This module performs Multi-Head Attention as described in "Attention Is All You Need".
    It projects the input Q, K, V tensors into multiple lower-dimensional spaces (heads),
    applies scaled dot-product attention independently in each head, concatenates the
    results, and finally projects them through a linear layer.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads. `embed_dim` must be divisible by `num_heads`.
        dropout (float, optional): Dropout probability for the attention weights. Defaults to 0.0.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, attention_mask=None):
        """
        Args:
            q (Tensor): Query tensor; shape (batch_size, seq_len_q, embed_dim)
            k (Tensor): Key tensor; shape (batch_size, seq_len_k, embed_dim)
            v (Tensor): Value tensor; shape (batch_size, seq_len_v, embed_dim)
            attention_mask (Tensor, optional): Mask to prevent attention to certain positions.
                                            Shape (batch_size, 1, seq_len_q, seq_len_k).
                                            Mask values of 0 indicate positions to be masked.
                                            Defaults to None.
        Returns:
            output (Tensor): Output tensor; shape (batch_size, seq_len_q, embed_dim)
            attn_weights (Tensor): Attention weights; shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)

        # Linear projections for Q, K, V. These will be split into multiple heads.
        # These are the layers often targeted by LoRA (e.g., 'q_proj', 'v_proj').
        q = self.q_proj(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_len_q, head_dim)
        k = self.k_proj(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_len_k, head_dim)
        v = self.v_proj(v).view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, seq_len_v, head_dim)

        # Apply scaled dot-product attention independently for each head.
        # The attention_mask is broadcasted or applied head-wise as needed by the scaled_dot_product_attention function.
        context, attn_weights = scaled_dot_product_attention(q, k, v, mask=attention_mask, dropout_p=self.dropout_p)

        # Concatenate attention outputs from all heads and apply final linear projection.
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim) # (batch_size, seq_len_q, embed_dim)
        output = self.out_proj(context) # Final linear layer, combines information from all heads
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """
    Implements a Position-wise Feed-Forward Network (FFN).

    This module consists of two linear transformations with a non-linear activation
    function in between, applied to each position independently and identically.
    It's a standard component in Transformer blocks.

    Args:
        embed_dim (int): Input and output dimension of the network.
        hidden_dim (int): Dimension of the inner linear layer.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        activation_fn (str, optional): Activation function to use ('relu' or 'gelu'). Defaults to "relu".
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, activation_fn="relu"):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x) # Second linear layer
        return x

class TransformerBlock(nn.Module):
    """
    Implements a single Transformer block (encoder or decoder layer).

    A Transformer block consists of a multi-head self-attention sublayer and a
    position-wise feed-forward sublayer. Layer normalization and residual connections
    are applied around each sublayer. Dropout is also applied.

    Args:
        embed_dim (int): Dimension of the input and output embeddings.
        num_heads (int): Number of attention heads for the multi-head attention sublayer.
        hidden_dim (int): Hidden dimension for the position-wise feed-forward sublayer.
        dropout (float, optional): Dropout probability for sublayer outputs. Defaults to 0.1.
        activation_fn (str, optional): Activation function for the FFN ('relu' or 'gelu'). Defaults to "relu".
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1, activation_fn="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, hidden_dim, dropout=dropout, activation_fn=activation_fn)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x (Tensor): Input tensor; shape (batch_size, seq_len, embed_dim)
            attention_mask (Tensor, optional): Mask for self-attention.
                                            Shape (batch_size, 1, seq_len, seq_len).
                                            Mask values of 0 indicate positions to be masked.
                                            Defaults to None.
        Returns:
            output (Tensor): Output tensor; shape (batch_size, seq_len, embed_dim)
        """
        # First sublayer: Multi-Head Self-Attention
        # Input `x` is used for Q, K, and V (self-attention).
        # `attention_mask` is passed to prevent attending to certain positions (e.g., padding or future tokens).
        attn_output, _ = self.self_attn(x, x, x, attention_mask=attention_mask)
        x = x + self.dropout(attn_output) # Add (residual connection) & Norm
        x = self.norm1(x)

        # Second sublayer: Position-wise Feed-Forward Network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output) # Add (residual connection) & Norm
        x = self.norm2(x)
        return x

class CustomTransformerLM(nn.Module):
    """
    A custom Transformer-based Language Model (LM).

    This model implements a decoder-only Transformer architecture suitable for language modeling.
    It includes token embeddings, positional embeddings, a stack of Transformer blocks,
    layer normalization, and a final linear layer (LM head) to produce logits over the vocabulary.

    Args:
        model_cfg (dict): A configuration dictionary containing model parameters:
            - vocab_size (int): Size of the vocabulary.
            - seq_len (int): Maximum sequence length the model can handle.
            - embed_dim (int): Dimension of token and positional embeddings.
            - n_layers (int): Number of Transformer blocks.
            - n_heads (int): Number of attention heads in each Transformer block.
            - hidden_dim (int, optional): Hidden dimension of the FFN in Transformer blocks.
                                       Defaults to `embed_dim * 4`.
            - dropout (float): Dropout probability used in embeddings and Transformer blocks.
            - activation_fn (str, optional): Activation for FFN ('relu' or 'gelu'). Defaults to "relu".
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.config = model_cfg # Store config for later use if needed
        self.vocab_size = model_cfg['vocab_size']
        self.seq_len = model_cfg['seq_len']
        self.embed_dim = model_cfg['embed_dim']
        self.n_layers = model_cfg['n_layers']
        self.n_heads = model_cfg['n_heads']
        self.hidden_dim = model_cfg.get('hidden_dim', self.embed_dim * 4) # Default if not provided
        self.dropout_p = model_cfg['dropout']
        self.activation_fn = model_cfg.get('activation_fn', "relu") # Default activation

        self.token_embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        # Using learnable positional embeddings
        self.position_embeddings = nn.Embedding(self.seq_len, self.embed_dim)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.embed_dim, self.n_heads, self.hidden_dim, self.dropout_p, self.activation_fn)
             for _ in range(self.n_layers)]
        )

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        # Optional: Tie weights
        # self.lm_head.weight = self.token_embeddings.weight

        self.dropout = nn.Dropout(self.dropout_p) # Dropout for embeddings

        # Note: The `target_modules` for LoRA (e.g., ['q_proj', 'v_proj']) are identified by
        # their names within the `MultiHeadAttention` class (e.g., `transformer_blocks.0.self_attn.q_proj`).
        # This model structure is compatible with such targeting if LoRA is applied externally.


    def _create_causal_mask(self, size, device):
        """
        Creates a causal mask for self-attention.

        Args:
            size (int): The sequence length.
            device (torch.device): The device to create the mask on.

        Returns:
            Tensor: A boolean causal mask of shape (1, 1, size, size).
                    Mask values of `True` indicate positions to keep (i.e. attend to),
                    while `False` indicates positions to mask (cannot attend to future tokens).
                    The `scaled_dot_product_attention` function expects mask values of 0 to be masked,
                    so this mask is used as `mask == 0` there.
        """
        # Creates a lower triangular matrix of ones.
        mask = torch.ones(size, size, device=device, dtype=torch.bool).tril(diagonal=0)
        # Reshape to (1, 1, size, size) for broadcasting with attention scores (batch, num_heads, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the CustomTransformerLM.

        Args:
            input_ids (Tensor): Input token IDs; shape (batch_size, seq_len).
            attention_mask (Tensor, optional): Padding mask; shape (batch_size, seq_len).
                                            Values of 1 indicate non-padded tokens, 0 for padded.
                                            Defaults to None (no padding).
            labels (Tensor, optional): Target token IDs for loss calculation; shape (batch_size, seq_len).
                                     Defaults to None (no loss calculated).

        Returns:
            dict: A dictionary containing:
                - "logits" (Tensor): Output logits over the vocabulary; shape (batch_size, seq_len, vocab_size).
                - "loss" (Tensor, optional): Cross-entropy loss if `labels` are provided. Otherwise None.
        """
        batch_size, seq_length = input_ids.size()
        assert seq_length <= self.seq_len, \
            f"Input sequence length ({seq_length}) exceeds model's maximum sequence length ({self.seq_len})"

        # 1. Get Token and Positional Embeddings
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0) # Shape: (1, seq_len)
        tok_emb = self.token_embeddings(input_ids)    # Shape: (batch_size, seq_len, embed_dim)
        pos_emb = self.position_embeddings(positions) # Shape: (1, seq_len, embed_dim) -> broadcasts to (batch_size, seq_len, embed_dim)
        x = self.dropout(tok_emb + pos_emb) # Apply dropout to the sum of embeddings

        # 2. Prepare Combined Attention Mask for self-attention in Transformer blocks
        # Causal mask: ensures that attention is only applied to the left part of the sequence.
        # Shape: (1, 1, seq_len, seq_len)
        causal_mask = self._create_causal_mask(seq_length, x.device)

        # Padding mask: prevents attention to padding tokens if `attention_mask` is provided.
        # `attention_mask` from input is (batch_size, seq_len), where 1 means token, 0 means pad.
        if attention_mask is not None:
            # Expand `attention_mask` to be broadcastable with `causal_mask` and MHA's expected format.
            # Shape: (batch_size, 1, 1, seq_len)
            # `scaled_dot_product_attention` expects mask values of 0 to indicate masked positions.
            # Our `attention_mask` has 0 for pad, so it's already in the correct format for masking.
            padding_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len)
            # Combine masks: if a position is masked by either causal or padding, it should be masked.
            # Since both masks use 0 for masked (after tril for causal), logical AND works.
            # For `scaled_dot_product_attention`, the `mask` argument is `attn_scores.masked_fill(mask == 0, float('-inf'))`.
            # So `combined_mask` should be `True` for positions to keep.
            # `causal_mask` is `True` for positions to keep.
            # `padding_mask_expanded` (if derived from 0/1 input) needs to be boolean.
            # If attention_mask has 1 for keep and 0 for mask:
            combined_mask = causal_mask & padding_mask_expanded.bool() # (batch_size, 1, seq_len, seq_len) after broadcasting
        else:
            combined_mask = causal_mask # Broadcasts to (batch_size, 1, seq_len, seq_len)

        # 3. Pass through Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=combined_mask) # Pass combined_mask to each block's self-attention

        # 4. Final Layer Normalization and LM Head
        x = self.ln_f(x) # Final layer normalization before the LM head
        logits = self.lm_head(x) # Project to vocabulary size: (batch_size, seq_len, vocab_size)

        # 5. Calculate Loss (if labels are provided)
        loss = None
        if labels is not None:
            # For Causal LM, shift logits and labels for next token prediction.
            # Logits for token at position `i` are used to predict token at position `i+1`.
            shift_logits = logits[..., :-1, :].contiguous() # Shape: (batch_size, seq_len-1, vocab_size)
            shift_labels = labels[..., 1:].contiguous()     # Shape: (batch_size, seq_len-1)
            # Flatten tokens and calculate cross-entropy loss.
            # `CrossEntropyLoss` expects logits of shape (N, C) and labels of shape (N).
            loss_fct = nn.CrossEntropyLoss() # Standard CE loss, ignores index -100 by default (useful for padding)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"logits": logits, "loss": loss}

    @torch.no_grad() # Disable gradient calculations for generation
    def generate(self, input_ids, generation_config=None, max_new_tokens=None):
        """
        Generates token sequences based on input_ids and a generation configuration.

        This method implements a basic auto-regressive generation loop.
        It supports greedy decoding and sampling (with temperature and top-k).

        Args:
            input_ids (Tensor): Context tokens; shape (batch_size, current_seq_len).
            generation_config (transformers.GenerationConfig, optional):
                Configuration object from Hugging Face `transformers` library.
                If None, a default GenerationConfig is used.
                Key parameters used: `max_new_tokens`, `pad_token_id`, `eos_token_id`,
                `do_sample`, `temperature`, `top_k`.
            max_new_tokens (int, optional): Overrides `max_new_tokens` in `generation_config`
                                         or sets a default if `generation_config` is also None.
        Returns:
            output_ids (Tensor): Generated token sequences, including input_ids;
                                 shape (batch_size, original_seq_len + generated_len).
        """
        # Ensure generation_config is available
        if generation_config is None:
            generation_config = GenerationConfig() # Use default HF config
            # Set max_new_tokens if not in default config and provided directly via argument
            if max_new_tokens is not None:
                 generation_config.max_new_tokens = max_new_tokens
            # Fallback if still not set in GenerationConfig instance
            elif not hasattr(generation_config, 'max_new_tokens') or generation_config.max_new_tokens is None:
                 generation_config.max_new_tokens = 20 # Default number of new tokens

        # Override generation_config.max_new_tokens if max_new_tokens is explicitly passed as an argument
        if max_new_tokens is not None:
            generation_config.max_new_tokens = max_new_tokens

        current_ids = input_ids # Start with the provided input_ids
        batch_size = input_ids.size(0)

        # Retrieve generation parameters from the configuration
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id # This can be an int or a list of ints

        # Standardize eos_token_id to a list for easier checking later
        _eos_token_id_list = []
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                _eos_token_id_list = [eos_token_id]
            elif isinstance(eos_token_id, list):
                _eos_token_id_list = eos_token_id

        # Keep track of which sequences in the batch have already reached an EOS token
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Main auto-regressive generation loop
        for _ in range(generation_config.max_new_tokens):
            # Prepare model inputs for the next token prediction
            # Crop the context if its length exceeds `self.seq_len - 1` (to make space for the new token)
            if current_ids.size(1) >= self.seq_len:
                # The context for the next token prediction must be less than self.seq_len
                model_input_ids = current_ids[:, -(self.seq_len - 1):]
            else:
                model_input_ids = current_ids

            # Forward pass to get logits for the next token.
            # `attention_mask` is None here assuming `model_input_ids` are dense (not padded within this context window).
            # The model's internal causal mask handles masking future tokens.
            outputs = self.forward(model_input_ids, attention_mask=None, labels=None)
            next_token_logits = outputs["logits"][:, -1, :]  # Get logits for the very last token position: (batch_size, vocab_size)

            # Apply temperature scaling to logits (modifies the distribution to be sharper or flatter)
            if hasattr(generation_config, 'temperature') and generation_config.temperature > 0 and generation_config.temperature != 1.0:
                next_token_logits = next_token_logits / generation_config.temperature

            # Apply top-k filtering (constrains sampling to the k most likely next tokens)
            if hasattr(generation_config, 'top_k') and generation_config.top_k is not None and generation_config.top_k > 0:
                v, _ = torch.topk(next_token_logits, min(generation_config.top_k, next_token_logits.size(-1)))
                # Set logits for tokens not in the top-k to -infinity to effectively exclude them from softmax
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            # Calculate probabilities and sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            if hasattr(generation_config, 'do_sample') and generation_config.do_sample:
                next_token_id = torch.multinomial(probs, num_samples=1) # Sample based on probability: (batch_size, 1)
            else: # Greedy decoding: pick the token with the highest probability
                next_token_id = torch.argmax(probs, dim=-1, keepdim=True) # (batch_size, 1)

            # If a sequence in the batch has already reached an EOS token,
            # continue to fill its subsequent generated tokens with `pad_token_id`.
            if pad_token_id is not None: # Only if pad_token_id is defined
                 next_token_id[eos_reached] = pad_token_id

            # Check if any new sequences in the batch reached an EOS token with the `next_token_id` just generated.
            # Update `eos_reached` status for those sequences.
            if _eos_token_id_list: # Only perform check if EOS token(s) are defined
                for e_id in _eos_token_id_list: # Iterate if multiple EOS tokens
                    eos_reached = eos_reached | (next_token_id.squeeze(-1) == e_id) # .squeeze() to match shape

            # Append the newly generated token to the current sequences in the batch
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)

            # Stop generation early if all sequences in the batch have reached an EOS token
            if eos_reached.all():
                break

        return current_ids

# Example of how model_cfg might look (usually loaded from a YAML or JSON)
        Args:
# (This part is fine, no changes needed for the example test code section)

# Example of how model_cfg might look (usually loaded from a YAML or JSON)
# model_cfg_example = {
#     'vocab_size': 50257,  # Example: GPT-2 vocab size
#     'seq_len': 1024,     # Max sequence length
#     'embed_dim': 768,    # Embedding dimension
#     'n_layers': 12,      # Number of Transformer blocks
#     'n_heads': 12,       # Number of attention heads
#     'hidden_dim': 768 * 4, # Hidden dimension in FFN (often 4x embed_dim)
#     'dropout': 0.1,
#     'activation_fn': 'gelu', # 'relu' or 'gelu'
#     # 'target_modules': ['q_proj', 'v_proj'] # For LoRA
# }

# To test the model (basic check):
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     config_test = {
#         'vocab_size': 1000,
#         'seq_len': 64,
#         'embed_dim': 128,
#         'n_layers': 2,
#         'n_heads': 4,
#         'hidden_dim': 128 * 4,
#         'dropout': 0.1,
#         'activation_fn': 'relu'
#     }
#     model = CustomTransformerLM(config_test).to(device)
#     print(f"Model instantiated with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

#     # Test forward pass
#     print("\nTesting forward pass...")
#     dummy_input_ids = torch.randint(0, config_test['vocab_size'], (2, 10), device=device) # (batch_size=2, seq_len=10)
#     dummy_labels = torch.randint(0, config_test['vocab_size'], (2, 10), device=device)
#     attention_m = torch.ones_like(dummy_input_ids)
#     if dummy_input_ids.shape[0] > 1:
#       attention_m[1, 7:] = 0

#     output = model(dummy_input_ids, attention_mask=attention_m, labels=dummy_labels)
#     print("Logits shape:", output['logits'].shape)
#     print("Loss:", output['loss'])
#     assert output['logits'].shape == (dummy_input_ids.shape[0], dummy_input_ids.shape[1], config_test['vocab_size'])
#     assert output['loss'] is not None

#     # Test generation
#     print("\nTesting generation...")
#     gen_config = GenerationConfig(
#         max_new_tokens=10,
#         pad_token_id=0,
#         eos_token_id=50,
#         do_sample=True,
#         temperature=0.7,
#         top_k=5
#     )
#     start_ids = torch.tensor([[1, 2, 3]], device=device)
#     generated_ids = model.generate(start_ids, generation_config=gen_config)
#     print("Generated IDs shape:", generated_ids.shape)
#     print("Generated IDs:", generated_ids)
#     assert generated_ids.shape[1] <= start_ids.shape[1] + gen_config.max_new_tokens

#     print("\nTesting generation with different max_new_tokens...")
#     generated_ids_short = model.generate(start_ids, max_new_tokens=5)
#     print("Generated IDs (short) shape:", generated_ids_short.shape)
#     assert generated_ids_short.shape[1] <= start_ids.shape[1] + 5

#     print("\nTest with batch_size > 1 for generation")
#     start_ids_batch = torch.tensor([[1,2,3], [4,5,6]], device=device)
#     generated_ids_batch = model.generate(start_ids_batch, max_new_tokens=5)
#     print("Generated IDs (batch) shape:", generated_ids_batch.shape)
#     assert generated_ids_batch.shape[0] == start_ids_batch.shape[0]
#     assert generated_ids_batch.shape[1] <= start_ids_batch.shape[1] + 5
#     print("Generated IDs (batch):", generated_ids_batch)

#     print("\nAll basic tests passed!")
