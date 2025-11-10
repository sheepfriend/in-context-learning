"""
Autoregressive Transformer for table connectivity path search.

Architecture:
- First 2 layers: Block diagonal attention (for schema processing)
- Remaining layers: Causal attention (for autoregressive generation)
- Positional encoding: Low-rank for first 2 layers, standard for remaining
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class AutoregressiveTransformerModel(nn.Module):
    """
    Autoregressive Transformer with hybrid attention pattern.
    
    - First 2 layers: Block diagonal for schema
    - Remaining layers: Causal for autoregressive generation
    """
    
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, 
                 V=5, C=3, vocab_size=None, schema_len=None):
        """
        Args:
            n_dims: Input dimension
            n_positions: Maximum sequence length
            n_embd: Embedding dimension
            n_layer: Number of layers
            n_head: Number of attention heads
            V: Number of tables
            C: Columns per table
            vocab_size: Output vocabulary size
            schema_len: Length of schema encoding (V*(C+1)+1)
        """
        super().__init__()
        
        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.V = V
        self.C = C
        
        # Schema length: V*(C+1) + 1 (separator)
        self.schema_len = schema_len if schema_len else V * (C + 1) + 1
        
        # Vocabulary size: [PAD, START, SEP, END] + columns
        self.vocab_size = vocab_size if vocab_size else 4 + V * C
        
        # GPT2 backbone
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        
        self.name = f"autoregressive_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}_V={V}_C={C}"
        
        # Input projection
        self._read_in = nn.Linear(n_dims, n_embd)
        
        # Backbone
        self._backbone = GPT2Model(configuration)
        
        # Output head for next token prediction
        self._output_head = nn.Linear(n_embd, self.vocab_size)
        
        # Custom positional embeddings
        # Low-rank for schema part
        self.pos_emb_schema = nn.Parameter(torch.randn(self.schema_len, n_embd) * 0.02)
        # Standard for path part
        self.pos_emb_path = nn.Parameter(torch.randn(n_positions - self.schema_len, n_embd) * 0.02)
        
        # Disable default positional embeddings
        self._backbone.wpe.weight.requires_grad = False
        
        # Register attention masks
        self._register_attention_masks()
    
    def _register_attention_masks(self):
        """Create attention masks for different layer groups."""
        seq_len = self.n_positions
        
        # Mask for first 2 layers: block diagonal for schema + causal for path
        mask_first_layers = torch.zeros(seq_len, seq_len)
        
        # Schema part: block diagonal
        for i in range(self.V):
            start = i * (self.C + 1)
            end = start + (self.C + 1)
            if end <= self.schema_len:
                mask_first_layers[start:end, start:end] = 1
        
        # Separator can see all schema
        if self.schema_len > 0:
            mask_first_layers[self.schema_len - 1, :self.schema_len] = 1
        
        # Path part: causal mask
        for i in range(self.schema_len, seq_len):
            mask_first_layers[i, self.schema_len:i + 1] = 1
        
        # Convert to attention mask format
        self.mask_first_layers = (1 - mask_first_layers) * -1e9
        
        # Mask for remaining layers: pure causal
        mask_remaining_layers = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            mask_remaining_layers[i, :i + 1] = 1
        
        self.mask_remaining_layers = (1 - mask_remaining_layers) * -1e9
        
        # Register as buffers
        self.register_buffer('mask_first_buffer', self.mask_first_layers)
        self.register_buffer('mask_remaining_buffer', self.mask_remaining_layers)
    
    def _get_positional_embeddings(self, seq_len):
        """Get positional embeddings for the sequence."""
        if seq_len <= self.schema_len:
            return self.pos_emb_schema[:seq_len]
        else:
            path_len = seq_len - self.schema_len
            return torch.cat([
                self.pos_emb_schema,
                self.pos_emb_path[:path_len]
            ], dim=0)
    
    def _apply_custom_attention(self, embeds):
        """Apply transformer blocks with custom attention masks."""
        hidden_states = embeds
        
        for i, block in enumerate(self._backbone.h):
            # Choose mask based on layer
            if i < 2:
                mask = self.mask_first_buffer[:embeds.shape[1], :embeds.shape[1]]
            else:
                mask = self.mask_remaining_buffer[:embeds.shape[1], :embeds.shape[1]]
            
            # Expand mask for batch and heads
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
            mask = mask.expand(embeds.shape[0], self.n_head, -1, -1)
            
            # Apply block
            outputs = block(
                hidden_states,
                attention_mask=mask,
            )
            hidden_states = outputs[0]
        
        # Final layer norm
        hidden_states = self._backbone.ln_f(hidden_states)
        
        return hidden_states
    
    def forward(self, xs, ys=None):
        """
        Forward pass.
        
        Args:
            xs: [batch_size, seq_len, n_dims] input sequences
            ys: [batch_size, path_len] target next tokens (optional, for training)
        
        Returns:
            If ys is None: [batch_size, seq_len, vocab_size] logits
            If ys is not None: [batch_size, path_len, vocab_size] logits for path part only
        """
        batch_size, seq_len, _ = xs.shape
        
        # Input projection
        embeds = self._read_in(xs)  # [batch, seq, embd]
        
        # Add positional embeddings
        pos_emb = self._get_positional_embeddings(seq_len)
        embeds = embeds + pos_emb.unsqueeze(0)
        
        # Apply transformer with custom attention
        hidden_states = self._apply_custom_attention(embeds)
        
        # Project to vocabulary
        logits = self._output_head(hidden_states)  # [batch, seq, vocab_size]
        
        # If training, return only path part logits
        if ys is not None:
            # Return logits for path positions (after schema)
            path_logits = logits[:, self.schema_len:self.schema_len + ys.shape[1], :]
            return path_logits
        
        return logits
    
    def generate(self, xs_schema, start_tokens, max_length=15, temperature=1.0):
        """
        Generate a path autoregressively.
        
        Args:
            xs_schema: [batch_size, schema_len, n_dims] schema encoding
            start_tokens: [batch_size, start_len, n_dims] initial path tokens
            max_length: Maximum path length
            temperature: Sampling temperature
        
        Returns:
            generated: [batch_size, max_length, n_dims] generated sequence
            token_ids: [batch_size, max_length] generated token IDs
        """
        batch_size = xs_schema.shape[0]
        device = xs_schema.device
        
        # Initialize with schema + start tokens
        xs = torch.cat([xs_schema, start_tokens], dim=1)
        token_ids = []
        
        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                logits = self.forward(xs)  # [batch, seq, vocab]
                
                # Get logits for last position
                next_logits = logits[:, -1, :] / temperature  # [batch, vocab]
                
                # Sample next token
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # [batch, 1]
                
                token_ids.append(next_token)
                
                # Check if END token
                if (next_token == 3).all():  # END_TOKEN = 3
                    break
                
                # Append to sequence (placeholder, need to encode)
                # For now, just use zero embedding
                next_embed = torch.zeros(batch_size, 1, self.n_dims, device=device)
                xs = torch.cat([xs, next_embed], dim=1)
        
        token_ids = torch.cat(token_ids, dim=1) if token_ids else torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        
        return xs, token_ids


def build_autoregressive_model(conf):
    """Build autoregressive model from config."""
    if conf.model.family == "autoregressive_gpt2":
        model = AutoregressiveTransformerModel(
            n_dims=conf.model.n_dims,
            n_positions=conf.model.n_positions,
            n_embd=conf.model.n_embd,
            n_layer=conf.model.n_layer,
            n_head=conf.model.n_head,
            V=conf.model.V,
            C=conf.model.C,
            vocab_size=conf.model.get('vocab_size', None),
            schema_len=conf.model.get('schema_len', None),
        )
        return model
    else:
        raise NotImplementedError(f"Model family {conf.model.family} not implemented")

