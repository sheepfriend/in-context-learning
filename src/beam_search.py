"""
Beam search implementation for autoregressive path generation.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


class BeamSearcher:
    def __init__(self, model, beam_width=5, max_length=15, length_penalty=1.0, 
                 end_token=3, pad_token=0):
        """
        Beam search for autoregressive generation.
        
        Args:
            model: Autoregressive model
            beam_width: Number of beams to keep
            max_length: Maximum generation length
            length_penalty: Length normalization factor (alpha in Google NMT paper)
            end_token: ID of END token
            pad_token: ID of PAD token
        """
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.end_token = end_token
        self.pad_token = pad_token
    
    def length_normalize(self, log_probs, length):
        """
        Apply length normalization.
        
        From Google NMT paper: lp(Y) = ((5 + |Y|) / (5 + 1))^alpha
        """
        if self.length_penalty == 0:
            return log_probs
        return log_probs / (((5.0 + length) / 6.0) ** self.length_penalty)
    
    def search(self, xs_schema, column_embeddings, device='cuda'):
        """
        Perform beam search.
        
        Args:
            xs_schema: [1, schema_len, n_dims] schema encoding (single example)
            column_embeddings: [vocab_size, n_dims-1] column embeddings for decoding
            device: Device to run on
        
        Returns:
            best_sequences: List of (sequence, score) tuples
        """
        batch_size = 1  # Beam search on single example
        
        # Initialize beams
        # Each beam: (sequence_embeddings, token_ids, log_prob, finished)
        beams = [(
            xs_schema,  # Start with just schema
            [],  # No tokens yet
            0.0,  # Log probability
            False  # Not finished
        )]
        
        finished_beams = []
        
        for step in range(self.max_length):
            candidates = []
            
            for seq_embed, tokens, log_prob, finished in beams:
                if finished:
                    candidates.append((seq_embed, tokens, log_prob, True))
                    continue
                
                # Get model predictions
                with torch.no_grad():
                    logits = self.model.forward(seq_embed)  # [1, seq_len, vocab_size]
                    next_logits = logits[0, -1, :]  # [vocab_size]
                    log_probs = F.log_softmax(next_logits, dim=-1)  # [vocab_size]
                
                # Get top-k candidates
                top_log_probs, top_indices = torch.topk(log_probs, self.beam_width)
                
                for i in range(self.beam_width):
                    token_id = top_indices[i].item()
                    token_log_prob = top_log_probs[i].item()
                    
                    # New sequence
                    new_tokens = tokens + [token_id]
                    new_log_prob = log_prob + token_log_prob
                    
                    # Check if finished
                    is_finished = (token_id == self.end_token)
                    
                    if is_finished:
                        # Add to finished beams with length normalization
                        normalized_score = self.length_normalize(new_log_prob, len(new_tokens))
                        finished_beams.append((new_tokens, normalized_score))
                    else:
                        # Extend sequence embedding
                        # Encode the new token
                        if token_id >= 4:  # Column token
                            col_id = token_id - 4
                            if col_id < column_embeddings.shape[0]:
                                token_embed = torch.cat([
                                    column_embeddings[col_id],
                                    torch.tensor([col_id], device=device, dtype=torch.float)
                                ], dim=0)
                            else:
                                token_embed = torch.zeros(xs_schema.shape[2], device=device)
                        else:
                            # Special token - use zero embedding
                            token_embed = torch.zeros(xs_schema.shape[2], device=device)
                        
                        # Append to sequence
                        new_seq_embed = torch.cat([
                            seq_embed,
                            token_embed.unsqueeze(0).unsqueeze(0)
                        ], dim=1)
                        
                        candidates.append((new_seq_embed, new_tokens, new_log_prob, is_finished))
            
            # Keep top-k beams
            if not candidates:
                break
            
            # Sort by score (with length normalization for unfinished beams)
            candidates.sort(key=lambda x: self.length_normalize(x[2], len(x[1]) + 1), reverse=True)
            beams = candidates[:self.beam_width]
            
            # Stop if all beams are finished
            if all(finished for _, _, _, finished in beams):
                break
        
        # Add remaining unfinished beams to finished
        for seq_embed, tokens, log_prob, finished in beams:
            if not finished and tokens:
                normalized_score = self.length_normalize(log_prob, len(tokens))
                finished_beams.append((tokens, normalized_score))
        
        # Sort finished beams
        finished_beams.sort(key=lambda x: x[1], reverse=True)
        
        return finished_beams[:self.beam_width]


def beam_search_inference(model, xs_batch, column_embeddings, beam_width=5, 
                          max_length=15, schema_len=None, device='cuda'):
    """
    Run beam search on a batch of examples.
    
    Args:
        model: Autoregressive model
        xs_batch: [batch_size, seq_len, n_dims] input batch
        column_embeddings: [vocab_size, n_dims-1] column embeddings
        beam_width: Beam width
        max_length: Max generation length
        schema_len: Length of schema part
        device: Device
    
    Returns:
        predictions: List of predicted sequences for each example
    """
    model.eval()
    
    if schema_len is None:
        schema_len = model.schema_len
    
    searcher = BeamSearcher(
        model=model,
        beam_width=beam_width,
        max_length=max_length
    )
    
    predictions = []
    
    for i in range(xs_batch.shape[0]):
        # Get schema for this example
        xs_schema = xs_batch[i:i+1, :schema_len, :]  # [1, schema_len, n_dims]
        
        # Run beam search
        beams = searcher.search(xs_schema, column_embeddings, device=device)
        
        # Get best sequence
        if beams:
            best_tokens, best_score = beams[0]
            predictions.append({
                'tokens': best_tokens,
                'score': best_score,
                'all_beams': beams
            })
        else:
            predictions.append({
                'tokens': [],
                'score': float('-inf'),
                'all_beams': []
            })
    
    return predictions


def evaluate_with_beam_search(model, xs_batch, ys_batch, labels_batch, 
                              column_embeddings, beam_width=5, device='cuda'):
    """
    Evaluate using beam search and compute accuracy.
    
    Args:
        model: Autoregressive model
        xs_batch: [batch_size, seq_len, n_dims]
        ys_batch: [batch_size, path_len] ground truth tokens
        labels_batch: [batch_size] ground truth labels (1/-1)
        column_embeddings: Column embeddings
        beam_width: Beam width
        device: Device
    
    Returns:
        accuracy: Accuracy on final labels
        exact_match: Exact sequence match rate
    """
    predictions = beam_search_inference(
        model, xs_batch, column_embeddings, 
        beam_width=beam_width, device=device
    )
    
    correct_labels = 0
    exact_matches = 0
    total = len(predictions)
    
    for i, pred in enumerate(predictions):
        pred_tokens = pred['tokens']
        true_tokens = ys_batch[i].cpu().tolist()
        true_label = labels_batch[i].item()
        
        # Check if reached END token (connected) or not (not connected)
        if pred_tokens and pred_tokens[-1] == 3:  # END token
            pred_label = 1
        else:
            pred_label = -1
        
        if pred_label == true_label:
            correct_labels += 1
        
        # Check exact match (for connected cases)
        if true_label == 1:
            # Remove padding and END token from true sequence
            true_tokens_clean = [t for t in true_tokens if t != 0 and t != 3]
            pred_tokens_clean = [t for t in pred_tokens if t != 3]
            
            if pred_tokens_clean == true_tokens_clean:
                exact_matches += 1
    
    accuracy = correct_labels / total if total > 0 else 0
    exact_match_rate = exact_matches / total if total > 0 else 0
    
    return accuracy, exact_match_rate

