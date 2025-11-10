"""
Autoregressive task for table connectivity path search.
Next token prediction + final label prediction.
"""

import torch
import torch.nn as nn


class TableConnectivityAutoregressiveTask:
    def __init__(self, n_dims, batch_size, V=5, C=3, vocab_size=None):
        """
        Autoregressive task for table connectivity.
        
        Args:
            n_dims: Input dimension
            batch_size: Batch size
            V: Number of tables
            C: Columns per table
            vocab_size: Size of vocabulary (special tokens + columns)
        """
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.V = V
        self.C = C
        
        # Vocabulary: [PAD, START, SEP, END] + [col_0, col_1, ..., col_{V*C-1}]
        self.vocab_size = vocab_size if vocab_size else 4 + V * C
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.SEP_TOKEN = 2
        self.END_TOKEN = 3
        self.FIRST_COL_ID = 4
    
    def evaluate(self, xs_b):
        """
        Placeholder for evaluation.
        For autoregressive task, this is not used during training.
        """
        # Return dummy labels
        return torch.zeros(xs_b.shape[0], xs_b.shape[1])
    
    @staticmethod
    def get_training_metric():
        """Return the loss function for training."""
        return nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    @staticmethod
    def get_metric_name():
        """Return metric name."""
        return "cross_entropy"


def get_autoregressive_task(task_name, n_dims, batch_size, **task_kwargs):
    """Get autoregressive task by name."""
    if task_name == "table_connectivity_autoregressive":
        return TableConnectivityAutoregressiveTask(
            n_dims=n_dims,
            batch_size=batch_size,
            **task_kwargs
        )
    else:
        raise NotImplementedError(f"Unknown autoregressive task: {task_name}")

