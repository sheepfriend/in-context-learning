"""
Autoregressive sampler for table connectivity path search.
Generates training data with complete BFS paths.
"""

import torch
import networkx as nx
from collections import deque
import random


class TableConnectivityAutoregressiveSampler:
    def __init__(self, n_dims, V=5, C=3, rho=0.5, embedding_seed=42, max_path_samples=5, **kwargs):
        """
        Sampler for autoregressive path search in table connectivity.
        
        Args:
            n_dims: Total number of dimensions
            V: Number of tables
            C: Number of columns per table
            rho: Graph connectivity (edge probability)
            embedding_seed: Seed for generating fixed embeddings
            max_path_samples: Maximum number of paths to sample per query
        """
        self.n_dims = n_dims
        self.V = V
        self.C = C
        self.rho = rho
        self.total_columns = V * C
        self.max_path_samples = max_path_samples
        
        # Generate FIXED embeddings
        torch.manual_seed(embedding_seed)
        self.table_embeddings = torch.randn(V, n_dims - 1)
        self.column_embeddings = torch.randn(self.total_columns, n_dims - 1)
        self.separator_embedding = torch.randn(n_dims - 1)
        torch.manual_seed(torch.initial_seed())
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.SEP_TOKEN = 2
        self.END_TOKEN = 3
        self.FIRST_COL_ID = 4  # Column IDs start from 4
    
    def _bfs_find_all_paths(self, G, table_cols, start_col, end_col, max_length=10):
        """
        Find all paths from start_col to end_col using BFS.
        
        Returns:
            valid_paths: List of paths that reach the end_col
            explored_paths: List of partial paths that didn't reach end_col
        """
        valid_paths = []
        explored_paths = []
        
        # Find which tables contain each column
        col_to_tables = {}
        for table_id, cols in table_cols.items():
            for col in cols:
                if col not in col_to_tables:
                    col_to_tables[col] = []
                col_to_tables[col].append(table_id)
        
        # BFS to find all paths
        queue = deque([(start_col, [start_col], set())])  # (current_col, path, visited_tables)
        
        while queue and len(valid_paths) < 100:  # Limit to prevent infinite exploration
            current_col, path, visited_tables = queue.popleft()
            
            # Check if we reached the target
            if current_col == end_col:
                valid_paths.append(path)
                continue
            
            # Don't explore too long paths
            if len(path) >= max_length:
                explored_paths.append(path)
                continue
            
            # Find tables containing current column
            if current_col not in col_to_tables:
                explored_paths.append(path)
                continue
            
            # Explore neighbors through connected tables
            for table_id in col_to_tables[current_col]:
                if table_id in visited_tables:
                    continue
                
                # Check if this table is connected to other tables
                neighbors = list(G.neighbors(table_id))
                
                for neighbor_table in neighbors:
                    if neighbor_table in visited_tables:
                        continue
                    
                    # Explore columns in the neighbor table
                    for next_col in table_cols[neighbor_table]:
                        if next_col not in path:  # Avoid cycles in column path
                            new_visited = visited_tables | {table_id, neighbor_table}
                            queue.append((next_col, path + [next_col], new_visited))
            
            # Also add this as an explored path if we didn't find neighbors
            if len(path) > 1:
                explored_paths.append(path)
        
        return valid_paths, explored_paths
    
    def _encode_schema(self, table_cols):
        """Encode the schema part (same as before)."""
        n_points_schema = self.V * (self.C + 1) + 1  # +1 for separator
        xs_schema = torch.zeros(n_points_schema, self.n_dims)
        
        row_idx = 0
        for table_id in range(self.V):
            # Encode table name
            if row_idx < n_points_schema:
                xs_schema[row_idx, :-1] = self.table_embeddings[table_id]
                xs_schema[row_idx, -1] = 1000 + table_id
                row_idx += 1
            
            # Encode columns
            for col_id in table_cols[table_id]:
                if row_idx < n_points_schema:
                    xs_schema[row_idx, :-1] = self.column_embeddings[col_id]
                    xs_schema[row_idx, -1] = col_id
                    row_idx += 1
        
        # Separator
        if row_idx < n_points_schema:
            xs_schema[row_idx, :-1] = self.separator_embedding * 0
            xs_schema[row_idx, -1] = 9999
            row_idx += 1
        
        return xs_schema
    
    def _encode_path(self, path, max_path_len):
        """
        Encode a path as a sequence.
        
        Returns:
            xs_path: [max_path_len, n_dims] encoding of the path
            ys_path: [max_path_len] next token targets
            mask: [max_path_len] valid positions (1) vs padding (0)
        """
        xs_path = torch.zeros(max_path_len, self.n_dims)
        ys_path = torch.zeros(max_path_len, dtype=torch.long)
        mask = torch.zeros(max_path_len)
        
        # Encode the path
        for i, col_id in enumerate(path):
            if i >= max_path_len:
                break
            
            xs_path[i, :-1] = self.column_embeddings[col_id]
            xs_path[i, -1] = col_id
            mask[i] = 1
            
            # Target is the next column (or END token if last)
            if i + 1 < len(path):
                ys_path[i] = self.FIRST_COL_ID + path[i + 1]
            else:
                ys_path[i] = self.END_TOKEN
        
        return xs_path, ys_path, mask
    
    def sample_xs(self, n_points, b_size, seeds=None, max_path_len=15):
        """
        Sample input data for autoregressive table connectivity.
        
        Returns:
            xs_b: [b_size, total_seq_len, n_dims] 
            ys_b: [b_size, max_path_len] next token targets
            labels: [b_size] final labels (1 for connected, -1 for not connected)
            masks: [b_size, max_path_len] valid position masks
        """
        if seeds is None:
            generator = None
        else:
            generator = torch.Generator()
        
        all_xs = []
        all_ys = []
        all_labels = []
        all_masks = []
        
        samples_generated = 0
        attempts = 0
        max_attempts = b_size * 10
        
        while samples_generated < b_size and attempts < max_attempts:
            attempts += 1
            
            if seeds is not None and samples_generated < len(seeds):
                seed = seeds[samples_generated]
                generator.manual_seed(seed)
                import numpy as np
                np.random.seed(seed)
                random.seed(seed)
            
            # Generate graph
            G = nx.erdos_renyi_graph(self.V, self.rho, seed=seed if seeds else None)
            
            # Assign columns to tables
            table_cols = {}
            for table_id in range(self.V):
                if seeds is not None:
                    cols = torch.randint(0, self.total_columns, (self.C,), generator=generator).tolist()
                else:
                    cols = torch.randint(0, self.total_columns, (self.C,)).tolist()
                table_cols[table_id] = cols
            
            # Make connected tables share columns
            for u, v in G.edges():
                if seeds is not None:
                    u_col_idx = torch.randint(0, self.C, (1,), generator=generator).item()
                    v_col_idx = torch.randint(0, self.C, (1,), generator=generator).item()
                else:
                    u_col_idx = torch.randint(0, self.C, (1,)).item()
                    v_col_idx = torch.randint(0, self.C, (1,)).item()
                
                shared_col_id = table_cols[u][u_col_idx]
                table_cols[v][v_col_idx] = shared_col_id
            
            # Encode schema
            xs_schema = self._encode_schema(table_cols)
            
            # Generate query columns
            if seeds is not None:
                col1 = torch.randint(0, self.total_columns, (1,), generator=generator).item()
                col2 = torch.randint(0, self.total_columns, (1,), generator=generator).item()
            else:
                col1 = torch.randint(0, self.total_columns, (1,)).item()
                col2 = torch.randint(0, self.total_columns, (1,)).item()
            
            # Find paths using BFS
            valid_paths, explored_paths = self._bfs_find_all_paths(G, table_cols, col1, col2)
            
            # Generate samples from valid paths (connected)
            if valid_paths:
                num_samples = min(len(valid_paths), self.max_path_samples)
                sampled_paths = random.sample(valid_paths, num_samples)
                
                for path in sampled_paths:
                    if samples_generated >= b_size:
                        break
                    
                    xs_path, ys_path, mask = self._encode_path(path, max_path_len)
                    
                    # Concatenate schema and path
                    xs = torch.cat([xs_schema, xs_path], dim=0)
                    
                    all_xs.append(xs)
                    all_ys.append(ys_path)
                    all_labels.append(1)
                    all_masks.append(mask)
                    samples_generated += 1
            
            # Generate samples from explored paths (not connected)
            if explored_paths and samples_generated < b_size:
                num_samples = min(len(explored_paths), self.max_path_samples)
                sampled_paths = random.sample(explored_paths, num_samples)
                
                for path in sampled_paths:
                    if samples_generated >= b_size:
                        break
                    
                    xs_path, ys_path, mask = self._encode_path(path, max_path_len)
                    
                    # Concatenate schema and path
                    xs = torch.cat([xs_schema, xs_path], dim=0)
                    
                    all_xs.append(xs)
                    all_ys.append(ys_path)
                    all_labels.append(-1)
                    all_masks.append(mask)
                    samples_generated += 1
        
        # Pad to batch size if needed
        while samples_generated < b_size:
            all_xs.append(torch.zeros_like(all_xs[0]))
            all_ys.append(torch.zeros_like(all_ys[0]))
            all_labels.append(0)
            all_masks.append(torch.zeros_like(all_masks[0]))
            samples_generated += 1
        
        # Stack into tensors
        xs_b = torch.stack(all_xs, dim=0)
        ys_b = torch.stack(all_ys, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.float)
        masks = torch.stack(all_masks, dim=0)
        
        return xs_b, ys_b, labels, masks


def get_autoregressive_sampler(data_name, n_dims, **kwargs):
    """Get autoregressive sampler by name."""
    if data_name == "table_connectivity_autoregressive":
        return TableConnectivityAutoregressiveSampler(n_dims, **kwargs)
    else:
        raise NotImplementedError(f"Unknown autoregressive sampler: {data_name}")

