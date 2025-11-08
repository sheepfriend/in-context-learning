import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "table_connectivity": TableConnectivitySampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, **kwargs):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        # Ignore extra kwargs (e.g., V, C, rho from task_kwargs)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class TableConnectivitySampler(DataSampler):
    def __init__(self, n_dims, V=5, C=3, rho=0.5, bias=None, scale=None, **kwargs):
        """
        Sampler for TableConnectivity task.
        
        Generates graph structures and encodes them in x along with queries.
        
        Args:
            n_dims: Total number of dimensions
            V: Number of tables
            C: Number of columns per table  
            rho: Graph connectivity (edge probability)
            bias: Optional bias for the features
            scale: Optional scaling transformation
        """
        super().__init__(n_dims)
        self.V = V
        self.C = C
        self.rho = rho
        self.total_columns = V * C
        self.bias = bias
        self.scale = scale
        
        # Generate fixed random vectors for each table name and column name
        # Table names: 0 to V-1
        self.table_embeddings = torch.randn(V, n_dims - 1)
        # Column names: 0 to total_columns-1
        self.column_embeddings = torch.randn(self.total_columns, n_dims - 1)
        # Separator embedding
        self.separator_embedding = torch.randn(n_dims - 1)

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        """
        Sample input data for TableConnectivity task.
        
        Encoding format - each row is a "name token":
        Schema: table1_name | col1 | col2 | ... | colC | table2_name | col1 | col2 | ... 
        Query:  query_col1 | query_col2 | ...
        
        Name encoding (last dimension):
        - Table names: 1000 + table_id (e.g., 1000, 1001, 1002, ...)
        - Column names: column_id (e.g., 0, 1, 2, ..., V*C-1)
        
        Graph edges are implicitly encoded: columns with same ID appearing in different tables are connected.
        """
        import networkx as nx
        
        n_points = self.V * (self.C+1) + 3
        xs_b = torch.zeros(b_size, n_points, self.n_dims)
        # print(self.V,self.C,self.rho)
        # print(xs_b.shape)
        if seeds is None:
            generator = None
        else:
            generator = torch.Generator()
            assert len(seeds) == b_size
        
        for i in range(b_size):
            # print(seeds[i])
            if seeds is not None:
                generator.manual_seed(seeds[i])
                import numpy as np
                np.random.seed(seeds[i])
            
            # Generate graph
            G = nx.erdos_renyi_graph(self.V, self.rho, seed=seeds[i] if seeds else None)
            
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
            
            # Encode schema: table_name | col1 | col2 | ... | colC | table_name | ...
            row_idx = 0
            for table_id in range(self.V):
                # Encode table name using fixed random vector
                if row_idx < n_points:
                    xs_b[i, row_idx, :-1] = self.table_embeddings[table_id]
                    xs_b[i, row_idx, -1] = 1000 + table_id  # Table name encoding
                    row_idx += 1
                
                # Encode columns of this table using fixed random vectors
                for col_id in table_cols[table_id]:
                    if row_idx < n_points:
                        xs_b[i, row_idx, :-1] = self.column_embeddings[col_id]
                        xs_b[i, row_idx, -1] = col_id  # Column name encoding
                        row_idx += 1
            
            # Insert separator between schema and queries (using 9999 as separator)
            if row_idx < n_points:
                xs_b[i, row_idx, :-1] = self.separator_embedding*0
                xs_b[i, row_idx, -1] = 9999  # Separator token
                row_idx += 1
            
            # Encode queries: query_col1 | query_col2 | ...
            num_queries = (n_points - row_idx) // 2  # Each query takes 2 rows
            # print(num_queries)
            for q in range(num_queries):
                if row_idx < n_points:
                    # Generate random query columns
                    if seeds is not None:
                        col1 = torch.randint(0, self.total_columns, (1,), generator=generator).item()
                        col2 = torch.randint(0, self.total_columns, (1,), generator=generator).item()
                    else:
                        col1 = torch.randint(0, self.total_columns, (1,)).item()
                        col2 = torch.randint(0, self.total_columns, (1,)).item()
                    
                    # Encode query_col1 using fixed random vector
                    xs_b[i, row_idx, :-1] = self.column_embeddings[col1]
                    xs_b[i, row_idx, -1] = col1
                    row_idx += 1
                    
                    # Encode query_col2 using fixed random vector
                    if row_idx < n_points:
                        xs_b[i, row_idx, :-1] = self.column_embeddings[col2]
                        xs_b[i, row_idx, -1] = col2
                        row_idx += 1
        
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:-1] = 0  # Don't zero out the last dimension (name ID)
        
        return xs_b
