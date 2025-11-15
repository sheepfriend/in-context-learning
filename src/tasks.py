import math

import torch
import networkx as nx


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "table_connectivity": TableConnectivity,
        "matrix_chain": MatrixChain,
        "matrix_chain_vector": MatrixChainVector,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return xs_b,ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class TableConnectivity(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        V=5,
        C=3,
        rho=0.5,
    ):
        """
        Graph-based table connectivity task.
        
        The task parses graph structure from x and evaluates connectivity.
        
        Args:
            n_dims: dimension of input  
            batch_size: batch size
            V: maximum number of tables (for reference)
            C: maximum number of columns per table (for reference)
            rho: connectivity parameter (not used in evaluate, only for reference)
        """
        super(TableConnectivity, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.V = V
        self.C = C
        self.rho = rho
    
    def evaluate(self, xs_b):
        """
        Evaluate table connectivity by parsing graph structure from x.
        
        Input xs_b encoding (each row is a name token):
        Format: table1 | col1 | col2 | col3 | table2 | col4 | ... | query_col1 | query_col2 | ...
        
        Last dimension encoding:
        - Table names: >= 1000 (e.g., 1000, 1001, 1002, ...)
        - Column names: < 1000 (e.g., 0, 1, 2, ..., V*C-1)
        
        Output: 1 if columns are connected, -1 otherwise
        """
        batch_size = xs_b.shape[0]
        num_points = xs_b.shape[1]
        ys_b = torch.zeros(batch_size, num_points, device=xs_b.device)
        
        for i in range(batch_size):
            # Parse schema: identify tables and their columns
            col_to_tables = {}
            current_table = None
            schema_end = None
            
            for j in range(num_points):
                name_id = int(xs_b[i, j, -1].item())
                
                if name_id == 9999:  # Separator token
                    schema_end = j + 1
                    break
                elif name_id >= 1000:  # Table name
                    current_table = name_id - 1000
                elif current_table is not None:  # Column name following a table
                    col_id = name_id
                    if col_id not in col_to_tables:
                        col_to_tables[col_id] = []
                    if current_table not in col_to_tables[col_id]:
                        col_to_tables[col_id].append(current_table)
            
            # Build table connectivity graph based on shared columns
            table_graph = {}
            for col_id, tables in col_to_tables.items():
                if len(tables) > 1:
                    for t1 in tables:
                        if t1 not in table_graph:
                            table_graph[t1] = set()
                        for t2 in tables:
                            if t1 != t2:
                                table_graph[t1].add(t2)
            
            # BFS connectivity check
            def is_connected(t1, t2):
                if t1 == t2:
                    return True
                if t1 not in table_graph:
                    return False
                visited = set([t1])
                queue = [t1]
                while queue:
                    current = queue.pop(0)
                    if current == t2:
                        return True
                    for neighbor in table_graph.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                return False
            
            # Evaluate queries: consecutive pairs of columns
            if schema_end is None:
                schema_end = num_points
            
            j = schema_end
            while j < num_points - 1:
                col1 = int(xs_b[i, j, -1].item())
                col2 = int(xs_b[i, j + 1, -1].item())
                
                # Skip if these are table names
                if col1 >= 1000 or col2 >= 1000:
                    j += 1
                    continue
                
                # Find tables containing these columns
                tables_with_col1 = col_to_tables.get(col1, [])
                tables_with_col2 = col_to_tables.get(col2, [])
                
                # Check connectivity
                connected = False
                for t1 in tables_with_col1:
                    for t2 in tables_with_col2:
                        if is_connected(t1, t2):
                            connected = True
                            break
                    if connected:
                        break
                
                # Store result at the second position of the query pair
                ys_b[i, j + 1] = 1.0 if connected else -1.0
                j += 2  # Move to next query pair
        
        xs_b[:, :, -1] = 0

        return xs_b, ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, V=5, C=3, rho=0.5, **kwargs):
        """Generate a pool of graph structures."""
        # For now, we'll use seeds to generate different graphs
        # This allows deterministic generation
        return {"seeds": list(range(num_tasks))}
    
    @staticmethod
    def get_metric():
        return accuracy
    
    @staticmethod
    def get_training_metric():
        return cross_entropy


class MatrixChain(Task):
    """
    Matrix chain transformation task: Y = AX, Z = YB (fixed order)
    
    For each prompt:
    1. Sample L matrices X_i (each n×n where each row ~ N(0, I_n))
    2. Apply transformations: Y_i = A @ X_i, Z_i = Y_i @ B (shared A, B for all i)
    3. Assemble block diagonal matrices M_i = diag(X_i, Y_i, Z_i)
    4. Concatenate all M_i along sequence dimension
    5. Train on predicting Y and Z from previous positions (next token prediction)
    
    For simplicity, we use square matrices: n = m = p = q
    """
    
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        L=3,
        n=4,
        m=4,
        p=4,
        q=4,
    ):
        """
        Args:
            n_dims: Feature dimension (should be 3*n for square matrices)
            batch_size: Batch size
            pool_dict: Optional pool of transformation matrices
            seeds: Optional seeds for reproducibility
            L: Number of X matrices in a prompt
            n, m, p, q: Matrix dimensions (for simplicity, use n=m=p=q)
        
        Matrix dimensions:
            X: n * m
            A: n * n (so Y = AX is n * m)
            B: m * m (so Z = YB is n * m)
            For simplicity: n = m, then all matrices are n x n
        """
        super(MatrixChain, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.L = L
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        
        # For simplicity, use n = m = p = q (all square matrices of same size)
        assert n == m == p == q, "For simplicity, use n=m=p=q (all same size)"
        
        # Store configuration for generating A and B
        self.use_seeds = seeds is not None
        self.use_pool = pool_dict is not None
        
        # Generate transformation matrices A and B based on mode
        if pool_dict is not None:
            # Use pool: fixed A and B from pool
            assert "A" in pool_dict and "B" in pool_dict
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A_b = pool_dict["A"][indices]
            self.B_b = pool_dict["B"][indices]
        elif seeds is not None:
            # Use seeds: generate deterministic A and B
            self.A_b = torch.zeros(self.b_size, self.n, self.n)
            self.B_b = torch.zeros(self.b_size, self.n, self.n)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A_b[i] = torch.randn(self.n, self.n, generator=generator)
                self.B_b[i] = torch.randn(self.n, self.n, generator=generator)
        else:
            # No pool, no seeds: will generate fresh A and B in evaluate()
            self.A_b = None
            self.B_b = None
    
    def evaluate(self, xs_b):
        """
        Apply matrix transformations and assemble block diagonal matrices.
        
        Input:
            xs_b: shape (b_size, L, n, n) - L matrices per batch item
        
        Output:
            xs_assembled: shape (b_size, L*3*n, 3*n) - assembled block diagonal matrices
            ys_b: shape (b_size, L*3*n, 3*n) - targets (full embeddings for next token prediction)
        
        Block structure for each M_i (size 3n x 3n):
            [X  0  0]
            [0  Y  0]
            [0  0  Z]
        where X, Y, Z are each n x n matrices
        
        Transformations:
        - Y = AX
        - Z = YB
        
        For next token prediction, the target at position i is the embedding at position i+1.
        """
        b_size = xs_b.shape[0]
        L = xs_b.shape[1]
        n = self.n
        
        # Generate or retrieve A and B matrices
        if self.A_b is None:
            # No seeds, no pool: generate fresh random A and B for each evaluate call
            A_b = torch.randn(b_size, n, n, device=xs_b.device)
            B_b = torch.randn(b_size, n, n, device=xs_b.device)
        else:
            # Use pre-generated A and B (from seeds or pool)
            A_b = self.A_b.to(xs_b.device)
            B_b = self.B_b.to(xs_b.device)
        self.last_A_b = A_b
        self.last_B_b = B_b
        
        # Each block M_i is 3n x 3n
        block_size = 2 * n
        total_rows = 2 * L * block_size
        
        # Initialize assembled matrices
        xs_assembled = torch.zeros(b_size, total_rows, n, device=xs_b.device)
        
        for i in range(b_size):
            for j in range(L):
                # Get X_j for this batch item
                X = xs_b[i, j]  # shape (n, n)
                
                # Compute Y = XA
                Y =  A_b[i] @ X 
                
                # Compute Z = YB
                Z = B_b[i] @ Y
                
                # Create block diagonal matrix M_j
                # M_j has shape (3n, 3n) with blocks:
                # [X  0  0]
                # [0  Y  0]
                # [0  0  Z]
                
                block_start = j * block_size
                

                # Fill X block (top-left: rows [0:n], cols [0:n])
                xs_assembled[i, block_start:block_start+n, :n] = X
                
                # Fill Y block (middle: rows [n:2n], cols [n:2n])
                xs_assembled[i, block_start+n:block_start+2*n, :n] = Y
                
                # Fill Z block (bottom-right: rows [2n:3n], cols [2n:3n])
                # xs_assembled[i, block_start+2*n:block_start+3*n, 2*n:3*n] = Z
        
            # print(xs_assembled[0,block_start:,:])
            # exit()
        # For next token prediction, ys[i] = xs[i+1]
        # The target is to predict the next embedding
        ys_b = xs_assembled.clone()

        
        return xs_assembled, ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, L=3, n=4, m=4, p=4, q=4, **kwargs):
        """Generate a pool of transformation matrices (assuming n=m=p=q)."""
        return {
            "A": torch.randn(num_tasks, n, n),
            "B": torch.randn(num_tasks, n, n),
        }
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error


class MatrixChainVector(Task):
    """
    Matrix chain transformation task with X reshaped as column vector: Y = AX, Z = YB
    
    For each prompt:
    1. Sample L matrices X_i (each n×n where each row ~ N(0, I_n))
    2. Reshape X_i into column vector x_i (n^2 × 1)
    3. Apply transformations: Y_i = A @ X_i, Z_i = Y_i @ B (shared A, B for all i)
    4. Assemble data as [x, 0, 0; 0, Y, 0; 0, 0, Z] for each i
    5. Concatenate all along sequence dimension
    
    The key difference from MatrixChain: X is stored as a flattened column vector in the first block.
    """
    
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, L=3, n=4, m=4, p=4, q=4):
        super(MatrixChainVector, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.L = L
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        assert n == m == p == q, "For simplicity, use n=m=p=q (all same size)"
        
        self.use_seeds = seeds is not None
        self.use_pool = pool_dict is not None
        
        # Fixed transformation order: Y=XA, Z=YB (no randomness)
        if pool_dict is not None:
            assert "A" in pool_dict and "B" in pool_dict
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A_b = pool_dict["A"][indices]
            self.B_b = pool_dict["B"][indices]
        elif seeds is not None:
            self.A_b = torch.zeros(self.b_size, self.n, self.n)
            self.B_b = torch.zeros(self.b_size, self.n, self.n)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A_b[i] = torch.randn(self.n, self.n, generator=generator)
                self.B_b[i] = torch.randn(self.n, self.n, generator=generator)
        else:
            # No seeds: A and B will be generated dynamically in evaluate()
            self.A_b = None
            self.B_b = None
    
    def evaluate(self, xs_b):
        b_size = xs_b.shape[0]
        L = xs_b.shape[1]
        n = self.n
        
        # Generate A and B dynamically if not already set (no seeds case)
        if self.A_b is None:
            A_b = torch.randn(b_size, n, n, device=xs_b.device)
            B_b = torch.randn(b_size, n, n, device=xs_b.device)
            self.last_A_b = A_b
            self.last_B_b = B_b
        else:
            A_b = self.A_b.to(xs_b.device)
            B_b = self.B_b.to(xs_b.device)
        
        # For vectorized format: first n^2 rows for x vector, then n rows for Y, then n rows for Z
        # Total rows per block: n^2 + n + n = n^2 + 2n
        rows_per_block = 1+1
        
        # n_dims should accommodate the maximum width needed
        # x block: needs 1 column (but we'll use n for consistency)
        # Y block: needs n columns
        # Z block: needs n columns
        # So we use n columns total (n_dims = n)

        xs_assembled = torch.zeros(b_size, (1+1)*L, n*n, device=xs_b.device)
        
        for i in range(b_size):
            for j in range(L):
                X = xs_b[i, j]  # (n, n)
                
                # Fixed transformation order: Y = X @ A, Z = Y @ B
                Y = X @ A_b[i] 
                # print(X[:,0])
                # print(A_b[i][0,:])
                # if i == 0 and j == L-1:
                    # print(X[0,:]@A_b[i][:,0])
                    # print(X[0,:])
                    # print(Y[:,0])
                Z = Y @ B_b[i] 
                
                # Assemble the block
                block_start = j * rows_per_block
                
                # x part: flatten X column-wise and place in first column
                # X flattened: (n*n,)  place in first n*n rows, first column
                x_flat = X.T.reshape(-1)  # Flatten column-wise (Fortran order)
                # print(x_flat[::4])
                xs_assembled[i, block_start, 0:n*n] = x_flat
                
                # Y part: place Y in the diagonal block
                # Rows [n*n : n*n+n], columns [1:n+1]
                y_start = block_start + 1
                xs_assembled[i, y_start:y_start+1, 0:n] = Y[:,0]
                
                # Z part: place Z in the diagonal block
                # Rows [n*n+n : n*n+2n], columns [0:n]
                # z_start = y_start + n
                # xs_assembled[i, z_start:z_start+n, n*n+n:(n*n+n*2)] = Z
        
        ys_b = xs_assembled.clone()


        # print(xs_assembled[0,0,::4])
        # print(self.last_A_b[0,:,0])
        # print(xs_assembled[0,0,::4]@self.last_A_b[0,:,0])
        # exit()
        # print(ys_b.shape)
        # for i in ys_b[0]:
            # print(i)
        # exit()
        return xs_assembled, ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, L=3, n=4, m=4, p=4, q=4, **kwargs):
        """Generate a pool of transformation matrices."""
        return {
            "A": torch.randn(num_tasks, n, n),
            "B": torch.randn(num_tasks, n, n),
        }
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error

class MatrixChainVector_bak(Task):
    """
    Matrix chain transformation task with X reshaped as column vector: Y = AX, Z = YB
    
    For each prompt:
    1. Sample L matrices X_i (each n×n where each row ~ N(0, I_n))
    2. Reshape X_i into column vector x_i (n^2 × 1)
    3. Apply transformations: Y_i = A @ X_i, Z_i = Y_i @ B (shared A, B for all i)
    4. Assemble data as [x, 0, 0; 0, Y, 0; 0, 0, Z] for each i
    5. Concatenate all along sequence dimension
    
    The key difference from MatrixChain: X is stored as a flattened column vector in the first block.
    """
    
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, L=3, n=4, m=4, p=4, q=4):
        super(MatrixChainVector, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.L = L
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        assert n == m == p == q, "For simplicity, use n=m=p=q (all same size)"
        
        self.use_seeds = seeds is not None
        self.use_pool = pool_dict is not None
        
        # Fixed transformation order: Y=XA, Z=YB (no randomness)
        if pool_dict is not None:
            assert "A" in pool_dict and "B" in pool_dict
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A_b = pool_dict["A"][indices]
            self.B_b = pool_dict["B"][indices]
        elif seeds is not None:
            self.A_b = torch.zeros(self.b_size, self.n, self.n)
            self.B_b = torch.zeros(self.b_size, self.n, self.n)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A_b[i] = torch.randn(self.n, self.n, generator=generator)
                self.B_b[i] = torch.randn(self.n, self.n, generator=generator)
        else:
            # No seeds: A and B will be generated dynamically in evaluate()
            self.A_b = None
            self.B_b = None
    
    def evaluate(self, xs_b):
        b_size = xs_b.shape[0]
        L = xs_b.shape[1]
        n = self.n
        
        # Generate A and B dynamically if not already set (no seeds case)
        if self.A_b is None:
            A_b = torch.randn(b_size, n, n, device=xs_b.device)
            B_b = torch.randn(b_size, n, n, device=xs_b.device)
        else:
            A_b = self.A_b.to(xs_b.device)
            B_b = self.B_b.to(xs_b.device)
        
        # For vectorized format: first n^2 rows for x vector, then n rows for Y, then n rows for Z
        # Total rows per block: n^2 + n + n = n^2 + 2n
        rows_per_block = 2*n+1
        
        # n_dims should accommodate the maximum width needed
        # x block: needs 1 column (but we'll use n for consistency)
        # Y block: needs n columns
        # Z block: needs n columns
        # So we use n columns total (n_dims = n)

        xs_assembled = torch.zeros(b_size, (n*2+1)*L, n*n+n*2, device=xs_b.device)
        
        for i in range(b_size):
            for j in range(L):
                X = xs_b[i, j]  # (n, n)
                
                # Fixed transformation order: Y = X @ A, Z = Y @ B
                Y = X @ A_b[i]
                Z = Y @ B_b[i]
                
                # Assemble the block
                block_start = j * rows_per_block
                
                # x part: flatten X column-wise and place in first column
                # X flattened: (n*n,)  place in first n*n rows, first column
                x_flat = X.T.reshape(-1)  # Flatten column-wise (Fortran order)
                xs_assembled[i, block_start, 0:n*n] = x_flat
                
                # Y part: place Y in the diagonal block
                # Rows [n*n : n*n+n], columns [1:n+1]
                y_start = block_start + 1
                xs_assembled[i, y_start:y_start+n, ] = Y
                
                # Z part: place Z in the diagonal block
                # Rows [n*n+n : n*n+2n], columns [0:n]
                z_start = y_start + n
                xs_assembled[i, z_start:z_start+n, n*n+n:(n*n+n*2)] = Z
        
        ys_b = xs_assembled.clone()
        # print(ys_b.shape)
        # for i in ys_b[0]:
            # print(i)
        # exit()
        return xs_assembled, ys_b
    
    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, L=3, n=4, m=4, p=4, q=4, **kwargs):
        """Generate a pool of transformation matrices."""
        return {
            "A": torch.randn(num_tasks, n, n),
            "B": torch.randn(num_tasks, n, n),
        }
    
    @staticmethod
    def get_metric():
        return squared_error
    
    @staticmethod
    def get_training_metric():
        return mean_squared_error
