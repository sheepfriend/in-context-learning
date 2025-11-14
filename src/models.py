import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "lowrank_gpt2":
        # Extract V and C from task_kwargs if available
        V = getattr(conf, 'V', 20)
        C = getattr(conf, 'C', 3)
        model = LowRankTransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            V=V,
            C=C,
        )
    elif conf.family == "matrix_chain_transformer":
        # Extract L and n for matrix chain
        L = getattr(conf, 'L', 3)
        n = getattr(conf, 'n', 4)
        model = MatrixChainTransformer(
            n_dims=conf.n_dims,
            n_embd=conf.n_embd,
            n_head=conf.n_head,
            L=L,
            n=n,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "table_connectivity": [
        ],
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        # Output full n_dims dimensional vector for next token prediction
        self._read_out = nn.Linear(n_embd, n_dims)
        
        # Reinitialize positional embeddings to match input scale
        # Input x has norm ≈ sqrt(n_dims), we want pos_emb to have similar magnitude
        # For a vector of dimension n_embd with std σ, norm ≈ sqrt(n_embd) * σ
        # Setting σ = sqrt(n_dims / n_embd) gives norm ≈ sqrt(n_dims)
        pos_emb_std = 1.0 # (n_dims / n_embd) ** 0.5
        nn.init.normal_(self._backbone.wpe.weight, mean=0.0, std=pos_emb_std)

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        embeds = self._read_in(xs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        
        # Check if ys is scalar (shape: batch, seq_len) or vector (shape: batch, seq_len, n_dims)
        if len(ys.shape) == 2:
            # Scalar target: return only first dimension for backward compatibility
            return prediction[:, :, 0]
        else:
            # Vector target: return full n_dims dimensional prediction
            return prediction


class LowRankTransformerModel(nn.Module):
    """
    Transformer with low-rank positional embeddings and custom attention masks.
    
    Positional embedding structure:
    - First V*(C+1) positions: share a single learned embedding of length C+1, repeated V times
    - Last 3 positions: have independent learned embeddings
    
    Attention mask structure:
    - First 2 layers: Each group of C+1 tokens attend to each other (block diagonal),
                      last 3 tokens attend to themselves only
    - Remaining layers: Last token of each C+1 group and the last 3 tokens attend to each other
    """
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, V=20, C=3):
        super(LowRankTransformerModel, self).__init__()
        self.V = V
        self.C = C
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        
        # Expected sequence length: V*(C+1) + 3
        self.seq_len = V * (C + 1) + 3
        
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"lowrank_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}_V={V}_C={C}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)
        
        # Low-rank positional embeddings
        # Match the scale of input data: x has norm ≈ sqrt(n_dims)
        # For n_embd dimensional embeddings with std σ, norm ≈ sqrt(n_embd) * σ
        # Setting σ = sqrt(n_dims / n_embd) gives norm ≈ sqrt(n_dims)
        pos_emb_std = 1.0 # (n_dims / n_embd) ** 0.5
        
        # One shared embedding for the repeated pattern of length C+1
        self.pos_emb_shared = nn.Parameter(torch.randn(C + 1, n_embd) * pos_emb_std)
        # Three independent embeddings for the last 3 positions
        self.pos_emb_last3 = nn.Parameter(torch.randn(3, n_embd) * pos_emb_std)
        
        # Disable the default positional embeddings in GPT2
        # We'll add our custom ones manually
        self._backbone.wpe.weight.requires_grad = False
        
        # Create attention masks for different layers
        self._register_attention_masks()

    def _register_attention_masks(self):
        """Create and register attention masks for different layers."""
        # Mask for first 2 layers: block diagonal + last 3 tokens
        mask_first_layers = torch.zeros(self.seq_len, self.seq_len)
        
        # Each C+1 group attends to itself
        for i in range(self.V):
            start = i * (self.C + 1)
            end = start + (self.C + 1)
            mask_first_layers[start:end, start:end] = 1
        
        # Last 3 tokens attend to themselves
        mask_first_layers[-3, -3] = 1
        mask_first_layers[-2, -2] = 1
        mask_first_layers[-1, -1] = 1
        
        # Mask for remaining layers: last token of each group + last 3 tokens
        mask_remaining_layers = torch.zeros(self.seq_len, self.seq_len)
        
        # Collect indices of last tokens in each C+1 group
        last_token_indices = [(i + 1) * (self.C + 1) - 1 for i in range(self.V)]
        last_token_indices.extend([self.seq_len - 3, self.seq_len - 2, self.seq_len - 1])
        
        # These tokens can all attend to each other
        for i in last_token_indices:
            for j in last_token_indices:
                mask_remaining_layers[i, j] = 1
        
        # Convert to attention mask format (0 = attend, -inf = mask)
        # We need to invert: 1 -> 0 (can attend), 0 -> -inf (cannot attend)
        self.mask_first_layers = (1 - mask_first_layers) * -1e9
        self.mask_remaining_layers = (1 - mask_remaining_layers) * -1e9
        
    def _get_low_rank_pos_embeddings(self, seq_len):
        """Generate low-rank positional embeddings."""
        # For the first V*(C+1) positions, repeat the shared embedding
        pos_emb_repeated = self.pos_emb_shared.repeat(self.V, 1)  # [V*(C+1), n_embd]
        
        # Concatenate with the last 3 independent embeddings
        pos_emb = torch.cat([pos_emb_repeated, self.pos_emb_last3], dim=0)  # [V*(C+1)+3, n_embd]
        
        return pos_emb[:seq_len]  # Truncate if needed
    
    def _apply_custom_attention(self, embeds):
        """Apply transformer with custom attention masks."""
        # Get position embeddings and add to input embeddings
        seq_len = embeds.shape[1]
        pos_emb = self._get_low_rank_pos_embeddings(seq_len)
        embeds = embeds + pos_emb.unsqueeze(0)
        
        # We need to manually pass through transformer layers with custom masks
        # For simplicity, we'll modify the GPT2 model's forward pass
        # This requires accessing internal layers
        
        hidden_states = embeds
        for i, block in enumerate(self._backbone.h):
            # Choose mask based on layer index
            if i < 2:
                attn_mask = self.mask_first_layers[:seq_len, :seq_len].to(embeds.device)
            else:
                attn_mask = self.mask_remaining_layers[:seq_len, :seq_len].to(embeds.device)
            
            # Expand mask for batch and heads: [batch, 1, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            
            # Apply transformer block with custom attention mask
            outputs = block(hidden_states, attention_mask=attn_mask)
            hidden_states = outputs[0]
        
        # Apply final layer norm
        hidden_states = self._backbone.ln_f(hidden_states)
        
        return hidden_states

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        embeds = self._read_in(xs)
        
        # Use custom attention with low-rank positional embeddings
        output = self._apply_custom_attention(embeds)
        
        prediction = self._read_out(output)
        return prediction[:, :, 0]  # predict only on xs


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)


class MatrixChainTransformer(nn.Module):
    """
    Custom Transformer for Matrix Chain task.
    
    Architecture:
    1. Process [M_1, ..., M_L] and [M_1^T, ..., M_L^T] through separate 1-layer transformers
    2. Concatenate outputs and process through two transformers (no positional encoding)
    3. MLP to predict Y and Z of last M_i
    
    Training:
    - Predict Y of last M_i with Y masked
    - Predict Z of last M_i given ground truth Y
    - Loss only on last M_i's Y and Z
    """
    
    def __init__(self, n_dims, n_embd=128, n_head=4, L=3, n=4):
        super(MatrixChainTransformer, self).__init__()
        self.n_dims = n_dims  # Should be 3*n for matrix chain
        self.n_embd = n_embd
        self.n_head = n_head
        self.L = L  # Number of M_i blocks
        self.n = n  # Size of each sub-matrix
        self.block_size = 3 * n  # Size of each M_i
        self.n_positions = L * self.block_size
        
        self.name = f"matrix_chain_transformer_L={L}_n={n}_embd={n_embd}"
        
        # Stage 1: Two separate 1-layer transformers for original and transposed sequences
        # Transformer 1: for rows of [M_1, ..., M_L]
        self.embed_1 = nn.Linear(n_dims, n_embd)  # Each row is a token (3n dimensions)
        encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=n_embd, 
            nhead=n_head, 
            dim_feedforward=4*n_embd,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=1)
        
        # Transformer 2: for rows of [M_1^T, ..., M_L^T]
        self.embed_2 = nn.Linear(n_dims, n_embd)  # Each row is a token (3n dimensions)
        encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4*n_embd,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=1)
        
        # Stage 2: Two transformers without positional encoding
        encoder_layer_3 = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4*n_embd,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=1)
        
        # Transformer 4 operates on reshaped sequence with token dimension 3n
        encoder_layer_4 = nn.TransformerEncoderLayer(
            d_model=3*n,  # Token dimension is 3n after reshape
            nhead=n_head,
            dim_feedforward=4*3*n,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.transformer_4 = nn.TransformerEncoder(encoder_layer_4, num_layers=1)
        
        # Learnable pooling: MLP to compute attention weights for pooling
        self.pooling_mlp = nn.Sequential(
            nn.Linear(2 * n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, 1)  # Output attention weight for each token
        )
        
        # Stage 3: MLP to map global representation to Y and Z matrices
        # Output a 6n×6n matrix in the form [Y, 0; 0, Z] where Y and Z are each 3n×3n
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, 2 * n_embd),
            nn.GELU(),
            nn.Linear(2 * n_embd, 2 * (3*n) * (3*n))  # Output Y (3n×3n) and Z (3n×3n)
        )
        
    def forward(self, xs, ys, inds=None):
        """
        Forward pass.
        
        Args:
            xs: (batch_size, L*3*n, 3*n) - input matrices
            ys: (batch_size, L*3*n, 3*n) - target matrices
        
        Returns:
            predictions: (batch_size, L*3*n, 3*n) - predictions for all positions
        """
        batch_size = xs.shape[0]
        
        # Extract M_i blocks (shape: batch, L, 3n, 3n)
        M_blocks = xs.view(batch_size, self.L, self.block_size, self.n_dims)
        
        # Reshape to treat each row as a token: (batch, L*3n, 3n)
        M_rows = M_blocks.view(batch_size, self.L * self.block_size, self.n_dims)
        
        # Stage 1: Process original and transposed sequences
        # Each row of M_i is a token
        h1 = self.embed_1(M_rows)  # (batch, L*3n, n_embd)
        h1 = self.transformer_1(h1)  # (batch, L*3n, n_embd)
        
        # Transposed sequence: transpose each M_i and treat rows as tokens
        M_transposed = M_blocks.transpose(-2, -1)  # (batch, L, 3n, 3n)
        M_transposed_rows = M_transposed.reshape(batch_size, self.L * self.block_size, self.n_dims)
        
        h2 = self.embed_2(M_transposed_rows)  # (batch, L*3n, n_embd)
        h2 = self.transformer_2(h2)  # (batch, L*3n, n_embd)
        
        # Concatenate: [all rows of M, all rows of M^T]
        h_concat = torch.cat([h1, h2], dim=1)  # (batch, L*3n*2, n_embd)
        
        # Stage 2: Two transformers
        h3 = self.transformer_3(h_concat)  # (batch, L*3n*2, n_embd)
        
        # Reshape for second transformer: treat embedding dimensions as sequence
        # (batch, L*3n*2, n_embd) -> (batch, L*n_embd*2, 3n)
        h3_reshaped = h3.view(batch_size, -1, self.block_size)  # (batch, L*n_embd*2, 3n)
        h4 = self.transformer_4(h3_reshaped)  # (batch, L*n_embd*2, 3n)
        
        # Reshape back to match h3's shape
        h4_reshaped = h4.view(batch_size, self.L * self.block_size * 2, self.n_embd)  # (batch, L*3n*2, n_embd)
        
        # Concatenate h3 and h4
        h_final = torch.cat([h3, h4_reshaped], dim=-1)  # (batch, L*3n*2, 2*n_embd)
        
        # Learnable attention pooling over sequence dimension
        attn_weights = self.pooling_mlp(h_final)  # (batch, L*3n*2, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize over sequence
        h_pooled = (h_final * attn_weights).sum(dim=1)  # (batch, 2*n_embd)
        
        # MLP to generate Y and Z matrices
        mlp_out = self.mlp(h_pooled)  # (batch, 2*3n*3n)
        mlp_out = mlp_out.view(batch_size, 2, 3*self.n, 3*self.n)  # (batch, 2, 3n, 3n)
        
        # Extract Y and Z
        Y_pred = mlp_out[:, 0, :, :]  # (batch, 3n, 3n)
        Z_pred = mlp_out[:, 1, :, :]  # (batch, 3n, 3n)
        
        # Construct 6n×6n block diagonal matrix [Y, 0; 0, Z]
        output_6n = torch.zeros(batch_size, 6*self.n, 6*self.n, device=xs.device)
        output_6n[:, :3*self.n, :3*self.n] = Y_pred  # Top-left: Y
        output_6n[:, 3*self.n:, 3*self.n:] = Z_pred  # Bottom-right: Z
        
        # Reshape to match expected output format (batch, L*3n, 3n)
        # For now, we need to figure out how to map 6n×6n to L*3n×3n
        # Assuming we only fill in the last M_L block
        output = torch.zeros_like(xs)
        
        last_block_start = (self.L - 1) * self.block_size
        
        # Map the 6n×6n output to the last M_L's Y and Z positions
        # Y: from [0:3n, 0:3n] in output_6n to appropriate position in output
        # Z: from [3n:6n, 3n:6n] in output_6n to appropriate position in output
        
        # For the last M_L block, we have positions:
        # Y rows: [last_block_start+n : last_block_start+2n]
        # Z rows: [last_block_start+2n : last_block_start+3n]
        
        # Extract Y and Z submatrices (n×n each from the center of 3n×3n)
        Y_center = Y_pred[:, self.n:2*self.n, self.n:2*self.n]  # (batch, n, n)
        Z_center = Z_pred[:, self.n:2*self.n, self.n:2*self.n]  # (batch, n, n)
        
        # Fill in the output
        y_start = last_block_start + self.n
        z_start = last_block_start + 2 * self.n
        
        output[:, y_start:y_start+self.n, self.n:2*self.n] = Y_center
        output[:, z_start:z_start+self.n, 2*self.n:3*self.n] = Z_center
        
        return output
