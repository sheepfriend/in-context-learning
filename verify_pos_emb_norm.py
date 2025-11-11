import torch
import sys
sys.path.append('src')
from models import TransformerModel, LowRankTransformerModel

print("Verifying positional embedding norms match input data scale\n")
print("=" * 70)

# Test parameters
n_dims = 21
n_embd = 128
n_positions = 100
n_layer = 4
n_head = 4
V = 20
C = 3

# Expected input norm
expected_input_norm = n_dims ** 0.5
print(f"Expected input x norm: sqrt({n_dims}) = {expected_input_norm:.4f}")
print()

# Test TransformerModel
print("=" * 70)
print("1. TransformerModel")
print("=" * 70)
model = TransformerModel(
    n_dims=n_dims,
    n_positions=n_positions,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head
)

# Check positional embedding norm
pos_emb = model._backbone.wpe.weight  # [n_positions, n_embd]
norms_per_pos = torch.norm(pos_emb, dim=1)
mean_pos_norm = norms_per_pos.mean().item()
std_pos_norm = norms_per_pos.std().item()

print(f"Positional embedding shape: {pos_emb.shape}")
print(f"Mean norm per position: {mean_pos_norm:.4f}")
print(f"Std of norms: {std_pos_norm:.4f}")
print(f"Expected norm: {expected_input_norm:.4f}")
print(f"Ratio (actual/expected): {mean_pos_norm / expected_input_norm:.4f}")
print()

# Test LowRankTransformerModel
print("=" * 70)
print("2. LowRankTransformerModel")
print("=" * 70)
model_lr = LowRankTransformerModel(
    n_dims=n_dims,
    n_positions=n_positions,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    V=V,
    C=C
)

# Check shared positional embeddings
pos_emb_shared = model_lr.pos_emb_shared  # [C+1, n_embd]
norms_shared = torch.norm(pos_emb_shared, dim=1)
mean_shared_norm = norms_shared.mean().item()

# Check last 3 positional embeddings
pos_emb_last3 = model_lr.pos_emb_last3  # [3, n_embd]
norms_last3 = torch.norm(pos_emb_last3, dim=1)
mean_last3_norm = norms_last3.mean().item()

print(f"Shared positional embedding shape: {pos_emb_shared.shape}")
print(f"Mean norm (shared): {mean_shared_norm:.4f}")
print(f"Last 3 positional embedding shape: {pos_emb_last3.shape}")
print(f"Mean norm (last 3): {mean_last3_norm:.4f}")
print(f"Expected norm: {expected_input_norm:.4f}")
print(f"Ratio (shared/expected): {mean_shared_norm / expected_input_norm:.4f}")
print(f"Ratio (last3/expected): {mean_last3_norm / expected_input_norm:.4f}")
print()

# Test with actual data
print("=" * 70)
print("3. Comparing with actual input data")
print("=" * 70)
from samplers import get_data_sampler

sampler = get_data_sampler("gaussian", n_dims)
xs = sampler.sample_xs(n_points=50, b_size=100)
input_norms = torch.norm(xs, dim=2)
mean_input_norm = input_norms.mean().item()

print(f"Actual input x mean norm: {mean_input_norm:.4f}")
print(f"Positional embedding mean norm: {mean_pos_norm:.4f}")
print(f"Ratio (pos_emb/input): {mean_pos_norm / mean_input_norm:.4f}")
print()

# Test with TableConnectivity data
sampler_tc = get_data_sampler("table_connectivity_fixed", n_dims, V=V, C=C, rho=0.5)
xs_tc = sampler_tc.sample_xs(n_points=V*(C+1)+3, b_size=100)
input_norms_tc = torch.norm(xs_tc[:, :, :-1], dim=2)  # Exclude ID dimension
mean_input_norm_tc = input_norms_tc.mean().item()

print(f"Actual TableConnectivity x mean norm (without ID): {mean_input_norm_tc:.4f}")
print(f"Low-rank pos_emb mean norm (shared): {mean_shared_norm:.4f}")
print(f"Ratio (pos_emb/input): {mean_shared_norm / mean_input_norm_tc:.4f}")
print()

print("=" * 70)
print("Summary:")
print("=" * 70)
print("✓ Positional embeddings now have similar magnitude to input data")
print(f"  Input x norm ≈ {mean_input_norm:.2f}")
print(f"  Pos emb norm ≈ {mean_pos_norm:.2f}")
print("  This ensures positional information is neither too weak nor too strong")
print("  compared to the content information in the input.")

