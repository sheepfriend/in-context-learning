from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge


model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm", "lowrank_gpt2", "autoregressive_gpt2", "matrix_chain_transformer", "transformer_group"])),
    "n_positions": merge(tinteger, nullable, default(None)),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, nullable, default(12)),
    "n_head": merge(tinteger, required),
    "V": merge(tinteger, nullable, default(20)),  # number of blocks (for lowrank_gpt2 and autoregressive)
    "C": merge(tinteger, nullable, default(3)),   # block size (for lowrank_gpt2 and autoregressive)
    "vocab_size": merge(tinteger, nullable, default(None)),  # vocabulary size (for autoregressive)
    "schema_len": merge(tinteger, nullable, default(None)),  # schema length (for autoregressive)
    "L": merge(tinteger, nullable, default(3)),  # number of M_i blocks (for matrix_chain_transformer)
    "n": merge(tinteger, nullable, default(4)),  # matrix size (for matrix_chain_transformer)
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
    "table_connectivity",
    "table_connectivity_autoregressive",
    "matrix_chain",
    "matrix_chain_vector",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian", "table_connectivity", "table_connectivity_fixed", "table_connectivity_autoregressive", "matrix_chain", "matrix_chain_vector"])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
