import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, zs1, zs2, optimizer, loss_func, print_loss=False, block_size=3, n=1):
    optimizer.zero_grad()
    
    # Check if using custom MatrixChainTransformer
    from models import MatrixChainTransformer
    is_custom_matrix_chain = isinstance(model, MatrixChainTransformer)
    
        
    batch_size, seq_len, n_dims = ys.shape
    
    L = seq_len // block_size  # Number of M_i blocks

    loss = 0
    output = model._model1(xs, ys)
    for block_idx in [L-1]:
        block_start = block_idx * block_size
    
        y_start = block_start + n
        y_end = y_start + n
        if y_start > 0:
            for i in range(n):
                y_pred = output[:, (n-1+i)::block_size, i]
                y_target = ys[:, n+i::block_size, i]
                y_loss = loss_func(y_pred, y_target)
                loss += y_loss
    
    output21 = model._model2(zs1, zs1)
    for block_idx in [L-1]:
        block_start = block_idx * block_size
    
        y_start = block_start + n
        y_end = y_start + n
        if y_start > 0:
            for i in range(n):
                y_pred = output21[:, (n-1+i)::block_size, i]
                y_target = zs1[:, n+i::block_size, i]
                y_loss = loss_func(y_pred, y_target)
                loss += y_loss


    output22 = model._model2(zs2, zs2)
    for block_idx in [L-1]:
        block_start = block_idx * block_size
    
        y_start = block_start + n
        y_end = y_start + n
        if y_start > 0:
            for i in range(n):
                y_pred = output22[:, (n-1+i)::block_size, i]
                y_target = zs2[:, n+i::block_size, i]
                y_loss = loss_func(y_pred, y_target)
                loss += y_loss

    output = torch.cat([output21, output22], dim=-1)
        
    output = model._linear(output)

    for i in range(n):
        y_pred = output[:, (n-1+i)::block_size, i]
        y_target = zs2[:, n+i::block_size, i]
        y_loss = loss_func(y_pred, y_target)
        loss += y_loss

    if print_loss:
        print(f"Mean loss: {loss.item():.4f}")
        print(f"Last Y prediction: {output[0, block_start:, :]}")
        print(f"Last Y target:     {zs2[0, block_start:, :]}")
        print(f"First Y prediction: {output[0, :block_size, :]}")
        print(f"First Y target:     {zs2[0, :block_size, :]}")


    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = []
    while len(seeds) < count:
        seeds.append(randint(0, total_seeds - 1))
    return seeds


def train(model, args, test=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    # Pass task_kwargs to data_sampler for tasks that need special input formats
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **args.training.task_kwargs)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None and not test:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        # print(xs.shape)
        # print(xs[0,0,::4])
        xs, ys, zs1, zs2 = task.evaluate(xs)
        # print(xs[0,0,::4])
        # print(ys.shape)
        # print(xs[0,:,:], ys[0,-1])
        # exit()

        block_size = 1+1

        loss_func = task.get_training_metric()
        if i % 100 == 0:
            print_loss = True
            # print(task.last_A_b[0][0,:]@xs[0,:2,:2])
            # print(ys[0,:6,:])
            # print(xs[0,::2,::4])
            # print(xs[0,::2,::task.n]@task.last_A_b[0][:,0])
            # print(ys[0,1::2,0])
        else:
            print_loss = False
        if args.training.task == "matrix_chain":
            block_size = 2 * data_sampler.n
            loss, output = train_step(model, xs.cuda(), ys.cuda(), zs1.cuda(), zs2.cuda(), optimizer, loss_func, print_loss=print_loss, block_size=block_size, n=data_sampler.n)
        elif args.training.task == "matrix_chain_vector":
            block_size = 2 
            loss, output = train_step(model, xs.cuda(), ys.cuda(), zs1.cuda(), zs2.cuda(), optimizer, loss_func, print_loss=print_loss, block_size=block_size, n=1)#data_sampler.n)
        else:
            loss, output = train_step(model, xs.cuda(), ys.cuda(), zs1.cuda(), zs2.cuda(), optimizer, loss_func, print_loss=print_loss, block_size=block_size, n=1)#data_sampler.n)
        
        if test:
            exit()

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))
    
    print("Testing...")
    data_sampler_args["seeds"] = None
    xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
    task = task_sampler(**task_sampler_args)
    xs, ys, zs1, zs2 = task.evaluate(xs)

    if args.training.task == "matrix_chain":
        block_size = 2 * data_sampler.n
        loss, output = train_step(model, xs.cuda(), ys.cuda(), zs1.cuda(), zs2.cuda(), optimizer, loss_func, print_loss=True, block_size=block_size, n=data_sampler.n)
    elif args.training.task == "matrix_chain_vector":
        block_size = 2 
        loss, output = train_step(model, xs.cuda(), ys.cuda(), zs1.cuda(), zs2.cuda(), optimizer, loss_func, print_loss=True, block_size=block_size, n=1)#data_sampler.n)
    else:
        loss, output = train_step(model, xs.cuda(), ys.cuda(), zs1.cuda(), zs2.cuda(), optimizer, loss_func, print_loss=True, block_size=block_size, n=1)#data_sampler.n)
    
    print(f"Test loss: {loss}")

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
            mode="disabled",  # Disable wandb
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    # if not args.test_run:
        # _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "lowrank_gpt2", "matrix_chain_transformer"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
