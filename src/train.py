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


def train_step(model, xs, ys, optimizer, loss_func, print_loss=False, block_size=3, n=1):
    optimizer.zero_grad()
    
    # Check if using custom MatrixChainTransformer
    from models import MatrixChainTransformer
    is_custom_matrix_chain = isinstance(model, MatrixChainTransformer)
    
    if is_custom_matrix_chain:
        # Custom training procedure for MatrixChainTransformer:
        # 1. Mask Y of last M_i and predict it
        # 2. Use ground truth Y to predict Z
        # 3. Only compute loss on last M_i's Y and Z
        
        batch_size = xs.shape[0]
        n = model.n
        L = model.L
        block_size = 3 * n
        last_block_start = (L - 1) * block_size
        
        # Step 1: Mask Y and Z, then predict Y
        # (Z should also be masked because Z depends on Y)
        xs_masked_y = xs.clone()
        y_start = last_block_start + n
        y_end = last_block_start + 2 * n
        z_start = last_block_start + 2 * n
        z_end = last_block_start + 3 * n
        xs_masked_y[:, y_start:y_end, n:2*n] = 0  # Mask Y
        xs_masked_y[:, z_start:z_end, 2*n:3*n] = 0  # Mask Z (because Z depends on Y)
        
        output_y = model(xs_masked_y, ys)

        if print_loss:
            print(xs.shape, ys.shape, output_y.shape)
            # exit()
        y_pred = output_y[:, y_start:y_end, n:2*n]
        y_target = ys[:, y_start:y_end, n:2*n]
        y_loss = loss_func(y_pred, y_target)
        
        # Step 2: Use ground truth Y to predict Z
        xs_with_true_y = xs.clone()
        # Y is already in xs_with_true_y, just mask Z
        z_start = last_block_start + 2 * n
        z_end = last_block_start + 3 * n
        xs_with_true_y[:, z_start:z_end, 2*n:3*n] = 0  # Mask Z
        
        output_z = model(xs_with_true_y, ys)
        z_pred = output_z[:, z_start:z_end, 2*n:3*n]
        z_target = ys[:, z_start:z_end, 2*n:3*n]
        z_loss = loss_func(z_pred, z_target)
        
        # Total loss
        # loss = (y_loss + z_loss) / 2
        loss = y_loss
        
        if print_loss:
            print(f"Y loss: {y_loss.item():.4f}, Z loss: {z_loss.item():.4f}, Total: {loss.item():.4f}")
            print(f"Y pred (first row): {y_pred[0, 0]}")
            print(f"Y target (first row): {y_target[0, 0]}")
        
        output = output_z  # Return the output with both Y and Z predictions
        
    else:
        # Standard training for other models
        if xs.shape == ys.shape:
            output = model(xs, None)
        else:
            output = model(xs, ys)
        # if print_loss:
        #     print(xs.shape)
        #     print(output.shape)
        #     print(ys.shape)
        #     exit()
        # Handle both scalar targets and vector targets
        if ys is not None and len(ys.shape) == 2:
            # # Scalar targets (batch, seq_len): original behavior
            # print(output.shape)
            # print(ys.shape)
            # exit()
            loss = loss_func(output[:,::2], ys[:,:])
        elif block_size == 2:
            # Vector targets (batch, seq_len, n_dims): matrix_chain task
            # For matrix_chain: compute loss on Y and Z positions of each M_i
            # Each M_i has block_size = 3*n positions: X(n), Y(n), Z(n)
            # We want to predict Y and Z from previous positions
            
            ys = xs

            batch_size, seq_len, n_dims = ys.shape
            
            # print(ys.shape)
            # exit()
            # n = n_dims // 3  # Assuming n_dims = 3*n for matrix_chain
            # block_size = 3 * n
            L = seq_len // block_size  # Number of M_i blocks
            # print(seq_len, block_size, L)
            # exit()
            # Collect all Y and Z positions for loss computation
            loss = 0
            for block_idx in [L-1]:
                block_start = block_idx * block_size
                
                # Y positions: [block_start+n : block_start+2*n]
                # y_start = block_start + n
                # y_end = block_start + 2 * n
                # Z positions: [block_start+2*n : block_start+3*n]
                # z_start = block_start + 2 * n
                # z_end = block_start + 3 * n
                
                y_start = block_start + 1
                y_end = block_start + 1 + n
                # z_start = block_start + 1 + n
                # z_end = block_start + 1 + 2 * n

                # For Y: predict from previous position (y_start-1 to y_end-1)
                # Target: ys[:, y_start:y_end, :]
                if y_start > 0:
                    # y_pred = output[:, y_start-1:y_end-1, :]
                    # y_target = ys[:, y_start:y_end, :]
                    y_pred = output[:, ::block_size, 0]
                    y_target = ys[:, 1::block_size, 0]
                    y_loss = loss_func(y_pred, y_target)
                    loss += y_loss
                
                # For Z: predict from previous position (z_start-1 to z_end-1)
                # Target: ys[:, z_start:z_end, :]
                # if z_start > 0:
                #     z_pred = output[:, z_start-1:z_end-1, :]
                #     z_target = ys[:, z_start:z_end, :]
                #     z_loss = loss_func(z_pred, z_target)
                    # losses.append(z_loss)
            
            # Average loss over all Y and Z positions
            # loss 
            
            if print_loss:
                print(f"Mean loss: {loss.item():.4f}")
                # print(xs[0,y_start-1,:])
                print(f"First Y prediction: {output[0, y_start-1, :]}")
                print(f"First Y target:     {ys[0, y_start, :]}")
                # print(ys.shape)
                # print(xs.shape)
                # print(y_pred,y_target,y_loss)
    
        else: # X is a matrix not vector
            ys = xs
            batch_size, seq_len, n_dims = ys.shape
            
            L = seq_len // block_size  # Number of M_i blocks

            
            loss = 0
            for block_idx in [L-1]:
                block_start = block_idx * block_size
            
                y_start = block_start + n
                # print(seq_len, block_size, L)
                # print(block_start, y_start)
                # exit()
                y_end = y_start + n
                # z_start = block_start + 1 + n
                # z_end = block_start + 1 + 2 * n

                # For Y: predict from previous position (y_start-1 to y_end-1)
                # Target: ys[:, y_start:y_end, :]
                if y_start > 0:
                    # y_pred = output[:, y_start-1:y_end-1, :]
                    # y_target = ys[:, y_start:y_end, :]
                    y_pred = output[:, (n-1)::block_size, n:2*n]
                    y_target = ys[:, n::block_size, n:2*n]
                    y_loss = loss_func(y_pred, y_target)
                    loss += y_loss
                
            
            if print_loss:
                print(f"Mean loss: {loss.item():.4f}")
                # print(xs[0,y_start-1,:])
                print(f"First Y prediction: {output[0, y_start-1, n:2*n]}")
                print(f"First Y target:     {ys[0, y_start, n:2*n]}")



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
        xs, ys = task.evaluate(xs)
        # print(xs[0,0,::4])

        # print(xs[0,:,:], ys[0,-1])
        # exit()

        block_size = 1+1

        loss_func = task.get_training_metric()
        if i % 100 == 0:
            print_loss = True
            print(xs[0,:4,:4]@task.last_A_b[0][:,0])
            print(ys[0,4,4])
            # print(xs[0,::2,::4])
            # print(xs[0,::2,::task.n]@task.last_A_b[0][:,0])
            # print(ys[0,1::2,0])
        else:
            print_loss = False
        if args.training.task == "matrix_chain":
            block_size = 3 * data_sampler.n
            loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, print_loss=print_loss, block_size=block_size, n=data_sampler.n)
        elif args.training.task == "matrix_chain_vector":
            block_size = 2 
            loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, print_loss=print_loss, block_size=block_size, n=1)#data_sampler.n)
        else:
            loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, print_loss=print_loss, block_size=block_size, n=1)#data_sampler.n)
        
        if test:
            exit()

        # point_wise_tags = list(range(curriculum.n_points))
        # point_wise_loss_func = task.get_metric()
        # point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        # baseline_loss = (
        #     sum(
        #         max(curriculum.n_dims_truncated - ii, 0)
        #         for ii in range(curriculum.n_points)
        #     )
        #     / curriculum.n_points
        # )

        # if i % args.wandb.log_every_steps == 0 and not args.test_run:
            # wandb.log(
            #     {
            #         "overall_loss": loss,
            #         "excess_loss": loss / baseline_loss,
            #         "pointwise/loss": dict(
            #             zip(point_wise_tags, point_wise_loss.cpu().numpy())
            #         ),
            #         "n_points": curriculum.n_points,
            #         "n_dims": curriculum.n_dims_truncated,
            #     },
            #     step=i,
            # )

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
    xs, ys = task.evaluate(xs)
    if args.training.task == "matrix_chain":
        block_size = 3 * data_sampler.n
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, print_loss=True, block_size=block_size, n=data_sampler.n)
    elif args.training.task == "matrix_chain_vector":
        block_size = 2 
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, print_loss=True, block_size=block_size, n=1)#data_sampler.n)
    else:
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func, print_loss=True, block_size=block_size, n=1)#data_sampler.n)
    
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
