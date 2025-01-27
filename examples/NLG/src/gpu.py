#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os

def add_gpu_params(parser: argparse.ArgumentParser):
    """Add GPU-related arguments to the parser."""
    parser.add_argument("--platform", default="local", type=str, help="Platform type: local, azure, philly, k8s.")
    parser.add_argument("--local_rank", default=0, type=int, help="Local rank for distributed training.")
    parser.add_argument("--rank", default=0, type=int, help="Global rank for distributed training.")
    parser.add_argument("--device", default=0, type=int, help="Device index for single GPU training.")
    parser.add_argument("--world_size", default=1, type=int, help="World size for distributed training.")
    parser.add_argument("--random_seed", default=10, type=int, help="Random seed for reproducibility.")


def distributed_opt(args, model, opt, grad_acc=1):
    """Wrap model and optimizer for distributed training."""
    if args.platform in ["azure", "philly", "k8s", "local"] and args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=False, broadcast_buffers=False
        )
    return model, opt


def distributed_sync(args):
    """Synchronize across all processes."""
    if args.world_size > 1 and hasattr(args, "dist"):
        args.dist.barrier()


def distributed_gather(args, tensor):
    """Gather tensors across all processes."""
    if args.world_size > 1 and hasattr(args, "dist"):
        g_y = [torch.zeros_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(g_y, tensor)
        return torch.stack(g_y)
    return tensor.unsqueeze(0)  # For single process, return the tensor wrapped in a list.


def parse_gpu(args):
    """Initialize GPU or distributed training setup."""
    torch.manual_seed(args.random_seed)

    if args.platform == "local":
        # Single GPU or local multi-GPU setup
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cpu")
        args.rank = 0
        args.device = device
        args.world_size = 1
    elif args.platform == "azure":
        import horovod.torch as hvd
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        args.local_rank = hvd.local_rank()
        args.rank = hvd.rank()
        args.world_size = hvd.size()
        args.device = torch.device("cuda", hvd.local_rank())
        args.hvd = hvd
    elif args.platform == "philly":
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="gloo")
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.device = torch.device("cuda", args.local_rank)
        args.dist = dist
    elif args.platform == "k8s":
        master_uri = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.local_rank = local_rank
        dist.init_process_group(
            backend="gloo",
            init_method=master_uri,
            world_size=int(os.environ["OMPI_COMM_WORLD_SIZE"]),
            rank=int(os.environ["OMPI_COMM_WORLD_RANK"]),
        )
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.device = torch.device("cuda", local_rank)
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.dist = dist
    print(
        f"Rank: {args.rank}, Local Rank: {args.local_rank}, "
        f"World Size: {args.world_size}, Device: {args.device}"
    )


def cleanup(args):
    """Clean up distributed training setup."""
    if hasattr(args, "dist") and args.world_size > 1:
        args.dist.destroy_process_group()


# Example usage for local single-GPU training:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed training setup.")
    add_gpu_params(parser)
    args = parser.parse_args()

    # Example of setting local environment variables for single GPU
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # Parse GPU and distributed setup
    parse_gpu(args)

    # Dummy model and optimizer for demonstration
    model = nn.Linear(10, 10).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Wrap for distributed training if necessary
    model, optimizer = distributed_opt(args, model, optimizer)

    # Example forward-backward loop
    x = torch.randn(4, 10).to(args.device)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Sync or cleanup
    distributed_sync(args)
    cleanup(args)
