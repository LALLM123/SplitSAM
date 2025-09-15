import argparse
import time
import math
import os
import itertools
import warnings
import torch
import random
import copy
from torch.utils.data import DataLoader
import pandas as pd

from gpu import add_gpu_params, parse_gpu, distributed_opt, distributed_sync, cleanup
from optimizer import (
    create_optimizer_scheduler,
    add_optimizer_params,
    create_adam_optimizer_from_args,
)

from data_utils import FT_Dataset
from splitmodel import GPT2Config, GPT2LMModel_Server, GPT2LMModel_Client
from exp_utils import create_exp_dir

import loralib as lora

torch.set_printoptions(threshold=100000)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="PyTorch GPT2 ft script")

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument(
    "--train_data0", required=True, help="location of training data corpus"
)
parser.add_argument(
    "--train_data1", required=True, help="location of training data corpus"
)
parser.add_argument(
    "--train_data2", required=True, help="location of training data corpus"
)
parser.add_argument(
    "--valid_data", required=True, help="location of validation data corpus"
)
parser.add_argument("--train_batch_size", type=int, default=8, help="training batch size")
parser.add_argument("--valid_batch_size", type=int, default=4, help="validation batch size")
parser.add_argument("--grad_acc", type=int, default=1, help="gradient accumulation steps")
parser.add_argument("--clip", type=float, default=0.0, help="gradient clip")
parser.add_argument("--seq_len", type=int, default=512, help="number of tokens to predict.")
parser.add_argument(
    "--model_card",
    default="gpt2.md",
    choices=["gpt2.sm", "gpt2.md", "gpt2.lg"],
    help="model names",
)
parser.add_argument("--init_checkpoint", default=None, help="pretrained checkpoint path")
parser.add_argument("--fp16", action="store_true", help="train model with fp16")
parser.add_argument("--log_interval", type=int, default=100, help="log interval")
parser.add_argument("--eval_interval", type=int, default=2000, help="eval interval")
parser.add_argument("--save_interval", type=int, default=500, help="save interval")
parser.add_argument(
    "--work_dir",
    type=str,
    default=os.getenv("PT_OUTPUT_DIR", "gpt2_model"),
    help="working folder.",
)
parser.add_argument("--lora_dim", type=int, default=0, help="lora attn dimension")
parser.add_argument("--lora_alpha", type=int, default=128, help="lora attn alpha")
parser.add_argument(
    "--obj",
    default="clm",
    choices=["jlm", "clm"],
    help="language model training objective",
)
parser.add_argument(
    "--lora_dropout",
    default=0.0,
    type=float,
    help="dropout probability for lora layers",
)
parser.add_argument("--label_smooth", default=0.0, type=float, help="label smoothing")
parser.add_argument("--roll_interval", type=int, default=-1, help="rolling interval")
parser.add_argument("--roll_lr", type=float, default=0.00001, help="rolling learning rate")
parser.add_argument("--roll_step", type=int, default=100, help="rolling step")
parser.add_argument("--eval_epoch", type=int, default=1, help="eval per number of epochs")

parser.add_argument("--sam_rho", type=float, default=0.2, help="Rho for SAM.")
parser.add_argument('--rho_min', type=float, default=0.0001, help='Minimum value for SAM rho')
parser.add_argument('--rho_max', type=float, default=1.0, help='Maximum value for SAM rho')
parser.add_argument('--beta', type=float, default=0.0005, help='Rho learning rate')
parser.add_argument('--eta', type=float, default=0.1, help='Inner loop learning rate for LETS')

def print_args(args):
    if args.rank == 0:
        print("=" * 100)
        for k, v in args.__dict__.items():
            print(f"        - {k} : {v}")
        print("=" * 100)


def save_checkpoint(w_glob_client, model_server, args, train_step, num_clients):
    if args.rank != 0:
        return

    model_state_dict = {}

    # rename the key in client model
    for key, value in w_glob_client.items():
        new_key = ""
        if key.startswith("transformer_Client"):
            new_key = key.replace("transformer_Client", "module.transformer")
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    # rename the key in server model
    for key, value in model_server.state_dict().items():
        new_key = ""
        if key.startswith("module.transformer_Server"):
            new_key = key.replace("module.transformer_Server", "module.transformer")

        if new_key.startswith("module.transformer.h."):
            parts = key.split(".")
            layer_idx = int(parts[3])
            new_key = ".".join(["module.transformer.h", str(layer_idx + 3)] + parts[4:])
            model_state_dict[new_key] = value
        else:
            if not new_key:
                # no replacement
                model_state_dict[key] = value
            else:
                model_state_dict[new_key] = value

    if args.model_card == "gpt2.md":
        model_path = os.path.join(
            "./trained_models/GPT2_M/e2e",
            f"model_sfl_LETS.{train_step}_r={args.lora_dim}_num={num_clients}_block=3_seed={args.random_seed}_rho={args.sam_rho}.pt",
        )
    if args.model_card == "gpt2.sm":
        model_path = os.path.join(
            "./trained_models/GPT2_S/e2e",
            f"model_sfl_LETS.{train_step}_r={args.lora_dim}_num={num_clients}_block=3_seed={args.random_seed}_rho={args.sam_rho}.pt",
        )
    print("saving checkpoint", model_path)
    torch.save({"model_state_dict": model_state_dict}, model_path)


def fed_avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_norms(params):

    norm_w_sq = 0.0  
    norm_g_sq = 0.0  

    for p in params:
        if p.data is not None: 
            norm_w_sq += p.data.norm(2).item()**2
        if p.grad is not None:
            norm_g_sq += p.grad.data.norm(2).item()**2

    norm_w = math.sqrt(norm_w_sq) if norm_w_sq > 0 else 1.0
    norm_g = math.sqrt(norm_g_sq) if norm_g_sq > 0 else 1.0

    return norm_w, norm_g


def optimizer_step(
    _loss,
    optimizer_server,
    model_server,
    optimizer_client,
    _schedule,
    client_hidden_states,
    hidden_states,
    args,
    sam_rho,
    is_update=True,
    model_client=None,
    _input=None,
    _target=None,
    _msk=None,
    label_smooth=0.0
):
    device = next(model_server.parameters()).device
    
    def compute_asam_perturbation(params, rho, xi=0.01):
        perturbations = []
        for p in params:
            if p.grad is None:
                perturbations.append(None)
                continue

            if p.dim() == 4:
                outC = p.size(0)
                w_reshaped = p.view(outC, -1)
                norms = torch.norm(w_reshaped, dim=1, keepdim=True)  # [outC,1]
                T_theta = norms.view(outC, 1, 1, 1) + xi
            elif p.dim() == 2:
                outC = p.size(0)
                w_reshaped = p.view(outC, -1)
                norms = torch.norm(w_reshaped, dim=1, keepdim=True)  # [outC,1]
                T_theta = norms + xi
            else:
                T_theta = torch.abs(p) + xi
            grad_norm = torch.norm(p.grad)
            epsilon = (T_theta * p.grad) / (grad_norm + 1e-12)
            perturbations.append(rho * epsilon)
        return perturbations


    if args.fp16:
        with amp.scale_loss(_loss, optimizer_server) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    all_params_server = [p for p in model_server.parameters() if p.requires_grad]
    all_params_client = [p for p in model_client.parameters() if p.requires_grad]

    original_params_server = [p.data.clone() for p in all_params_server]
    original_params_client = [p.data.clone() for p in all_params_client]

    server_perturbs = compute_asam_perturbation(all_params_server, sam_rho, xi=0.01)
    client_perturbs = compute_asam_perturbation(all_params_client, sam_rho, xi=0.01)

    for p, pert in zip(all_params_server, server_perturbs):
        if pert is not None:
            p.data.add_(pert)
    for p, pert in zip(all_params_client, client_perturbs):
        if pert is not None:
            p.data.add_(pert)
 
    optimizer_server.zero_grad()
    optimizer_client.zero_grad()
    
    hidden_states_sam, presents_sam, _ = model_client(_input)
    _, _loss_sam = model_server(
        _input.shape,
        hidden_states_sam,
        presents_sam,
        lm_labels=_target,
        lm_mask=_msk,
        label_smooth=label_smooth,
    )
    _loss_sam = _loss_sam.mean()
    
    if args.fp16:
        with amp.scale_loss(_loss_sam, optimizer_server) as _scaled_loss_sam:
            _scaled_loss_sam.backward()
    else:
        _loss_sam.backward()

    for p, orig_p in zip(all_params_server, original_params_server):
        p.data.copy_(orig_p)
    for p, orig_p in zip(all_params_client, original_params_client):
        p.data.copy_(orig_p)

    grads_server = []
    for p in all_params_server:
        if p.grad is None:
            grads_server.append(None)
        else:
            grads_server.append(p.grad.data.clone())

    grads_client = []
    for p in all_params_client:
        if p.grad is None:
            grads_client.append(None)
        else:
            grads_client.append(p.grad.data.clone())

    if is_update and args.clip > 0:
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_server), args.clip)
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_client), args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model_server.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(model_client.parameters(), args.clip)

    optimizer_server.step()
    optimizer_client.step()
    optimizer_server.zero_grad()
    optimizer_client.zero_grad()

    if _schedule is not None:
        _schedule.step()

    return _loss_sam.item(), grads_server, grads_client 

def compute_val_gradients(model_client, model_server, val_data, args):
    device = next(model_server.parameters()).device
    model_client.eval()
    model_server.eval()

    _input = val_data["input"].to(device)
    _target = val_data["target"].to(device)
    _msk   = val_data["mask"].to(device)

    for p in model_client.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in model_server.parameters():
        if p.grad is not None:
            p.grad.zero_()

    with torch.enable_grad():
        hidden_states, presents, _ = model_client(_input)
        _, val_loss = model_server(
            _input.shape,
            hidden_states,
            presents,
            lm_labels=_target,
            lm_mask=_msk,
            label_smooth=args.label_smooth,
        )
        val_loss = val_loss.mean()

    # backward
    val_loss.backward()

    grads_server = []
    for p in model_server.parameters():
        if p.requires_grad:
            grads_server.append(None if p.grad is None else p.grad.data.clone())
    grads_client = []
    for p in model_client.parameters():
        if p.requires_grad:
            grads_client.append(None if p.grad is None else p.grad.data.clone())

    return val_loss.item(), grads_server, grads_client


def approximate_hessian(grads):
    diag = []
    for g in grads:
        if g is None:
            diag.append(None)
        else:
            diag.append(g*g)
    return diag

def update_rho_lets(
    sam_rho,
    train_loss, val_loss,
    train_grads_server, train_grads_client,
    val_grads_server,   val_grads_client,
    args
):

    gap = (val_loss - train_loss)

    hess_train_server = approximate_hessian(train_grads_server)
    hess_train_client = approximate_hessian(train_grads_client)

    grad_rho = 0.0

    for gv, gt, ht in zip(val_grads_server, train_grads_server, hess_train_server):
        if gv is not None and gt is not None and ht is not None:
            grad_rho += torch.sum((gv - gt) * (ht * gt)).item()
    for gv, gt, ht in zip(val_grads_client, train_grads_client, hess_train_client):
        if gv is not None and gt is not None and ht is not None:
            grad_rho += torch.sum((gv - gt) * (ht * gt)).item()

    total_grad_rho = gap * grad_rho
    new_rho = sam_rho - args.beta * total_grad_rho

    new_rho = max(min(new_rho, args.rho_max), args.rho_min)
    return new_rho

def evaluate(model_client, model_server, valid_loader,args):
    model_client.eval()
    model_server.eval()
    device = torch.device("cuda")
    model_server = model_server.to(device)

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value.to(device) for key, value in data.items()}

            _input = data["input"]
            _target = data["target"]
            _msk = data["mask"]

            hidden_states, presents, _ = model_client(_input)

            _, _loss = model_server(
                _input.shape, hidden_states, presents, lm_labels=_target, lm_mask=_msk
            )
            loss = _loss.mean()

            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print("eval samples:", idx, "loss:", loss.float())

        print("average loss", avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model_client,
    model_server,
    client_models,
    optimizers,
    optimizer_server,
    scheduler_server,
    train_loader0,
    train_loader1,
    train_loader2,
    valid_loader,
    args,
    train_step=0,
    epoch=0,
    sam_rho = None,
    prev_validation_loss = float('inf')  
):
    global log_list
    
    model_client.train()
    model_server.train()
    avg_lm_loss = AverageMeter()
    num_clients = len(client_models)
    print("start to train the model................", epoch)
    
    best_val_ppl = None
    device = torch.device("cuda")
    train_loader0.sampler.set_epoch(epoch)

    client_rhos = [sam_rho] * len(client_models) 
    net_glob_client = GPT2LMModel_Client(config)
    net_glob_client = net_glob_client.to(device)
    net_glob_client.load_weight(state_dict)
    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(net_glob_client)
    net_glob_client.train()
    w_glob_client = net_glob_client.state_dict()
    aggregate_step = 100
    w_locals_client = []

    val_iter = iter(valid_loader)

    log_start_time = time.time()

    for idx, data in enumerate(zip(train_loader0, train_loader1, train_loader2)):
        for i in range(num_clients):
            client_data = {key: value.to(device) for key, value in data[i].items()}

            _input = client_data["input"]
            _target = client_data["target"]
            _msk = client_data["mask"]

            client_models[i].train()
            _input = _input.to(device)

            hidden_states, presents, w_client = client_models[i](_input)
            train_step += 1

            if (train_step + num_clients) % aggregate_step <= num_clients:
                w_locals_client.append(copy.deepcopy(w_client))

            client_hidden_states = hidden_states.clone().detach().requires_grad_(True)

            _, first_loss = model_server(
                _input.shape,
                client_hidden_states,
                presents,
                lm_labels=_target,
                lm_mask=_msk,
                label_smooth=args.label_smooth,
            )
            first_loss = first_loss.mean()

            is_update = train_step % args.grad_acc == 0
            
            effective_rho = 0.0 if train_step < 500 else sam_rho
            final_train_loss, grads_server_train, grads_client_train = optimizer_step(
                _loss = first_loss,
                optimizer_server = optimizer_server,
                model_server = model_server,
                optimizer_client = optimizers[i],
                _schedule = scheduler_server,
                client_hidden_states = client_hidden_states,
                hidden_states = hidden_states,
                args = args,
                sam_rho = effective_rho,
                is_update=is_update,
                model_client=client_models[i],
                _input=_input,
                _target=_target,
                _msk=_msk,
                label_smooth=args.label_smooth
            )
            avg_lm_loss.update(final_train_loss)
            

            if (train_step + num_clients) % aggregate_step <= num_clients:
                w_locals_client.append(copy.deepcopy(w_client))

            try:
                val_data = next(val_iter)
            except StopIteration:
                val_iter = iter(valid_loader)
                val_data = next(val_iter)

            val_loss, grads_server_val, grads_client_val = compute_val_gradients(
                client_models[i], model_server, val_data, args
            )
            if train_step >= 0:
                new_rho = update_rho_lets(
                    sam_rho=sam_rho,
                    train_loss=final_train_loss,
                    val_loss=val_loss,
                    train_grads_server=grads_server_train,
                    train_grads_client=grads_client_train,
                    val_grads_server=grads_server_val,
                    val_grads_client=grads_client_val,
                    args=args
                )
                sam_rho = new_rho
                client_rhos[i] = new_rho


            if train_step % aggregate_step == 0:
                global_rho = sum(client_rhos) / len(client_rhos)
                client_rhos = [global_rho] * num_clients
                sam_rho = global_rho

                temp_dict = {}
                w_locals_client_lora = []
                for w_client_t in w_locals_client:
                    for key, value in w_client_t.items():
                        if key.endswith("lora_A") or key.endswith("lora_B"):
                            temp_dict[key] = value
                    w_locals_client_lora.append(copy.deepcopy(temp_dict))

                w_glob_client_lora = fed_avg(w_locals_client_lora)
                w_glob_client_lora_new = {}

                for key, value in w_glob_client_lora.items():
                    new_key = "transformer_Client." + key
                    w_glob_client_lora_new[new_key] = value

                for key, value in w_glob_client.items():
                    if key.endswith("lora_A") or key.endswith("lora_B"):
                        w_glob_client[key] = w_glob_client_lora_new[key]

                net_glob_client.load_state_dict(w_glob_client)
                for client_model in client_models:
                    client_model.load_state_dict(w_glob_client)
                w_locals_client = []


            # ----------- log output -----------
            if train_step % args.log_interval == 0:
                elapsed = time.time() - log_start_time
                lr = optimizer_server.param_groups[0]["lr"]
                log_str = (
                    f"| epoch {epoch:3d} step {train_step:>8d} | {idx*3 + i + 1:>6d} batches | "
                    f"lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | "
                    f"loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | "
                    f"ppl {math.exp(avg_lm_loss.avg):5.2f} | "
                    f"rho {sam_rho:.6f}"
                )

                log_list.append(log_str)
                if args.rank == 0:
                    print(log_str)
                log_start_time = time.time()
                avg_lm_loss.reset()

            # ----------- save -----------
            if train_step % args.save_interval == 0:
                save_checkpoint(
                    w_glob_client, model_server, args, train_step, num_clients
                )
            distributed_sync(args)

            # ----------- eval -----------
            if train_step % args.eval_interval == 0:
                eval_start_time = time.time()

                valid_loss, valid_ppl = evaluate(
                    net_glob_client, model_server, valid_loader, args
                )
                
                train_loss = avg_lm_loss.avg
                prev_validation_loss = valid_loss
                
                if best_val_ppl is None or valid_ppl < best_val_ppl:
                    best_val_ppl = valid_ppl

                log_str = (
                    f"| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | "
                    f"time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | "
                    f"valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} "
                )
                log_list.append(log_str)

                if args.rank == 0:
                    print("-" * 100)
                    print(log_str)
                    print("-" * 100)

                net_glob_client.train()
                model_server.train()
                distributed_sync(args)

            if train_step == args.max_step:
                df = pd.DataFrame(log_list, columns=["Log"])
                df.to_excel(
                    f"LETS_{args.model_card} rank={args.lora_dim} num={num_clients} block=3 seed={args.random_seed} rho={args.sam_rho}.xlsx",
                    sheet_name="Sheet1",
                    index=False,
                )
                break

    if train_step == args.max_step:
        save_checkpoint(w_glob_client, model_server, args, train_step, num_clients)
    distributed_sync(args)
    return train_step, sam_rho


if __name__ == "__main__":
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)

    if args.fp16:
        try:
            from apex import amp
        except Exception:
            warnings.warn("Could not import amp, apex may not be installed")

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    train_data0 = FT_Dataset(
        args.train_data0,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )
    train_data1 = FT_Dataset(
        args.train_data1,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )
    train_data2 = FT_Dataset(
        args.train_data2,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )
    valid_data = FT_Dataset(
        args.valid_data,
        args.valid_batch_size,
        args.seq_len,
    )

    train_loader0 = DataLoader(
        train_data0,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data0, seed=args.random_seed, shuffle=True
        ),
    )
    train_loader1 = DataLoader(
        train_data1,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data1, seed=args.random_seed, shuffle=True
        ),
    )
    train_loader2 = DataLoader(
        train_data2,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data2, seed=args.random_seed, shuffle=True
        ),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.valid_batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            valid_data, seed=args.random_seed
        ),
    )

    if args.model_card == "gpt2.sm":
        config = GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == "gpt2.md":
        config = GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == "gpt2.lg":
        config = GPT2Config(
            n_embd=1280,
            n_layer=36,
            n_head=20,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    lm_net_Client = GPT2LMModel_Client(config)
    lm_net_Server = GPT2LMModel_Server(config)

    state_dict = torch.load(args.init_checkpoint)
    if args.init_checkpoint is not None:
        print("loading model pretrained weight.")
        lm_net_Client.load_weight(state_dict)
        lm_net_Server.load_weight(state_dict)

    lm_net_Client = lm_net_Client.cuda()
    lm_net_Server = lm_net_Server.cuda()

    if args.lora_dim > 0:
        lora.mark_only_lora_as_trainable(lm_net_Client)
        lora.mark_only_lora_as_trainable(lm_net_Server)

    optimizer_Client = create_adam_optimizer_from_args(lm_net_Client, args)
    optimizer_Server = create_adam_optimizer_from_args(lm_net_Server, args)

    num_clients = 3
    client_models = []
    optimizers = []
    for i in range(num_clients):
        client_model = GPT2LMModel_Client(config)
        client_model.load_weight(state_dict)
        client_model = client_model.cuda()
        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(client_model)
        optimizer = create_adam_optimizer_from_args(client_model, args)
        client_models.append(client_model)
        optimizers.append(optimizer)

    if args.max_step is None:
        args.max_step = (
            args.max_epoch * train_data0.num_batches * 3 + args.world_size - 1
        ) // args.world_size
        print("set max_step:", args.max_step)

    scheduler_Client = create_optimizer_scheduler(optimizer_Client, args)
    scheduler_Server = create_optimizer_scheduler(optimizer_Server, args)

    if args.fp16:
        lm_net_Client, optimizer_Client = amp.initialize(
            lm_net_Client, optimizer_Client, opt_level="O1"
        )
        lm_net_Server, optimizer_Server = amp.initialize(
            lm_net_Server, optimizer_Server, opt_level="O1"
        )

    lm_net_Client, optimizer_Client = distributed_opt(
        args, lm_net_Client, optimizer_Client, grad_acc=args.grad_acc
    )
    lm_net_Server, optimizer_Server = distributed_opt(
        args, lm_net_Server, optimizer_Server, grad_acc=args.grad_acc
    )

    log_list = []

    try:
        train_step = 0
        sam_rho = args.sam_rho
        for epoch in itertools.count(start=1):
            train_step,sam_rho= train_validate(
                model_client=lm_net_Client,
                model_server=lm_net_Server,
                client_models=client_models,
                optimizers=optimizers,
                optimizer_server=optimizer_Server,
                scheduler_server=scheduler_Server,
                train_loader0=train_loader0,
                train_loader1=train_loader1,
                train_loader2=train_loader2,
                valid_loader=valid_loader,
                args=args,
                train_step=train_step,
                epoch=epoch,
                sam_rho=sam_rho
            )
            if train_step >= args.max_step or (
                args.max_epoch is not None and epoch >= args.max_epoch
            ):
                if args.rank == 0:
                    print("-" * 100)
                    print("End of training")
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print("-" * 100)
            print("Exiting from training early")

    distributed_sync(args)
    print("cleanup dist ...")
    cleanup(args)
