#!/usr/bin/env python
# python standard libraries
import os
import socket
import sys
from pathlib import Path

# third-party packages
import numpy as np
import setproctitle
import torch
import wandb

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.traffic_junction.TrafficJunction_Env import TrafficJunctionEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "TrafficJunction":
                env = TrafficJunctionEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "TrafficJunction":
                env = TrafficJunctionEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    # Traffic Junction environment specific arguments

    parser.add_argument('--num_agents', type=int, default=3,
                        help="Number of cars/agents in the environment")
    parser.add_argument('--dim', type=int, default=6,
                        help="Dimension of box (i.e length of road). Must be even for easy/medium difficulty")
    parser.add_argument('--vision', type=int, default=1,
                        help="Vision of car")
    parser.add_argument('--add_rate_min', type=float, default=0.05,
                        help="Rate at which to add car (till curr. start)")
    parser.add_argument('--add_rate_max', type=float, default=0.2,
                        help="Max rate at which to add car")
    parser.add_argument('--curr_start', type=float, default=0,
                        help="Start making harder after this many epochs [0]")
    parser.add_argument('--curr_end', type=float, default=0,
                        help="When to make the game hardest [0]")
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard'],
                        help="Difficulty level: easy|medium|hard")
    parser.add_argument('--vocab_type', type=str, default='bool',
                        choices=['bool', 'scalar'],
                        help="Type of location vector to use: bool|scalar")
    parser.add_argument("--eval_deterministic", action="store_false",
                        default=True,
                        help="by default True. If False, sample action according to probability")

    all_args = parser.parse_known_args(args)[0]

    # Set nagents to match num_agents for consistency with TrafficJunctionEnv
    all_args.nagents = all_args.num_agents

    # Ensure n_embd and hidden_size are equal
    if hasattr(all_args, 'n_embd') and hasattr(all_args, 'hidden_size'):
        if all_args.n_embd != all_args.hidden_size:
            print(f"!!!!!!!!! WARNING !!!!!!!!!")
            print(f"n_embd ({all_args.n_embd}) and hidden_size ({all_args.hidden_size}) are different!")
            print(f"Setting both to n_embd value ({all_args.n_embd}) for consistency.")
            print(f"!!!!!!!!! WARNING !!!!!!!!!")
            all_args.hidden_size = all_args.n_embd

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    # Print all arguments
    print("=" * 80)
    print("TRAINING CONFIGURATION ARGUMENTS")
    print("=" * 80)
    args_dict = vars(all_args)
    for key in sorted(args_dict.keys()):
        value = args_dict[key]
        print(f"{key:<40}: {value}")
    print("=" * 80)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name="-".join([
                             all_args.algorithm_name,
                             all_args.experiment_name,
                             "seed" + str(all_args.seed)
                         ]),
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle("-".join([
        all_args.env_name,
        all_args.algorithm_name,
        all_args.experiment_name
    ]) + "@" + all_args.user_name)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    if all_args.share_policy:
        from onpolicy.runner.shared.traffic_junction_runner import TrafficJunctionRunner as Runner
    else:
        raise NotImplementedError("Not support yet!")

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
