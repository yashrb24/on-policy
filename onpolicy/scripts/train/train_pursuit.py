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
from onpolicy.envs.pursuit.pursuit_env import PursuitEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Pursuit":
                env = PursuitEnv(all_args)
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
            if all_args.env_name == "Pursuit":
                env = PursuitEnv(all_args)
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
    parser.add_argument('--n_pursuers', type=int, default=8,
                        help="Number of pursuer agents controlled by MAPPO")
    parser.add_argument('--n_evaders', type=int, default=30,
                        help="Number of evader entities (env-controlled)")
    parser.add_argument('--x_size', type=int, default=16,
                        help="Pursuit grid width")
    parser.add_argument('--y_size', type=int, default=16,
                        help="Pursuit grid height")
    parser.add_argument('--max_cycles', type=int, default=500,
                        help="Max cycles per episode in pursuit_v4 (align with --episode_length)")
    parser.add_argument("--eval_deterministic", action="store_false",
                        default=True,
                        help="by default True. If False, sample action according to probability")

    all_args = parser.parse_known_args(args)[0]

    # MAPPO framework reads num_agents; mirror n_pursuers into it.
    all_args.num_agents = all_args.n_pursuers

    # Ensure n_embd and hidden_size are equal (transformer base requires this).
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
        if wandb.run is not None:
            print(f"Detected WandB sweep run: {wandb.run.id}")
            wandb.config.update(vars(all_args), allow_val_change=True)
            wandb.run.tags = all_args.wandb_tags
            run = wandb.run
        else:
            run = wandb.init(config=all_args,
                             project=all_args.wandb_name if all_args.wandb_name else all_args.env_name,
                             entity=all_args.user_name,
                             notes=socket.gethostname(),
                             name="-".join([
                                 all_args.algorithm_name,
                                 all_args.experiment_name,
                                 "seed" + str(all_args.seed)
                             ]),
                             group=f"{all_args.n_pursuers}P-{all_args.n_evaders}E",
                             dir=str(run_dir),
                             job_type="training",
                             tags=all_args.wandb_tags,
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
    ]))

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
        from onpolicy.runner.shared.pursuit_runner import PursuitRunner as Runner
    else:
        raise NotImplementedError("Not support yet!")

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        if wandb.run and wandb.run.sweep_id is None:
            run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
