"""
Equivalence + benchmark harness for the optimized TrafficJunction and PredatorPrey envs.

Usage:
    python -m bench.test_equiv_and_bench --mode equiv
    python -m bench.test_equiv_and_bench --mode bench
    python -m bench.test_equiv_and_bench --mode profile

- equiv   : asserts bit-exact obs/reward/done/info between orig and new across seeds x episodes.
- bench   : reports steps/sec for orig vs new on TJ medium, TJ hard, PP hard.
- profile : cProfile top-20 by cumulative time for each (env, version) pair.

The frozen originals live in bench/{traffic_junction_orig,predator_prey_orig}.py
and the patched live envs are imported from onpolicy.envs.{traffic_junction,predator_prey}.
"""
import argparse
import cProfile
import io
import pstats
import sys
import time
from argparse import Namespace
from statistics import median

import numpy as np

from onpolicy.envs.traffic_junction.TrafficJunction_Env import TrafficJunctionEnv as TJ_New
from onpolicy.envs.predator_prey.PredatorPrey_env import PredatorPreyEnv as PP_New
from bench.traffic_junction_orig import TrafficJunctionEnv as TJ_Orig
from bench.predator_prey_orig import PredatorPreyEnv as PP_Orig


# ---------- config builders ----------

def tj_args(difficulty, dim, vision, nagents, episode_length, seed, vocab_type='bool',
            add_rate_min=0.1, add_rate_max=0.1):
    return Namespace(
        dim=dim, vision=vision,
        add_rate_min=add_rate_min, add_rate_max=add_rate_max,
        curr_start=0, curr_end=0,
        difficulty=difficulty, vocab_type=vocab_type,
        nagents=nagents, episode_length=episode_length, seed=seed,
    )


def pp_args(dim, vision, num_agents, nenemies=1, mode='cooperative',
            enemy_comm=False, moving_prey=False, no_stay=False):
    return Namespace(
        dim=dim, vision=vision, num_agents=num_agents, nenemies=nenemies,
        mode=mode, enemy_comm=enemy_comm, moving_prey=moving_prey, no_stay=no_stay,
    )


CONFIGS = {
    'TJ medium': {
        'kind': 'tj',
        'args': dict(difficulty='medium', dim=14, vision=1, nagents=10, episode_length=40),
        'naction': 2,
        'nagents_for_action': 10,
    },
    'TJ hard': {
        'kind': 'tj',
        'args': dict(difficulty='hard', dim=18, vision=0, nagents=20, episode_length=80),
        'naction': 2,
        'nagents_for_action': 20,
    },
    'PP hard': {
        'kind': 'pp',
        'args': dict(dim=20, vision=1, num_agents=10, nenemies=1, mode='cooperative'),
        'naction': 5,
        'nagents_for_action': 10,
        'episode_length': 80,
    },
}


def build_env(cfg, version, seed):
    """Build an env for the given config + version ('orig'|'new') seeded reproducibly.

    PP's _multi_agent_init calls reset() once internally (before np_random is set), which falls back
    to global np.random. We seed global np.random before construction so both orig and new get the
    same initial-reset trajectory before the first explicit reset() from the test/bench.
    """
    np.random.seed(seed)
    if cfg['kind'] == 'tj':
        cls = TJ_Orig if version == 'orig' else TJ_New
        env = cls(tj_args(seed=seed, **cfg['args']))
    else:
        cls = PP_Orig if version == 'orig' else PP_New
        env = cls(pp_args(**cfg['args']))
    env.seed(seed)
    return env


# ---------- equivalence ----------

def _to_array(obs):
    """TJ returns a tuple of 1d arrays; PP returns a 2d ndarray. Normalize both to 2d ndarray."""
    if isinstance(obs, tuple):
        return np.stack(obs)
    return np.asarray(obs)


def equiv_one(name, cfg, seed, n_episodes, action_rng):
    env_o = build_env(cfg, 'orig', seed)
    env_n = build_env(cfg, 'new', seed)

    # Both envs have run their internal init-time reset by now. Force an explicit reset for a clean start
    # after np_random has been seeded, and use that as the reference point.
    np.random.seed(seed)
    o_o = env_o.reset()
    np.random.seed(seed)
    o_n = env_n.reset()
    if not np.array_equal(_to_array(o_o), _to_array(o_n)):
        raise AssertionError(f'{name} seed={seed}: initial-reset obs mismatch')

    steps_per_ep = cfg['args'].get('episode_length') or cfg.get('episode_length') or 80
    naction = cfg['naction']
    nagents = cfg['nagents_for_action']

    total_steps = 0
    for ep in range(n_episodes):
        for t in range(steps_per_ep):
            a = action_rng.randint(0, naction, size=nagents)
            oo, ro, do, info_o = env_o.step(a)
            on, rn, dn, info_n = env_n.step(a)
            if not np.array_equal(_to_array(oo), _to_array(on)):
                diff = (_to_array(oo) != _to_array(on))
                raise AssertionError(f'{name} seed={seed} ep={ep} step={t}: obs diff in {diff.sum()} entries')
            if not np.array_equal(np.asarray(ro), np.asarray(rn)):
                raise AssertionError(f'{name} seed={seed} ep={ep} step={t}: reward mismatch')
            if not np.array_equal(np.asarray(do), np.asarray(dn)):
                raise AssertionError(f'{name} seed={seed} ep={ep} step={t}: done mismatch')
            # TJ-specific info fields that matter.
            if cfg['kind'] == 'tj':
                if info_o['has_failed'] != info_n['has_failed']:
                    raise AssertionError(f'{name} seed={seed} ep={ep} step={t}: has_failed mismatch')
                if info_o.get('success') != info_n.get('success'):
                    raise AssertionError(f'{name} seed={seed} ep={ep} step={t}: success mismatch')
            total_steps += 1
            if bool(np.all(do)):
                break
        # reset for next episode; both envs reseed internally via self.np_random state — no reseed here,
        # we want the episode-level RNG stream to continue.
        if ep < n_episodes - 1:
            o_o = env_o.reset()
            o_n = env_n.reset()
            if not np.array_equal(_to_array(o_o), _to_array(o_n)):
                raise AssertionError(f'{name} seed={seed} ep={ep+1}: reset obs mismatch')
    return total_steps


def run_equiv(seeds=(0, 1, 2, 3, 4), n_episodes=10):
    print(f'=== Equivalence (seeds={list(seeds)}, {n_episodes} episodes each) ===')
    all_pass = True
    for name, cfg in CONFIGS.items():
        for seed in seeds:
            action_rng = np.random.RandomState(seed * 1009 + 7)
            try:
                steps = equiv_one(name, cfg, seed, n_episodes, action_rng)
                print(f'  {name:<10s} seed={seed} ep=0-{n_episodes-1:<2d}  ✓ bit-exact ({steps} steps)')
            except AssertionError as e:
                all_pass = False
                print(f'  {name:<10s} seed={seed}  ✗ {e}')
    if all_pass:
        print('All equivalence checks passed.')
    else:
        print('FAILED — see above.')
        sys.exit(1)


# ---------- benchmark ----------

def _rollout_steps(env, n_steps, action_rng, naction, nagents, episode_length):
    """Run n_steps steps of env, resetting on done or episode_length boundary. Returns elapsed seconds."""
    env.reset()
    t_in_ep = 0
    t0 = time.perf_counter()
    for _ in range(n_steps):
        a = action_rng.randint(0, naction, size=nagents)
        o, r, d, info = env.step(a)
        t_in_ep += 1
        if bool(np.all(d)) or t_in_ep >= episode_length:
            env.reset()
            t_in_ep = 0
    t1 = time.perf_counter()
    return t1 - t0


def run_bench(n_steps=8000, repeats=3, seed=0):
    print(f'=== Benchmark ({n_steps} steps x {repeats} repeats, median reported) ===')
    header = f'{"config":<10s} {"orig steps/s":>14s} {"new steps/s":>14s} {"speedup":>10s}'
    print(header)
    print('-' * len(header))
    results = []
    for name, cfg in CONFIGS.items():
        naction = cfg['naction']
        nagents = cfg['nagents_for_action']
        ep_len = cfg['args'].get('episode_length') or cfg.get('episode_length') or 80

        def run_one(version):
            times = []
            for rep in range(repeats):
                env = build_env(cfg, version, seed + rep)
                action_rng = np.random.RandomState((seed + rep) * 1009 + 7)
                dt = _rollout_steps(env, n_steps, action_rng, naction, nagents, ep_len)
                times.append(dt)
            return median(times)

        dt_o = run_one('orig')
        dt_n = run_one('new')
        sps_o = n_steps / dt_o
        sps_n = n_steps / dt_n
        speedup = sps_n / sps_o
        print(f'{name:<10s} {sps_o:>14.1f} {sps_n:>14.1f} {speedup:>9.2f}x')
        results.append((name, sps_o, sps_n, speedup))
    return results


# ---------- profile ----------

def run_profile(n_steps=2000, seed=0, top=20):
    print(f'=== cProfile top-{top} (cumulative) on {n_steps} steps ===')
    for name, cfg in CONFIGS.items():
        naction = cfg['naction']
        nagents = cfg['nagents_for_action']
        ep_len = cfg['args'].get('episode_length') or cfg.get('episode_length') or 80
        for version in ('orig', 'new'):
            env = build_env(cfg, version, seed)
            action_rng = np.random.RandomState(seed * 1009 + 7)
            pr = cProfile.Profile()
            pr.enable()
            _rollout_steps(env, n_steps, action_rng, naction, nagents, ep_len)
            pr.disable()
            buf = io.StringIO()
            ps = pstats.Stats(pr, stream=buf).sort_stats('cumulative')
            ps.print_stats(top)
            print(f'\n--- {name} [{version}] ---')
            # Trim to just the top lines (skip pstats preamble for brevity).
            out = buf.getvalue().splitlines()
            # Find where the column header starts.
            header_idx = next((i for i, l in enumerate(out) if l.strip().startswith('ncalls')), 0)
            print('\n'.join(out[header_idx:header_idx + top + 2]))


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['equiv', 'bench', 'profile', 'all'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--bench-steps', type=int, default=8000)
    parser.add_argument('--bench-repeats', type=int, default=3)
    parser.add_argument('--profile-steps', type=int, default=2000)
    args = parser.parse_args()

    if args.mode in ('equiv', 'all'):
        run_equiv(seeds=tuple(args.seeds), n_episodes=args.episodes)
    if args.mode in ('bench', 'all'):
        print()
        run_bench(n_steps=args.bench_steps, repeats=args.bench_repeats, seed=args.seeds[0])
    if args.mode in ('profile', 'all'):
        print()
        run_profile(n_steps=args.profile_steps, seed=args.seeds[0])


if __name__ == '__main__':
    main()
