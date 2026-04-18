"""
Equivalence + benchmark harness for the R-MAPPO training-side optimizations.

Usage:
    python -m bench.test_train_equiv_and_bench --mode equiv
    python -m bench.test_train_equiv_and_bench --mode bench
    python -m bench.test_train_equiv_and_bench --mode all

What this verifies (bit-exact across all cases):
    - SharedReplayBuffer.insert() / after_update()
    - SeparatedReplayBuffer.insert() / after_update()
    - ValueNorm.normalize() / denormalize()
    - PopArt.normalize() / denormalize()
    - compute_returns (GAE paths, with and without popart/valuenorm)
    - np.broadcast_to(expand_dims(...), ...) stored into a contiguous slot vs
      np.repeat(expand_dims(...), ...)
    - np.array(np.split(x, N)) == x.reshape(N, -1, *tail)
    - Feed-forward minibatch generator output after a full insert+compute cycle.

Every check compares the frozen pre-change module (bench/training_orig/) against
the live patched module. Failures print the step index, field, and diff summary.
"""
import argparse
import sys
import time
from argparse import Namespace
from statistics import median

import numpy as np
import torch

# Live (patched) modules
from onpolicy.utils.shared_buffer import SharedReplayBuffer as NewSharedBuffer
from onpolicy.utils.separated_buffer import SeparatedReplayBuffer as NewSeparatedBuffer
from onpolicy.utils.valuenorm import ValueNorm as NewValueNorm
from onpolicy.algorithms.utils.popart import PopArt as NewPopArt

# Frozen (original) modules
from bench.training_orig.shared_buffer import SharedReplayBuffer as OrigSharedBuffer
from bench.training_orig.separated_buffer import SeparatedReplayBuffer as OrigSeparatedBuffer
from bench.training_orig.valuenorm import ValueNorm as OrigValueNorm
from bench.training_orig.popart import PopArt as OrigPopArt


# ---------- configuration builders ----------

def make_buffer_args(episode_length=16, n_rollout_threads=4, hidden_size=32,
                     recurrent_N=1, gamma=0.99, gae_lambda=0.95, use_gae=True,
                     use_popart=False, use_valuenorm=False, use_proper_time_limits=False,
                     algorithm_name='rmappo'):
    return Namespace(
        episode_length=episode_length,
        n_rollout_threads=n_rollout_threads,
        hidden_size=hidden_size,
        recurrent_N=recurrent_N,
        gamma=gamma,
        gae_lambda=gae_lambda,
        use_gae=use_gae,
        use_popart=use_popart,
        use_valuenorm=use_valuenorm,
        use_proper_time_limits=use_proper_time_limits,
        algorithm_name=algorithm_name,
    )


class BoxSpace:
    """Minimal gym.spaces.Box stand-in (avoids importing gym for unit tests)."""
    def __init__(self, shape):
        self.shape = shape

    @property
    def __class__(self):  # pragma: no cover — class-check hack for buffer code
        return _BoxProxy


class _BoxProxy:
    __name__ = 'Box'


class DiscreteSpace:
    def __init__(self, n):
        self.n = n

    @property
    def __class__(self):  # pragma: no cover
        return _DiscreteProxy


class _DiscreteProxy:
    __name__ = 'Discrete'


def _make_spaces(obs_dim=8, share_obs_dim=16, n_actions=5):
    # SharedReplayBuffer uses get_shape_from_obs_space which handles gym.Box / list shapes.
    # We mimic gym.Box via a simple object with `shape`.
    class _Box:
        def __init__(self, shape):
            self.shape = shape
            self.__class__.__name__ = 'Box'
    class _Discrete:
        def __init__(self, n):
            self.n = n
            self.__class__.__name__ = 'Discrete'
    return _Box((obs_dim,)), _Box((share_obs_dim,)), _Discrete(n_actions)


# ---------- helpers ----------

def _check_arrays_equal(label, a, b):
    if a is None and b is None:
        return
    if (a is None) != (b is None):
        raise AssertionError(f'{label}: one side is None, other is not')
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f'{label}: shape mismatch {a.shape} vs {b.shape}')
    if a.dtype != b.dtype:
        raise AssertionError(f'{label}: dtype mismatch {a.dtype} vs {b.dtype}')
    if not np.array_equal(a, b):
        diff = (a != b).sum()
        raise AssertionError(f'{label}: {diff} element mismatch (shape={a.shape})')


def _diff_buffer(orig, new):
    fields = ['share_obs', 'obs', 'rnn_states', 'rnn_states_critic',
              'actions', 'action_log_probs', 'value_preds', 'rewards',
              'masks', 'bad_masks', 'active_masks', 'available_actions',
              'returns', 'advantages']
    for f in fields:
        if not hasattr(orig, f) and not hasattr(new, f):
            continue
        a = getattr(orig, f, None)
        b = getattr(new, f, None)
        _check_arrays_equal(f'buffer.{f}', a, b)


# ---------- tests ----------

def test_shared_buffer_insert_and_after_update(num_agents=3, seed=0):
    args = make_buffer_args()
    obs_space, share_obs_space, act_space = _make_spaces()
    orig = OrigSharedBuffer(args, num_agents, obs_space, share_obs_space, act_space)
    new = NewSharedBuffer(args, num_agents, obs_space, share_obs_space, act_space)

    # Starting state must already match (both zero-initialized).
    _diff_buffer(orig, new)

    rng = np.random.RandomState(seed)
    n_steps = args.episode_length
    for step in range(n_steps):
        share_obs = rng.randn(args.n_rollout_threads, num_agents, 16).astype(np.float32)
        obs = rng.randn(args.n_rollout_threads, num_agents, 8).astype(np.float32)
        rnn_a = rng.randn(args.n_rollout_threads, num_agents, args.recurrent_N, args.hidden_size).astype(np.float32)
        rnn_c = rng.randn(args.n_rollout_threads, num_agents, args.recurrent_N, args.hidden_size).astype(np.float32)
        actions = rng.randint(0, 5, size=(args.n_rollout_threads, num_agents, 1)).astype(np.float32)
        alp = rng.randn(args.n_rollout_threads, num_agents, 1).astype(np.float32)
        values = rng.randn(args.n_rollout_threads, num_agents, 1).astype(np.float32)
        rewards = rng.randn(args.n_rollout_threads, num_agents, 1).astype(np.float32)
        masks = rng.randint(0, 2, size=(args.n_rollout_threads, num_agents, 1)).astype(np.float32)
        active_masks = rng.randint(0, 2, size=(args.n_rollout_threads, num_agents, 1)).astype(np.float32)
        avail = rng.randint(0, 2, size=(args.n_rollout_threads, num_agents, 5)).astype(np.float32)

        # Insert identical inputs into both.
        orig.insert(share_obs.copy(), obs.copy(), rnn_a.copy(), rnn_c.copy(),
                    actions.copy(), alp.copy(), values.copy(), rewards.copy(),
                    masks.copy(), active_masks=active_masks.copy(),
                    available_actions=avail.copy())
        new.insert(share_obs, obs, rnn_a, rnn_c,
                   actions, alp, values, rewards,
                   masks, active_masks=active_masks,
                   available_actions=avail)
        _diff_buffer(orig, new)

    # after_update
    orig.after_update()
    new.after_update()
    _diff_buffer(orig, new)
    return n_steps


def test_separated_buffer_insert_and_after_update(seed=0):
    args = make_buffer_args()
    obs_space, share_obs_space, act_space = _make_spaces()
    orig = OrigSeparatedBuffer(args, obs_space, share_obs_space, act_space)
    new = NewSeparatedBuffer(args, obs_space, share_obs_space, act_space)

    rng = np.random.RandomState(seed)
    n_steps = args.episode_length
    for step in range(n_steps):
        share_obs = rng.randn(args.n_rollout_threads, 16).astype(np.float32)
        obs = rng.randn(args.n_rollout_threads, 8).astype(np.float32)
        rnn_a = rng.randn(args.n_rollout_threads, args.recurrent_N, args.hidden_size).astype(np.float32)
        rnn_c = rng.randn(args.n_rollout_threads, args.recurrent_N, args.hidden_size).astype(np.float32)
        actions = rng.randint(0, 5, size=(args.n_rollout_threads, 1)).astype(np.float32)
        alp = rng.randn(args.n_rollout_threads, 1).astype(np.float32)
        values = rng.randn(args.n_rollout_threads, 1).astype(np.float32)
        rewards = rng.randn(args.n_rollout_threads, 1).astype(np.float32)
        masks = rng.randint(0, 2, size=(args.n_rollout_threads, 1)).astype(np.float32)
        active_masks = rng.randint(0, 2, size=(args.n_rollout_threads, 1)).astype(np.float32)
        avail = rng.randint(0, 2, size=(args.n_rollout_threads, 5)).astype(np.float32)

        orig.insert(share_obs.copy(), obs.copy(), rnn_a.copy(), rnn_c.copy(),
                    actions.copy(), alp.copy(), values.copy(), rewards.copy(),
                    masks.copy(), active_masks=active_masks.copy(),
                    available_actions=avail.copy())
        new.insert(share_obs, obs, rnn_a, rnn_c,
                   actions, alp, values, rewards,
                   masks, active_masks=active_masks, available_actions=avail)
        _diff_buffer(orig, new)

    orig.after_update()
    new.after_update()
    _diff_buffer(orig, new)
    return n_steps


def test_valuenorm_bitexact(seed=0):
    torch.manual_seed(seed)
    orig = OrigValueNorm(input_shape=1, norm_axes=1)
    new = NewValueNorm(input_shape=1, norm_axes=1)
    rng = np.random.RandomState(seed)

    # Feed several updates and check normalize/denormalize at each step.
    for i in range(10):
        inp = rng.randn(32, 1).astype(np.float32)
        orig.update(inp)
        new.update(inp)
        # Running stats must match exactly (same param tensors, same ops).
        assert torch.equal(orig.running_mean, new.running_mean), f'step {i}: running_mean diverged'
        assert torch.equal(orig.running_mean_sq, new.running_mean_sq), f'step {i}: running_mean_sq diverged'
        assert torch.equal(orig.debiasing_term, new.debiasing_term), f'step {i}: debiasing_term diverged'

        test_in = rng.randn(8, 1).astype(np.float32)
        o_n = orig.normalize(test_in)
        n_n = new.normalize(test_in)
        assert torch.equal(o_n, n_n), f'step {i}: normalize output diverged'

        o_d = orig.denormalize(test_in)
        n_d = new.denormalize(test_in)
        _check_arrays_equal(f'valuenorm.denormalize step={i}', o_d, n_d)

    # Test higher norm_axes
    orig2 = OrigValueNorm(input_shape=4, norm_axes=2)
    new2 = NewValueNorm(input_shape=4, norm_axes=2)
    for i in range(5):
        inp = rng.randn(3, 5, 4).astype(np.float32)
        orig2.update(inp)
        new2.update(inp)
        t = rng.randn(2, 7, 4).astype(np.float32)
        _check_arrays_equal(f'valuenorm-na2.denormalize step={i}', orig2.denormalize(t), new2.denormalize(t))


def test_popart_bitexact(seed=1):
    torch.manual_seed(seed)
    orig = OrigPopArt(input_shape=8, output_shape=1, norm_axes=1)
    new = NewPopArt(input_shape=8, output_shape=1, norm_axes=1)
    # Copy original's weights into new so forward/normalize/denormalize match.
    with torch.no_grad():
        new.weight.copy_(orig.weight)
        new.bias.copy_(orig.bias)
        new.stddev.copy_(orig.stddev)
        new.mean.copy_(orig.mean)
        new.mean_sq.copy_(orig.mean_sq)
        new.debiasing_term.copy_(orig.debiasing_term)

    rng = np.random.RandomState(seed)
    for i in range(5):
        test_in = rng.randn(6, 1).astype(np.float32)
        o_n = orig.normalize(test_in)
        n_n = new.normalize(test_in)
        assert torch.equal(o_n, n_n), f'popart.normalize step={i}: diverged'
        o_d = orig.denormalize(test_in)
        n_d = new.denormalize(test_in)
        _check_arrays_equal(f'popart.denormalize step={i}', o_d, n_d)


def _drive_insert_and_returns(orig, new, num_agents, rng, next_value):
    """Fill both buffers with identical seeded data, compute_returns, diff."""
    args_el = orig.episode_length
    for step in range(args_el):
        share_obs = rng.randn(orig.n_rollout_threads, num_agents, 16).astype(np.float32)
        obs = rng.randn(orig.n_rollout_threads, num_agents, 8).astype(np.float32)
        rnn_a = rng.randn(orig.n_rollout_threads, num_agents, orig.recurrent_N, orig.hidden_size).astype(np.float32)
        rnn_c = rng.randn(orig.n_rollout_threads, num_agents, orig.recurrent_N, orig.hidden_size).astype(np.float32)
        actions = rng.randint(0, 5, size=(orig.n_rollout_threads, num_agents, 1)).astype(np.float32)
        alp = rng.randn(orig.n_rollout_threads, num_agents, 1).astype(np.float32)
        values = rng.randn(orig.n_rollout_threads, num_agents, 1).astype(np.float32)
        rewards = rng.randn(orig.n_rollout_threads, num_agents, 1).astype(np.float32)
        masks = rng.randint(0, 2, size=(orig.n_rollout_threads, num_agents, 1)).astype(np.float32)
        active_masks = rng.randint(0, 2, size=(orig.n_rollout_threads, num_agents, 1)).astype(np.float32)
        orig.insert(share_obs.copy(), obs.copy(), rnn_a.copy(), rnn_c.copy(),
                    actions.copy(), alp.copy(), values.copy(), rewards.copy(),
                    masks.copy(), active_masks=active_masks.copy())
        new.insert(share_obs, obs, rnn_a, rnn_c,
                   actions, alp, values, rewards,
                   masks, active_masks=active_masks)


def test_compute_returns_equiv(seed=0):
    """Exercise all GAE branches and assert returns/advantages are bit-identical."""
    obs_space, share_obs_space, act_space = _make_spaces()
    configs = [
        dict(use_gae=True, use_proper_time_limits=False, use_popart=False, use_valuenorm=False),
        dict(use_gae=True, use_proper_time_limits=False, use_popart=False, use_valuenorm=True),
        dict(use_gae=True, use_proper_time_limits=True, use_popart=False, use_valuenorm=False),
        dict(use_gae=True, use_proper_time_limits=True, use_popart=False, use_valuenorm=True),
        dict(use_gae=False, use_proper_time_limits=False, use_popart=False, use_valuenorm=False),
        dict(use_gae=False, use_proper_time_limits=True, use_popart=False, use_valuenorm=True),
    ]
    num_agents = 3
    for cfg in configs:
        args = make_buffer_args(**cfg)
        orig = OrigSharedBuffer(args, num_agents, obs_space, share_obs_space, act_space)
        new = NewSharedBuffer(args, num_agents, obs_space, share_obs_space, act_space)

        rng = np.random.RandomState(seed)
        _drive_insert_and_returns(orig, new, num_agents, rng, next_value=None)

        # Seeded value_normalizer (shared params so denormalize is deterministic).
        if cfg['use_valuenorm']:
            torch.manual_seed(seed)
            vn_orig = OrigValueNorm(input_shape=1, norm_axes=1)
            vn_new = NewValueNorm(input_shape=1, norm_axes=1)
            warmup = rng.randn(16, 1).astype(np.float32)
            vn_orig.update(warmup)
            vn_new.update(warmup)
        else:
            vn_orig = vn_new = None

        next_value = rng.randn(args.n_rollout_threads, num_agents, 1).astype(np.float32)
        orig.compute_returns(next_value.copy(), value_normalizer=vn_orig)
        new.compute_returns(next_value.copy(), value_normalizer=vn_new)
        try:
            _check_arrays_equal(f'compute_returns[cfg={cfg}].returns', orig.returns, new.returns)
            _check_arrays_equal(f'compute_returns[cfg={cfg}].advantages', orig.advantages, new.advantages)
        except AssertionError as e:
            raise AssertionError(f'compute_returns mismatch for cfg={cfg}: {e}')

    # Separated buffer GAE variants
    for cfg in configs:
        args = make_buffer_args(**cfg)
        orig_s = OrigSeparatedBuffer(args, obs_space, share_obs_space, act_space)
        new_s = NewSeparatedBuffer(args, obs_space, share_obs_space, act_space)
        rng = np.random.RandomState(seed + 1)
        el = args.episode_length
        for step in range(el):
            orig_s.insert(
                rng.randn(args.n_rollout_threads, 16).astype(np.float32).copy(),
                rng.randn(args.n_rollout_threads, 8).astype(np.float32).copy(),
                rng.randn(args.n_rollout_threads, args.recurrent_N, args.hidden_size).astype(np.float32).copy(),
                rng.randn(args.n_rollout_threads, args.recurrent_N, args.hidden_size).astype(np.float32).copy(),
                rng.randint(0, 5, size=(args.n_rollout_threads, 1)).astype(np.float32).copy(),
                rng.randn(args.n_rollout_threads, 1).astype(np.float32).copy(),
                rng.randn(args.n_rollout_threads, 1).astype(np.float32).copy(),
                rng.randn(args.n_rollout_threads, 1).astype(np.float32).copy(),
                rng.randint(0, 2, size=(args.n_rollout_threads, 1)).astype(np.float32).copy(),
            )
        # Reset rng for the new buffer so we get identical inputs.
        rng2 = np.random.RandomState(seed + 1)
        for step in range(el):
            new_s.insert(
                rng2.randn(args.n_rollout_threads, 16).astype(np.float32),
                rng2.randn(args.n_rollout_threads, 8).astype(np.float32),
                rng2.randn(args.n_rollout_threads, args.recurrent_N, args.hidden_size).astype(np.float32),
                rng2.randn(args.n_rollout_threads, args.recurrent_N, args.hidden_size).astype(np.float32),
                rng2.randint(0, 5, size=(args.n_rollout_threads, 1)).astype(np.float32),
                rng2.randn(args.n_rollout_threads, 1).astype(np.float32),
                rng2.randn(args.n_rollout_threads, 1).astype(np.float32),
                rng2.randn(args.n_rollout_threads, 1).astype(np.float32),
                rng2.randint(0, 2, size=(args.n_rollout_threads, 1)).astype(np.float32),
            )
        if cfg['use_valuenorm']:
            torch.manual_seed(seed + 1)
            vn_o = OrigValueNorm(input_shape=1, norm_axes=1)
            vn_n = NewValueNorm(input_shape=1, norm_axes=1)
            warm = np.random.RandomState(seed + 1).randn(16, 1).astype(np.float32)
            vn_o.update(warm)
            vn_n.update(warm)
        else:
            vn_o = vn_n = None
        nv = np.random.RandomState(seed + 2).randn(args.n_rollout_threads, 1).astype(np.float32)
        orig_s.compute_returns(nv.copy(), value_normalizer=vn_o)
        new_s.compute_returns(nv.copy(), value_normalizer=vn_n)
        _check_arrays_equal(f'sep.compute_returns[cfg={cfg}].returns', orig_s.returns, new_s.returns)


def test_broadcast_to_equivalence(seed=0):
    """The runner's np.repeat vs np.broadcast_to change: both must produce identical
    contents when written into a preallocated slot."""
    rng = np.random.RandomState(seed)
    n_threads, n_agents, obs_dim = 4, 3, 8
    obs = rng.randn(n_threads, n_agents, obs_dim).astype(np.float32)
    sh = obs.reshape(n_threads, -1)
    share_dim = sh.shape[-1]

    orig_share = np.expand_dims(sh, 1).repeat(n_agents, axis=1)
    new_share = np.broadcast_to(np.expand_dims(sh, 1), (n_threads, n_agents, share_dim))

    orig_slot = np.zeros((n_threads, n_agents, share_dim), dtype=np.float32)
    new_slot = np.zeros((n_threads, n_agents, share_dim), dtype=np.float32)
    orig_slot[...] = orig_share
    new_slot[...] = new_share
    _check_arrays_equal('broadcast_to vs repeat into slot', orig_slot, new_slot)
    # Also directly equal without writing into slot.
    _check_arrays_equal('broadcast_to vs repeat direct', orig_share, new_share)


def test_reshape_vs_split_stack(seed=0):
    rng = np.random.RandomState(seed)
    N, M = 4, 3
    for tail in [(), (5,), (2, 8), (1, 64)]:
        x = rng.randn(N * M, *tail).astype(np.float32)
        orig = np.array(np.split(x, N))
        new = x.reshape(N, -1, *x.shape[1:])
        _check_arrays_equal(f'split-stack vs reshape tail={tail}', orig, new)


def test_feed_forward_generator_equiv(seed=0):
    """After insert+compute_returns, iterate the minibatch generator and diff each batch."""
    args = make_buffer_args(episode_length=8, n_rollout_threads=4, use_valuenorm=True)
    obs_space, share_obs_space, act_space = _make_spaces()
    num_agents = 3
    orig = OrigSharedBuffer(args, num_agents, obs_space, share_obs_space, act_space)
    new = NewSharedBuffer(args, num_agents, obs_space, share_obs_space, act_space)

    rng = np.random.RandomState(seed)
    _drive_insert_and_returns(orig, new, num_agents, rng, next_value=None)

    torch.manual_seed(seed)
    vn_o = OrigValueNorm(input_shape=1, norm_axes=1)
    vn_n = NewValueNorm(input_shape=1, norm_axes=1)
    warm = rng.randn(16, 1).astype(np.float32)
    vn_o.update(warm); vn_n.update(warm)

    nv = rng.randn(args.n_rollout_threads, num_agents, 1).astype(np.float32)
    orig.compute_returns(nv.copy(), value_normalizer=vn_o)
    new.compute_returns(nv.copy(), value_normalizer=vn_n)

    # Minibatch iteration (non-transformer path). Reseed torch to give identical randperms.
    adv_o = orig.returns[:-1] - vn_o.denormalize(orig.value_preds[:-1])
    adv_n = new.returns[:-1] - vn_n.denormalize(new.value_preds[:-1])
    _check_arrays_equal('advantages pre-generator', adv_o, adv_n)

    # Generators are lazy — materialize each fully under its own fresh torch seed so
    # both randperm calls see the same initial state.
    torch.manual_seed(seed)
    batches_o = list(orig.feed_forward_generator(adv_o, num_mini_batch=4))
    torch.manual_seed(seed)
    batches_n = list(new.feed_forward_generator(adv_n, num_mini_batch=4))
    names = ['share_obs', 'obs', 'rnn_states', 'rnn_states_critic', 'actions',
             'value_preds', 'returns', 'masks', 'active_masks', 'action_log_probs',
             'adv_targ', 'available_actions']
    for i, (bo, bn) in enumerate(zip(batches_o, batches_n)):
        for name, arr_o, arr_n in zip(names, bo, bn):
            _check_arrays_equal(f'gen batch {i} {name}', arr_o, arr_n)


# ---------- registry ----------

EQUIV_TESTS = [
    ('shared_buffer.insert/after_update', test_shared_buffer_insert_and_after_update),
    ('separated_buffer.insert/after_update', test_separated_buffer_insert_and_after_update),
    ('valuenorm.normalize/denormalize', test_valuenorm_bitexact),
    ('popart.normalize/denormalize', test_popart_bitexact),
    ('compute_returns (all GAE branches)', test_compute_returns_equiv),
    ('broadcast_to vs repeat', test_broadcast_to_equivalence),
    ('reshape vs split-stack', test_reshape_vs_split_stack),
    ('feed_forward_generator (full cycle)', test_feed_forward_generator_equiv),
]


def run_equiv(seeds=(0, 1, 2)):
    print('=== Equivalence (bit-exact) ===')
    all_pass = True
    for name, fn in EQUIV_TESTS:
        for s in seeds:
            try:
                fn(seed=s) if 'seed' in fn.__code__.co_varnames else fn()
                print(f'  {name:<42s} seed={s}  ✓')
            except (AssertionError, Exception) as e:
                all_pass = False
                print(f'  {name:<42s} seed={s}  ✗ {type(e).__name__}: {e}')
    if not all_pass:
        print('FAILED — see above.')
        sys.exit(1)
    print('All equivalence checks passed.')


# ---------- benchmarks ----------

def bench_insert(n_calls=5000, num_agents=10, n_rollout_threads=8):
    args = make_buffer_args(episode_length=n_calls + 1,
                            n_rollout_threads=n_rollout_threads)
    obs_space, share_obs_space, act_space = _make_spaces(obs_dim=64, share_obs_dim=256)
    rng = np.random.RandomState(0)
    sh = rng.randn(n_rollout_threads, num_agents, 256).astype(np.float32)
    ob = rng.randn(n_rollout_threads, num_agents, 64).astype(np.float32)
    ra = rng.randn(n_rollout_threads, num_agents, args.recurrent_N, args.hidden_size).astype(np.float32)
    rc = np.zeros_like(ra)
    a = rng.randint(0, 5, size=(n_rollout_threads, num_agents, 1)).astype(np.float32)
    alp = rng.randn(n_rollout_threads, num_agents, 1).astype(np.float32)
    v = rng.randn(n_rollout_threads, num_agents, 1).astype(np.float32)
    r = rng.randn(n_rollout_threads, num_agents, 1).astype(np.float32)
    m = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)

    def _time_one(cls):
        buf = cls(args, num_agents, obs_space, share_obs_space, act_space)
        t0 = time.perf_counter()
        for _ in range(n_calls):
            buf.insert(sh, ob, ra, rc, a, alp, v, r, m)
            # Reset step so insert slot stays valid (we bumped episode_length so it won't wrap).
        return time.perf_counter() - t0

    t_o = median([_time_one(OrigSharedBuffer) for _ in range(3)])
    t_n = median([_time_one(NewSharedBuffer) for _ in range(3)])
    return t_o, t_n


def bench_compute_returns(num_agents=10, n_rollout_threads=8, episode_length=80,
                          use_valuenorm=True, repeats=5):
    args = make_buffer_args(episode_length=episode_length,
                            n_rollout_threads=n_rollout_threads,
                            use_valuenorm=use_valuenorm)
    obs_space, share_obs_space, act_space = _make_spaces(obs_dim=64, share_obs_dim=256)

    def _time_one(cls_buf, cls_vn):
        buf = cls_buf(args, num_agents, obs_space, share_obs_space, act_space)
        # Fill with random data once so compute_returns has something to do.
        rng = np.random.RandomState(0)
        for step in range(episode_length):
            buf.insert(
                rng.randn(n_rollout_threads, num_agents, 256).astype(np.float32),
                rng.randn(n_rollout_threads, num_agents, 64).astype(np.float32),
                np.zeros((n_rollout_threads, num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32),
                np.zeros((n_rollout_threads, num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32),
                rng.randint(0, 5, size=(n_rollout_threads, num_agents, 1)).astype(np.float32),
                rng.randn(n_rollout_threads, num_agents, 1).astype(np.float32),
                rng.randn(n_rollout_threads, num_agents, 1).astype(np.float32),
                rng.randn(n_rollout_threads, num_agents, 1).astype(np.float32),
                np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32),
            )
        if use_valuenorm:
            torch.manual_seed(0)
            vn = cls_vn(input_shape=1, norm_axes=1)
            vn.update(rng.randn(32, 1).astype(np.float32))
        else:
            vn = None
        nv = rng.randn(n_rollout_threads, num_agents, 1).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(repeats):
            buf.compute_returns(nv, value_normalizer=vn)
        return (time.perf_counter() - t0) / repeats

    t_o = median([_time_one(OrigSharedBuffer, OrigValueNorm) for _ in range(3)])
    t_n = median([_time_one(NewSharedBuffer, NewValueNorm) for _ in range(3)])
    return t_o, t_n


def run_bench():
    print('=== Benchmark (median of 3 repeats) ===')
    print(f"{'operation':<32s} {'orig (s)':>12s} {'new (s)':>12s} {'speedup':>10s}")
    print('-' * 70)

    t_o, t_n = bench_insert()
    print(f"{'shared_buffer.insert x5000':<32s} {t_o:>12.4f} {t_n:>12.4f} {(t_o / t_n):>9.2f}x")

    t_o, t_n = bench_compute_returns(use_valuenorm=True)
    print(f"{'compute_returns [valuenorm]':<32s} {t_o:>12.4f} {t_n:>12.4f} {(t_o / t_n):>9.2f}x")

    t_o, t_n = bench_compute_returns(use_valuenorm=False)
    print(f"{'compute_returns [no valuenorm]':<32s} {t_o:>12.4f} {t_n:>12.4f} {(t_o / t_n):>9.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['equiv', 'bench', 'all'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if args.mode in ('equiv', 'all'):
        run_equiv(seeds=tuple(args.seeds))
    if args.mode in ('bench', 'all'):
        print()
        run_bench()


if __name__ == '__main__':
    main()
