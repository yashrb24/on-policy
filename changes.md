# Changes Log

Tracks modifications made to align PredatorPrey (PP) and TrafficJunction (TJ)
training configs with IC3Net's published setup and with Aditya's MAPPO reference
(`AdityaKapoor74/Multi-Agent-Limited-Comms`).

Reference sources:
- IC3Net README: https://github.com/IC3Net/IC3Net
- Aditya's on-policy scripts: `AdityaKapoor74/Multi-Agent-Limited-Comms/on-policy/onpolicy/scripts`

---

## Predator Prey

### `onpolicy/scripts/train_pp_scripts/medium.sh`

Rewrote to mirror `hard.sh` format, with IC3Net medium-task overrides.

IC3Net PP medium (per README): `nagents=5`, `max_steps=40`, `vision=1`, `dim=10`.

| Param | Before | After |
|---|---|---|
| Format | compact single-command | `hard.sh`-style (uppercase vars, `BEST_*` block, DDCL/fakequant comment blocks, timestamp logdir) |
| `NUM_AGENTS` | 5 | 5 |
| `DIM` | 10 | 10 |
| `VISION` | 1 | 1 |
| `EPISODE_LENGTH` | 40 | 40 |
| `SCENARIO_NAME` | `medium` | `medium` |

(PP medium was already IC3Net-aligned on values; this change was purely cosmetic for layout consistency.)

### `onpolicy/scripts/train_pp_scripts/hard.sh`

No changes — already matched IC3Net PP hard (`nagents=10, dim=20, episode_length=80, vision=1`).

### Architecture (both PP scripts)

Already using `hidden_size=64, n_embd=64, n_head=4, n_block=2` — matches Aditya's TJ/PP/football MAPPO config. No change needed.

---

## Traffic Junction

### `onpolicy/scripts/train_tj_scripts/medium.sh`

Full rewrite plus semantic fixes. Target: IC3Net TJ medium + Aditya-style architecture.

IC3Net TJ medium (README easy base + medium overrides): `nagents=10`, `max_steps=40`, `dim=14`, `vision=0` (inherited from easy base), `add_rate_min=0.05`, `add_rate_max=0.02`, `curr_start=250`, `curr_end=1250`, `difficulty=medium`.

**Format:**
- Reformatted from compact single-command layout to `hard.sh`-style (`ENV_NAME`, `BEST_*` hyperparameter block, DDCL/fake-quant comment blocks, `TIMESTAMP` / `RUN_DIR` / `LOG_FILE` setup).
- Dropped inline `CUDA_VISIBLE_DEVICES=0` (now external).
- Dropped `use_wandb=False` conditional (WandB enabled by default via `--user_name`/`--wandb_name`).

**Task / curriculum:**

| Param | Before | After |
|---|---|---|
| `VISION` | 1 | 0 |
| `EPISODE_LENGTH` | 80 | 40 |
| `ADD_RATE_MIN` | 1.0 | 0.05 |
| `ADD_RATE_MAX` | 1.0 | 0.02 |
| `CURR_START` | 1 | 250 |
| `CURR_END` | 1 | 1250 |

Note: `add_rate_min > add_rate_max` is copied verbatim from the IC3Net README. Verified in env code (`TrafficJunction_Env.py:204-206`) that this triggers the curriculum guard `add_rate_range > 0`, disabling the curriculum ramp and holding spawn rate constant at `add_rate_min=0.05`. Intentional per IC3Net; no fix needed.

**Architecture:**

| Param | Before | After |
|---|---|---|
| `hidden_size` | 128 | 64 |
| `n_embd` | 128 | 64 |
| `n_block` | unset (default 1) | 2 |
| `n_head` | 4 | 4 |

Added `--n_block $BEST_N_BLOCK` to the python command (previously not passed).

**Other flags:**
- Removed `--use_active_masks_in_transformer`. Verified against all 6 of Aditya's on-policy training scripts — none pass this flag. It's a no-op in PP/football but had real effect in TJ; removing it unifies behavior with Aditya.

### `onpolicy/scripts/train_tj_scripts/hard.sh`

IC3Net TJ hard (README easy base + hard overrides): `nagents=20`, `max_steps=80`, `dim=18`, `vision=0` (inherited), `add_rate_min=0.02`, `add_rate_max=0.05`, `curr_start=250`, `curr_end=1250`, `difficulty=hard`.

Kept the existing compact layout (user preference). Only semantic values updated.

**Task / curriculum:**

| Param | Before | After |
|---|---|---|
| `vision` | 1 | 0 |
| `add_rate_min` | 0.05 | 0.02 |
| `add_rate_max` | 0.05 | 0.05 |
| `curr_start` | 1 | 250 |
| `curr_end` | 1 | 1250 |

Pre-change state had `add_rate_min == add_rate_max` → curriculum guard fails → spawn rate pinned at 0.05 throughout. Post-change state ramps spawn rate 0.02 → 0.05 over epochs 250–1250, matching IC3Net's published behavior.

**Architecture:**

| Param | Before | After |
|---|---|---|
| `hidden_size` | 128 | 64 |
| `n_embd` | 128 (inline in cmd) | 64 |
| `n_block` | unset (default 1) | 2 |
| `n_head` | 4 | 4 |

Added `--n_block ${n_block}` to the python command.

**Other flags:**
- Removed `--use_active_masks_in_transformer` (same reasoning as medium).

---

## Sweep infrastructure

### `onpolicy/scripts/sweeps/predatorprey/configs/sweep_config_ddcl_hard.yaml`

- Added top-level `name: ddcl sweep pp` so the sweep appears with that name in WandB.

### `onpolicy/scripts/sweeps/trafficjunction/` (new)

Created TJ sweep infrastructure mirroring the PP layout:

```
sweeps/trafficjunction/
├── configs/
│   └── sweep_config_ddcl_hard.yaml   # WandB sweep name: "ddcl sweep tj"
├── run_sweep.py                       # launcher (same pattern as PP)
└── sweep_wrapper.py                   # calls train_traffic_junction.main
```

**`sweep_wrapper.py` adaptations vs PP version:**
- Uses `--difficulty` instead of `--scenario_name` (TJ's train script accepts `--difficulty`).
- Adds TJ-specific args: `--add_rate_min`, `--add_rate_max`, `--curr_start`, `--curr_end`, `--dim`, `--vision`.
- Calls `from train_traffic_junction import main`.
- Optional toggles for `use_active_masks_in_transformer`, `use_fake_quantization`, `quant_bits` (off unless YAML enables).

**`sweep_config_ddcl_hard.yaml` fixed params** (aligned with updated `hard.sh`):
- Env: `num_agents=20`, `dim=18`, `vision=0`, `episode_length=80`, `difficulty=hard`
- Curriculum: `add_rate_min=0.02`, `add_rate_max=0.05`, `curr_start=250`, `curr_end=1250`
- Training: `num_env_steps=320000`, `n_rollout_threads=1`, `lr=1e-3`, `ppo_epoch=10`, `num_mini_batch=1`
- Arch: `hidden_size=64`, `n_embd=64`, `n_block=2`, `n_head=4`
- DDCL: `use_comms_channel=true`

**Swept dims:** `seed ∈ [8, 12, 18, 35, 41]`, `comm_coeff ∈ [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]`, `num_messages ∈ [10, 15, 20, 25, 30]`, `ddcl_variation ∈ [old, new]`. Grid size = 5 × 5 × 5 × 2 = 250 runs.

Launch command:
```bash
cd onpolicy/scripts/sweeps/trafficjunction
python run_sweep.py --config configs/sweep_config_ddcl_hard.yaml --project ddcl-applications --n-agents 3 --gpus 0
```

---

## Open items / notes (not yet addressed)

These came up in analysis but no code changes were applied. Flagged here for later.

### PP env has no time-limit enforcement

`onpolicy/envs/predator_prey/PredatorPrey_env.py` only sets `episode_over=True` on prey capture — no `max_steps` counter. Combined with the runner (`runner/shared/predator_prey_runner.py`) not setting `bad_masks`, episodes effectively run **indefinitely** across rollout boundaries until the prey is caught. This is a meaningful divergence from both IC3Net (trainer-enforced `max_steps=80` terminal) and the TJ env (self-enforced `max_steps` terminal).

Consequence: PP agents are trained on an **unbounded-horizon** variant of the task, not the bounded PP hard/medium from the paper. Win-rate/episode-length stats logged at rollout boundaries (`predator_prey_runner.py:48-69`) are also decoupled from true episode boundaries and misrepresent actual episode outcomes.

Two possible fixes (not applied):
1. **Match IC3Net (termination semantics):** add `self._step_count` to `PredatorPrey_env.py`, set `episode_over=True` at `step_count >= max_steps`. Existing `masks=0` path handles the rest.
2. **Proper truncation semantics:** env signals timeout with `info['bad_transition']=True`, runner populates `bad_masks`, add `--use_proper_time_limits` to the training script. Buffer already supports this via `shared_buffer.py:185-215`.

### IC3Net budget shortfall

IC3Net's published PP and TJ configs run for `num_epochs=2000 × epoch_size=10 × batch_size=500 × nprocesses=16 ≈ 160M–180M env steps`. Current scripts target `3M` (PP) and `320K–3M` (TJ). For fair budget-matched comparison, either scale up to IC3Net's budget or pick a smaller shared budget and report both methods at that budget.

### TJ env's `episode_length` coupling

`TrafficJunction_Env.py:70` sets `self.max_steps = args.episode_length`, coupling the env horizon to the rollout buffer size. Can't decouple rollout length from task horizon without adding a dedicated `max_steps` arg. Fine for 1 episode per rollout; would need to revisit if sweeping rollout length independently.

### Leftover flags elsewhere

`--use_active_masks_in_transformer` still present in:
- `onpolicy/scripts/train_tj_scripts/easy.sh`
- `onpolicy/scripts/train_tj_scripts/optimized_parallel_sweep.sh`

Not updated yet (out of scope for medium/hard alignment).
