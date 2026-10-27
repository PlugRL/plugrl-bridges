# plugrl-bridges

Trainers that were not written for PlugRL train environments that PlugRL env
clients run, through the [PlugRL protocol](https://github.com/PlugRL/plugrl-protocol).

An env client speaks the protocol from wherever its environment has to
live:
- another process, or another machine with no GPU and no deep learning stack;
- a program in another language: the C++ client in plugrl-protocol links
  nothing but the C++ runtime.

This package puts such clients behind the environment interface a trainer
already has:

| adapter | for | interface |
| --- | --- | --- |
| `plugrl_bridges.rlinf.PlugRLEnv` | [RLinf](https://github.com/RLinf/RLinf) | RLinf's env contract, `env_type: plugrl` |
| `plugrl_bridges.gym.PlugRLVectorEnv` | CleanRL, TorchRL, your own loop | `gymnasium.vector.VectorEnv` |
| `plugrl_bridges.sb3.PlugRLVecEnv` | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | SB3's `VecEnv` |

```python
import gymnasium as gym, numpy as np
from stable_baselines3 import PPO
from plugrl_bridges.sb3 import PlugRLVecEnv

env = PlugRLVecEnv(num_envs=16, port=8800,
                   action_space=gym.spaces.Box(-1, 1, (6,), np.float32))
PPO("MlpPolicy", env).learn(409_600)
```

The adapter blocks until 16 environments have connected, and reads the
observation space off what they sent.

## How it works

All three adapters sit on one `BridgeServer`, a PlugRL protocol server that
the trainer steps from its own thread:
- **Slots.** Each env client takes slots in the order it connects.
- **Steps.** Each step sends every client its actions. It returns once each
  client has fed back and asked for its next action.
- **Episode ends.** For each env whose episode ended, the step returns its
  reset observation, with the terminal one beside it, the way each trainer
  expects: `final_observation` for RLinf, `final_obs` under gymnasium's
  SAME_STEP autoreset, `terminal_observation` for SB3. Episode returns and
  lengths are counted by the bridge, from the raw rewards.
- **Observations.** The SB3 and gymnasium adapters give the trainer the
  clients' `state_keys`, flattened into float32, or, with `image_key`, that
  one image as sent. A uint8 [H, W, C] camera frame becomes a uint8 Box that
  SB3's CnnPolicy takes as an image. Discrete action spaces work too: an
  action crosses as one float per env.
- **Episode log.** With `episode_log=PATH` (RLinf: `env.train.episode_log`),
  or `PLUGRL_BRIDGES_EPISODE_LOG` set, the server writes one CSV line per
  episode: `step,slot,episode,return,length`. It is the same record
  whichever trainer steps the bridge, so runs of different trainers can be
  compared on it.
- **Features.** The server offers `reuse-feedback-obs`, so a client that
  uses it sends each observation once.

It is a **lockstep** server: these trainers step all their environments
together, so every infer and feedback on a connection must name all of that
connection's envs. A client running one-step chunks always does, and the
server's metadata says `action_horizon: 1`. It never fills in a step it did
not receive:
- a client that breaks lockstep is closed for a resync;
- a client too many is told to stop;
- a client lost mid-run stops the run (`ClientLost`).

Two things a trainer cannot do through it:
- **Reset environments itself.** The protocol has no way for a server to
  ask, so `reset()` returns the observations the next step starts from.
- **Use per-env methods.** Those environments are reached only through the
  protocol.

**gymnasium notes.** As of gymnasium 1.3:
- Its vector observation wrappers refuse SAME_STEP autoreset, so
  `plugrl_bridges.gym.NormalizeObservation` is provided.
- Its vector `RecordEpisodeStatistics` counts as if autoreset were
  NEXT_STEP. Read `infos["episode"]` from the adapter instead.

## Install

It is not on PyPI; install it from the repository:

```bash
pip install "plugrl-bridges[sb3] @ git+https://github.com/PlugRL/plugrl-bridges.git"   # or [gym], [rlinf]
```

RLinf has a closed list of environment types, so it needs a two-hunk patch
to know `plugrl` (`patches/`).

## Examples

- `examples/sb3/ppo.py`: SB3's PPO.
- `examples/cleanrl/ppo_continuous_action.py`: CleanRL's PPO script, with
  only its environment creation and episode logging changed (MIT,
  `LICENSE-CLEANRL`).
- `examples/rlinf/`: RLinf PPO-MLP configs.
- `examples/run_bridged.sh`: one run of any of them, with 16 env clients.
- `examples/smoke_probe.py`, `examples/drive_probe.py`: checks against real
  clients on SPEC.md's probe environment.

## Tests

```bash
uv run --group dev pytest
```

## Paper

The fault-injection runs of *Detecting Silent Faults at the Environment-Trainer
Boundary in Reinforcement Learning* (CoRL 2026 Workshop on Everything Beneath
the Policy; E54-E58 in plugrl-server's `experiments/`) trained through this
package's Stable-Baselines3 adapter, and its bridge log is one of the checks
the paper compares
([PDF](https://plugrl.github.io/assets/silent-faults-webp2026.pdf),
[summary and video](https://plugrl.github.io/#when-the-boundary-is-wrong-nothing-says-so)).
The step ledger the paper recommends is packaged separately as
[step-ledger](https://github.com/PlugRL/step-ledger).

## Who

plugrl-bridges is part of PlugRL, built by [Chenhao Lu](https://github.com/CTP314),
[Zuo Gou](https://github.com/tactino) and
[Zilin Kang](https://github.com/nothingbutbut).

## License

Apache-2.0. `examples/cleanrl/` is CleanRL's, under MIT.
