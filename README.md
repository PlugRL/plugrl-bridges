# plugrl-rlinf

[RLinf](https://github.com/RLinf/RLinf) trains environments that PlugRL env
clients run, through the [PlugRL protocol](https://github.com/PlugRL/plugrl-protocol).

RLinf runs its environments inside its own cluster, as Ray workers with
RLinf's Python and torch. This bridge is an RLinf environment type,
`plugrl`, whose environments are anywhere else that can reach a port:
- a `plugrl-env-client` on a machine with no GPU and no RLinf;
- the C++ client in plugrl-protocol, which links nothing but the C++
  runtime;
- anything else that speaks the protocol.

RLinf sees an ordinary vectorised environment. The client sees an ordinary
PlugRL training server.

## How it works

`PlugRLEnv` implements RLinf's environment contract, `reset()` and
`chunk_step(actions)` under `auto_reset`. It does so on top of
`BridgeServer`, a PlugRL server that runs in a thread of RLinf's
environment worker:
- **Reset** waits until every slot has asked for its first action.
- **Each `chunk_step`** sends every client its actions. It returns once each
  client has sent its feedback and asked again.
- **Episodes ending.** For each env whose episode ended, the terminal
  observation goes in `infos["final_observation"]`, and the episode's return
  and length in `infos["final_info"]["episode"]`. RLinf's own simulators do
  the same.
- **Features.** The server offers `reuse-feedback-obs`.

It is a **lockstep** server: RLinf steps all of its environments together,
so every infer and every feedback on a connection must name all of that
connection's envs. A client running one-step chunks always does, and the
server's metadata asks for `action_horizon: 1`. A client that breaks
lockstep is closed for a resync with the reason. A client that disconnects
mid-run stops the run (`ClientLost`). The bridge never fills in a step it
did not receive.

## Install

RLinf has a closed list of environment types, so it needs a two-line patch
to know `plugrl`. The patch is in `patches/`. Then:

```bash
pip install -e /path/to/RLinf          # with the patch applied
pip install plugrl-rlinf
```

## Tests

```bash
uv run --group dev pytest
```

The tests drive `PlugRLEnv` the way RLinf's environment worker does, against
fake clients that speak the real protocol.

## License

Apache-2.0.
