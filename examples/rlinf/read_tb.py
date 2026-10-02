"""Print RLinf's TensorBoard scalars for one run; argv: run_dir [tag_substring ...]."""

import pathlib
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = pathlib.Path(sys.argv[1])
want = sys.argv[2:]
events = sorted(run.rglob("events.out.tfevents*"))
acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
acc.Reload()
for tag in sorted(acc.Tags()["scalars"]):
    if want and not any(w in tag for w in want):
        continue
    values = [e.value for e in acc.Scalars(tag)]
    shown = " ".join(f"{v:.4g}" for v in values[:6])
    more = f" ... {values[-1]:.4g} ({len(values)})" if len(values) > 6 else ""
    print(f"{tag:<55} {shown}{more}")
