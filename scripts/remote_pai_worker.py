"""Per-GPU (or CPU) job worker for the PhysicalAI pipeline and later rounds.

Jobs live in <jobs dir>/jobs.json (default reports/pai-jobs) as an ordered list of
{"name", "after": [...], "cmd", "device": "gpu"|"cpu", "env": "model"|"scorer"}.
A worker claims the first job of its device whose dependencies are done (atomic mkdir), runs it
in the requested environment (GPU workers on their GPU, `--gpu -1` = CPU worker with CUDA hidden),
and writes <name>.done or <name>.failed. Dependents of a failed job are never started. The job file
is re-read on every pass, so jobs can be appended while workers run. Workers exit when every job
of their device is done, failed, blocked by a failure, or claimed by another worker; with
`--linger` they keep polling until <jobs dir>/STOP exists.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
JOBS = ROOT / "reports/pai-jobs"

def state(name):
    for s in ["done", "failed"]:
        if (JOBS / f"{name}.{s}").exists():
            return s
    return "claimed" if (JOBS / f"{name}.claim").exists() else "open"

def log(**kw):
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **kw}), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--jobs", default=str(JOBS)); p.add_argument("--linger", action="store_true")
    a = p.parse_args()
    JOBS = Path(a.jobs)
    device = "cpu" if a.gpu < 0 else "gpu"
    prefix = JOBS.name.replace("-jobs", "-job")
    while True:
        jobs = json.loads((JOBS / "jobs.json").read_text())
        names = {j["name"] for j in jobs}
        blocked = set()
        for j in jobs:                                           # propagate failures down the list
            if any(state(d) == "failed" or d in blocked for d in j.get("after", [])):
                blocked.add(j["name"])
        pick, waiting = None, False
        for j in jobs:
            st = state(j["name"])
            if st != "open" or j["name"] in blocked or j.get("device", "gpu") != device:
                continue
            deps = j.get("after", [])
            assert all(d in names for d in deps), (j["name"], deps)
            if all(state(d) == "done" for d in deps):
                try:
                    os.mkdir(JOBS / f"{j['name']}.claim")
                except FileExistsError:
                    continue
                pick = j
                break
            waiting = True
        if pick is None:
            if (JOBS / "STOP").exists() or (not waiting and not a.linger):
                log(gpu=a.gpu, event="exit"); break
            time.sleep(60); continue
        log(gpu=a.gpu, event="start", job=pick["name"])
        t0 = time.monotonic()
        hide = " && export CUDA_VISIBLE_DEVICES=" if device == "cpu" else ""
        cmd = (f"export LITE_GPU_IDS={max(a.gpu, 0)} && source {ROOT}/scripts/remote_env.sh {pick.get('env', 'model')} >/dev/null"
               f"{hide} && cd {ROOT}/scripts && {pick['cmd']}")
        with open(ROOT / "logs" / f"{prefix}-{pick['name']}.log", "a") as out:
            rc = subprocess.call(["bash", "-c", cmd], stdout=out, stderr=subprocess.STDOUT)
        (JOBS / f"{pick['name']}.{'done' if rc == 0 else 'failed'}").write_text(json.dumps(
            {"gpu": a.gpu, "rc": rc, "elapsed_s": time.monotonic() - t0}) + "\n")
        log(gpu=a.gpu, event="end", job=pick["name"], rc=rc, elapsed_s=round(time.monotonic() - t0))
