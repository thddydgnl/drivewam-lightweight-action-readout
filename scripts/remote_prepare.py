"""Install model or scorer runtime in the project directory on the GPU host."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path.home() / "DriveWAM_Lightweight_Action_Readout"
UV = str(ROOT / "tools/uv")
ROLE = sys.argv[1]
STATUS = ROOT / "reports" / (ROLE + "-install-status.json")
ENV = dict(os.environ, UV_CACHE_DIR=str(ROOT / "caches/uv"),
           UV_PYTHON_INSTALL_DIR=str(ROOT / "python"),
           HF_HOME=str(ROOT / "caches/huggingface"), PYTHONNOUSERSITE="1")

def status(stage, **extra):
    data = {"role": ROLE, "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    STATUS.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data), flush=True)

def run(args, cwd=ROOT, env=ENV):
    print("COMMAND " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)

def clone(name, url, commit):
    path = ROOT / "upstream" / name
    if not path.exists():
        path.mkdir()
        run(["git", "init", path])
        run(["git", "remote", "add", "origin", url], path)
    run(["git", "fetch", "--depth", "1", "origin", commit], path)
    run(["git", "checkout", "--detach", commit], path)
    return path

def lock(python):
    result = subprocess.check_output([python, "-m", "pip", "freeze"], env=ENV, text=True)
    (ROOT / "locks" / (ROLE + "-freeze.txt")).write_text(result)
    run([python, "-m", "pip", "check"])

try:
    env_path = ROOT / ".venvs" / ("drivewam-" + ROLE)
    python = str(env_path / "bin/python")
    status("creating_python_environment")
    if ROLE == "model":
        if not Path(python).exists():
            run([UV, "venv", "--python", "/usr/bin/python3", "--seed", env_path])
        status("installing_pytorch_cu126")
        # uv's HTTP client timed out on this host's unrouted IPv6 CDN endpoints;
        # pip uses the working system address fallback for the official torch index.
        run([python, ROOT / "scripts/remote_pip_ipv4.py", "install", "--timeout", "120", "torch==2.9.0",
             "torchvision==0.24.0", "--index-url", "https://download.pytorch.org/whl/cu126"])
        status("installing_model_dependencies")
        run([UV, "pip", "install", "--python", python, "-r",
             ROOT / "scripts/requirements-model.candidate.txt", "setuptools==75.8.0",
             "wheel==0.45.1", "opencv-python-headless==4.10.0.82", "scipy==1.15.3",
             "pandas==2.3.3", "pyarrow==23.0.1"])
        status("fetching_pinned_sources")
        drivewam = clone("DriveWAM", "https://github.com/chenshi3/DriveWAM",
                         "abbefb407c9c553f221522fc26077d307620adb4")
        work = ROOT / "work/DriveWAM"
        if not work.exists():
            run(["git", "worktree", "add", "-b", "research/lite-readout", work, "HEAD"], drivewam)
        # The pinned release has no torch 2.9 / cp310 Linux wheel. Compile FA2 locally.
        status("building_flash_attention_2")
        build_env = dict(ENV, CUDA_HOME="/usr/local/cuda",
                         PATH="/usr/local/cuda/bin:" + ENV.get("PATH", ""),
                         FLASH_ATTENTION_FORCE_BUILD="TRUE", FLASH_ATTN_CUDA_ARCHS="80",
                         MAX_JOBS="24", NVCC_THREADS="2")
        run([UV, "pip", "install", "--python", python, "flash-attn==2.8.3",
             "--no-build-isolation", "--no-deps"], env=build_env)
    elif ROLE == "scorer":
        run([UV, "python", "install", "3.9.25"])
        if not Path(python).exists():
            run([UV, "venv", "--python", "3.9.25", "--seed", env_path])
        status("fetching_pinned_sources")
        navsim = clone("navsim-v1.1", "https://github.com/autonomousvision/navsim",
                       "3e8291bfa89ff247231e0227778840cd0a036896")
        nuplan = clone("nuplan-devkit", "https://github.com/motional/nuplan-devkit",
                       "ce3c323af01c0d7ec5672f7832ef53f9c679aab0")
        status("installing_pytorch_cpu")
        run([python, ROOT / "scripts/remote_pip_ipv4.py", "install", "--timeout", "120", "torch==2.0.1",
             "torchvision==0.15.2", "--index-url", "https://download.pytorch.org/whl/cpu"])
        # Preserve official fixed requirements and freeze compatible legacy transitive versions.
        req = ROOT / "locks/scorer-input.txt"
        lines = navsim.joinpath("requirements.txt").read_text().splitlines()
        req.write_text("\n".join(x for x in lines if not x.startswith("nuplan-devkit")) + "\n")
        constraints = ROOT / "locks/scorer-constraints.txt"
        constraints.write_text("pandas==1.5.3\nscipy==1.10.1\nmatplotlib==3.7.5\n"
                               "geopandas==0.13.2\nShapely==2.0.7\ntorchmetrics==1.3.2\n"
                               "timm==0.9.16\n")
        status("installing_navsim_dependencies")
        run([UV, "pip", "install", "--python", python, "-r", req, "-c", constraints])
        status("installing_pinned_navsim_and_nuplan")
        run([UV, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation",
             "-e", nuplan, "-e", navsim])
    else:
        raise ValueError(ROLE)
    status("checking_dependencies")
    lock(python)
    status("installed", python=python)
except Exception as error:
    traceback.print_exc()
    status("failed", error=str(error))
    sys.exit(1)
