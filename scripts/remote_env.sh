# Source from an interactive shell: source /path/to/scripts/remote_env.sh model
# Use 'scorer' for the isolated official CPU evaluation environment.
LITE_ROOT="/home/yonghwi/DriveWAM_Lightweight_Action_Readout"
export LITE_ROOT
export UV_CACHE_DIR="$LITE_ROOT/caches/uv"
export UV_PYTHON_INSTALL_DIR="$LITE_ROOT/python"
export HF_HOME="$LITE_ROOT/caches/huggingface"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$LITE_ROOT/tools/runtime-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
case "${1:-model}" in
  model)
    source "$LITE_ROOT/.venvs/drivewam-model/bin/activate"
    export CUDA_HOME=/usr/local/cuda
    export CUDA_VISIBLE_DEVICES="${LITE_GPU_IDS:-0,1}"
    export PATH="$CUDA_HOME/bin:$PATH"
    ;;
  scorer)
    source "$LITE_ROOT/.venvs/drivewam-scorer/bin/activate"
    export CUDA_VISIBLE_DEVICES=""
    # The numpy 1.23.4 wheel's OpenBLAS 0.3.20 Cooperlake kernel gives wrong
    # pinv results on this Xeon. The measured Haswell path passes numeric checks.
    export OPENBLAS_CORETYPE=Haswell
    export OPENBLAS_NUM_THREADS=1
    export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
    export NUPLAN_MAPS_ROOT="$LITE_ROOT/data/maps"
    export NAVSIM_EXP_ROOT="$LITE_ROOT/runs/navsim"
    export NAVSIM_DEVKIT_ROOT="$LITE_ROOT/upstream/navsim-v1.1"
    export OPENSCENE_DATA_ROOT="$LITE_ROOT/data"
    # Avoid importing DriveWAM's vendored navsim through the current directory.
    cd "$LITE_ROOT"
    ;;
  *)
    echo "Use: source remote_env.sh model|scorer" >&2
    return 1
    ;;
esac
