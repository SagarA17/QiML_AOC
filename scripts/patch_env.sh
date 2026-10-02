#!/bin/bash
# QiML_AOC env fixer: idempotent, safe to re-run. Applies the env settings that `pip install -r requirements.txt`
# does not capture. Only touches the project env (default below, or $QIML_AOC_ENV).
#   Usage:  bash scripts/patch_env.sh
set -euo pipefail
PREFIX=${QIML_AOC_ENV:-/lus/eagle/projects/ATLAS_workflow_ALCF/sagar/conda/envs/QiML_AOC}
[ -x "$PREFIX/bin/python" ] || { echo "no env at $PREFIX"; exit 1; }

# Cap torch/BLAS thread pools: the login node has hundreds of cores, and by-gpu jobs share a node. Defaults
# are overridable (scripts/run.pbs sets OMP_NUM_THREADS to the job's CPU share).
ACT="$PREFIX/etc/conda/activate.d"; mkdir -p "$ACT"
cat > "$ACT/qiml_aoc_threads.sh" <<'EOF'
#!/bin/bash
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
EOF
echo "wrote: $ACT/qiml_aoc_threads.sh"
echo "done. Re-activate the env for the thread caps to take effect."
