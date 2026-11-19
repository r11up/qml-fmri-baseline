#!/usr/bin/env bash
# Experiments added for the IEEE Access resubmission, four jobs at a time.
#
#   tt_*       TTN-VQC and TensorHyper-VQC against parameter-matched twins (R2.3)
#   qmodels_*  the model suite again, now recording training accuracy (R2.7);
#              its test predictions must equal results/rerun22 exactly
#   mi_abide   the model suite on mutual-information connectivity (R2.5)
#   sens_*     failing classical comparators retuned inside each fold (R1.4)
#   kgen_*     effective dimension, concentration and expressibility of both
#              kernels against train and test performance (R2.6)
#   tt_abide_tuned  the tensor-train models with learning rate and stopping
#              epoch chosen inside each training fold, for all four arms
#   qmodels_adhd_nocontrol  ADHD-200 suite without the 148 subjects of the two
#              control-only sites, which otherwise only ever train (R1.3)
#
# Thread counts are pinned as in rerun_parallel.sh: each PyTorch job uses ~3
# cores, so four jobs fill twelve of fourteen and leave two for the system.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="./.venv/bin/python"
R="results/revision"
mkdir -p "$R"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 \
       OPENBLAS_NUM_THREADS=3 TORCH_NUM_THREADS=3 PYTHONPATH=src:scripts

UCLA=(--data-root data/UCLA-CNP-cc200 --cv stratified --folds 10)
jobs=(
  "tt_mdd|scripts/run_tensor_vqc.py --data-root data/REST-meta-MDD"
  "qmodels_mdd|scripts/run_quantum_models.py --data-root data/REST-meta-MDD"
  "tt_abide|scripts/run_tensor_vqc.py --data-root data/ABIDE-I"
  "qmodels_abide|scripts/run_quantum_models.py --data-root data/ABIDE-I"
  "mi_abide|scripts/run_quantum_models.py --data-root data/ABIDE-I --connectivity mutual_information"
  "tt_adhd|scripts/run_tensor_vqc.py --data-root data/ADHD-200"
  "qmodels_adhd|scripts/run_quantum_models.py --data-root data/ADHD-200"
  "tt_ucla|scripts/run_tensor_vqc.py ${UCLA[*]}"
  "qmodels_ucla|scripts/run_quantum_models.py ${UCLA[*]}"
  "sens_ucla|scripts/run_baseline_sensitivity.py --tag ucla ${UCLA[*]}"
  "sens_abide|scripts/run_baseline_sensitivity.py --tag abide --data-root data/ABIDE-I"
  "sens_adhd|scripts/run_baseline_sensitivity.py --tag adhd --data-root data/ADHD-200"
  "sens_mdd|scripts/run_baseline_sensitivity.py --tag mdd --data-root data/REST-meta-MDD"
  "kgen_ucla|scripts/run_kernel_generalization.py --tag ucla ${UCLA[*]}"
  "kgen_abide|scripts/run_kernel_generalization.py --tag abide --data-root data/ABIDE-I"
  "kgen_adhd|scripts/run_kernel_generalization.py --tag adhd --data-root data/ADHD-200"
  "kgen_mdd|scripts/run_kernel_generalization.py --tag mdd --data-root data/REST-meta-MDD"
  "tt_abide_tuned|scripts/run_tensor_vqc.py --data-root data/ABIDE-I --inner-select"
  "qmodels_adhd_nocontrol|scripts/run_quantum_models.py --data-root data/ADHD-200 --drop-single-class-sites"
)
[ $# -gt 0 ] && jobs=("$@")   # optional: run only the given "tag|command" entries

run () {
  local tag="${1%%|*}" cmd="${1#*|}"
  echo "[$(date -u +%H:%M:%SZ)] start $tag"
  # shellcheck disable=SC2086
  $PY -u $cmd --out "$R/$tag" > "$R/$tag.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] done  $tag (rc=$?)"
}
export -f run
export PY R
printf '%s\n' "${jobs[@]}" | xargs -P 4 -I{} bash -c 'run "$@"' _ {}
echo "all done"
