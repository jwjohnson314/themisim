#!/usr/bin/env bash
#
# run_pipeline.sh — download the THEMIS archive and build the FAISS index.
#
# Orchestrates the two CLIs (themis-download, then themis-build-index), storing
# everything under one data root. GPU is used automatically when available and
# the build falls back to CPU otherwise.
#
# Configure via environment variables (all optional):
#   DATA_ROOT   where CDFs are downloaded      (default ./data/cdf)
#   ARTIFACTS   where the index is written     (default ./data/artifacts)
#   WEIGHTS     checkpoint .tar                 (default: fetched into ./weights)
#   SITES       comma-separated site codes      (default: all sites)
#   START, END  date range, e.g. 2015-03        (default: full archive)
#   NLIST       IVF cells                        (default: auto)
#   WORKERS     parallel downloads              (default 6)
#   BATCH_SIZE  embed batch size                (default 128)
#
# Example:
#   SITES=fsmi START=2015-03 END=2015-03 ./scripts/run_pipeline.sh
#
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data/cdf}"
ARTIFACTS="${ARTIFACTS:-./data/artifacts}"
SITES="${SITES:-}"
START="${START:-}"
END="${END:-}"
NLIST="${NLIST:-auto}"
WORKERS="${WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Detect GPU so the user knows what to expect; the Python build also autodetects.
DEVICE="cpu"
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DEVICE="cuda"
fi

cat <<EOF
=== THEMIS ASI search pipeline ===
  data root : ${DATA_ROOT}
  artifacts : ${ARTIFACTS}
  sites     : ${SITES:-ALL}
  dates     : ${START:-(start)} .. ${END:-(end)}
  device    : ${DEVICE}
  nlist     : ${NLIST}

NOTE: the full archive is ~1M CDFs (multiple TB) and a full build is GPU-days.
      Restrict with SITES / START / END for a smaller run.
EOF

DL_ARGS=(--data-root "${DATA_ROOT}" --workers "${WORKERS}")
[ -n "${SITES}" ] && DL_ARGS+=(--sites "${SITES}")
[ -n "${START}" ] && DL_ARGS+=(--start "${START}")
[ -n "${END}" ]   && DL_ARGS+=(--end "${END}")

echo
echo ">>> Step 1/2: download"
themis-download "${DL_ARGS[@]}"

BUILD_ARGS=(--data-root "${DATA_ROOT}" --artifacts "${ARTIFACTS}"
            --nlist "${NLIST}" --device auto --batch-size "${BATCH_SIZE}")
[ -n "${WEIGHTS:-}" ] && BUILD_ARGS+=(--checkpoint "${WEIGHTS}")

echo
echo ">>> Step 2/2: build index"
themis-build-index "${BUILD_ARGS[@]}"

echo
echo "Done. Query with:"
echo "  themis-query --site <site> --datetime <YYYY-MM-DDTHH> --frame <n> --artifacts ${ARTIFACTS}"
