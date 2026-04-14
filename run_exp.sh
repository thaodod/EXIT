#!/usr/bin/env sh

set -eu

if [ "$#" -ne 2 ]; then
  echo "Usage: sh run_exp.sh <method> <output_dir>" >&2
  exit 1
fi

METHOD=$1
OUTPUT_DIR=$2

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REPO_ROOT=$SCRIPT_DIR
CONDA_ENV_NAME="${CONDA_ENV_NAME:-312}"

: "${ATTN_COMP_MODEL_NAME_OR_PATH:=meta-llama/Meta-Llama-3.1-8B-Instruct}"
: "${ATTN_COMP_CHECKPOINT_PATH:=$REPO_ROOT/attn_comp/checkpoints/llama-attention-layer13-SFT_epoch-7.pth}"

export ATTN_COMP_MODEL_NAME_OR_PATH
export ATTN_COMP_CHECKPOINT_PATH

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH. Please activate env ${CONDA_ENV_NAME} before running this script." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

run_job() {
  dataset_name=$1
  input_path=$2
  k=$3

  if [ ! -f "$REPO_ROOT/$input_path" ]; then
    echo "Missing input file: $REPO_ROOT/$input_path" >&2
    exit 1
  fi

  output_path="$OUTPUT_DIR/${METHOD}_k${k}_${dataset_name}.json"
  printf 'Running %s for %s with k=%s\n' "$METHOD" "$dataset_name" "$k"
  conda run --no-capture-output -n "$CONDA_ENV_NAME" python "$REPO_ROOT/compress2json.py" \
    --input "$REPO_ROOT/$input_path" \
    --method "$METHOD" \
    --k "$k" \
    --output "$output_path"
}

printf 'Using method: %s\n' "$METHOD"
printf 'Using model path: %s\n' "$ATTN_COMP_MODEL_NAME_OR_PATH"
printf 'Using checkpoint: %s\n' "$ATTN_COMP_CHECKPOINT_PATH"
printf 'Saving outputs to: %s\n' "$OUTPUT_DIR"

run_job 2wikimultihop retrieved/contriever-msmarco_2wikimultihop/dev.json 5
run_job 2wikimultihop retrieved/contriever-msmarco_2wikimultihop/dev.json 20
run_job HotpotQA retrieved/contriever-msmarco_HotpotQA/dev.json 5
run_job HotpotQA retrieved/contriever-msmarco_HotpotQA/dev.json 20
run_job musique retrieved/contriever-msmarco_musique/dev.json 5
run_job musique retrieved/contriever-msmarco_musique/dev.json 20
run_job NQ retrieved/contriever-msmarco_NQ/dev.json 5
run_job NQ retrieved/contriever-msmarco_NQ/dev.json 20
run_job TQA retrieved/contriever-msmarco_TQA/test.json 5
run_job TQA retrieved/contriever-msmarco_TQA/test.json 20

printf 'All compression jobs finished.\n'