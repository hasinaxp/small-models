#!/usr/bin/env bash
# End-to-end pretraining: tokenizer -> tokenized data -> training.
# Every stage skips work already done, so rerunning resumes.
#
#   ./run_pretrain.sh                         # 5B tokens, defaults
#   TOKENS=2e9 ./run_pretrain.sh --micro_batch 8
#
# Extra arguments go to pretrain.py.
set -euo pipefail
cd "$(dirname "$0")"

VOCAB=${VOCAB:-16384}
TOKENS=${TOKENS:-5e9}
MIX=${MIX:-fineweb-edu-dedup:0.7,cosmopedia-v2:0.3}

if [ ! -f tokenizer/tokenizer.json ]; then
    python train_tokenizer.py --vocab "$VOCAB" --out tokenizer
fi

python prepare_data.py --tokenizer tokenizer/tokenizer.json --out data/smollm \
    --tokens "$TOKENS" --mix "$MIX"

PYTHONUNBUFFERED=1 python pretrain.py --data data/smollm --train_tokens "$TOKENS" "$@"
