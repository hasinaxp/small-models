#!/usr/bin/env bash
# End-to-end distillation from SmolLM2-1.7B: student vocab + vocab map ->
# teacher-token data -> distillation. Every stage skips finished work, so
# rerunning resumes.
#
#   ./run_distill.sh                          # 2B student tokens, defaults
#   TOKENS=1e9 ./run_distill.sh --alpha 0.8   # extra args go to distill.py
set -euo pipefail
cd "$(dirname "$0")"

VOCAB=${VOCAB:-16384}
TOKENS=${TOKENS:-2e9}                         # student tokens to train on
TEACHER_TOKENS=${TEACHER_TOKENS:-3e9}         # teacher tokens to prepare (~1.1 student each)
MIX=${MIX:-fineweb-edu-dedup:0.7,cosmopedia-v2:0.3}
STUDENT=tokenizer/student$((VOCAB / 1024))k

if [ ! -f tokenizer/smollm2/tokenizer.json ]; then
    mkdir -p tokenizer/smollm2
    python -c "from huggingface_hub import hf_hub_download; import shutil; \
shutil.copy(hf_hub_download('HuggingFaceTB/SmolLM2-1.7B', 'tokenizer.json'), \
'tokenizer/smollm2/tokenizer.json')"
fi
if [ ! -f "$STUDENT/vocab_map.pt" ]; then
    python distill_vocab.py --vocab "$VOCAB" --out "$STUDENT"
fi

python prepare_data.py --tokenizer tokenizer/smollm2/tokenizer.json \
    --out data/smollm_t49k --tokens "$TEACHER_TOKENS" --mix "$MIX"

PYTHONUNBUFFERED=1 python distill.py --data data/smollm_t49k --train_tokens "$TOKENS" \
    --vocab_map "$STUDENT/vocab_map.pt" --student_tokenizer "$STUDENT/tokenizer.json" "$@"
