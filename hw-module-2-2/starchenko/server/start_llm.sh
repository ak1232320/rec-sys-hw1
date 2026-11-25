#!/bin/bash
# Runs the prefix-reranking generation on the free MIG slice of the A100.
cd /home/mstarchenko/xai_x || exit 1
cp ~/hw8_upload/run_prefix_llm.py ~/hw8_upload/prompts.jsonl hw8/
test -f models/qwen3-1.7b/.ok || { echo 'model not downloaded yet'; exit 1; }
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# CUDA_VISIBLE_DEVICES stays unset: CUDA picks the first MIG device, the free slice.
setsid nohup venv/bin/python hw8/run_prefix_llm.py hw8/prompts.jsonl hw8/out > logs/hw8_llm.log 2>&1 < /dev/null &
echo started
