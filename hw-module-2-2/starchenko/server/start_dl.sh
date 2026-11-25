#!/bin/bash
cd /home/mstarchenko/xai_x || exit 1
mkdir -p hw8 logs models
cp ~/hw8_upload/dl_qwen17.py hw8/
setsid nohup venv/bin/python hw8/dl_qwen17.py > logs/hw8_dl.log 2>&1 < /dev/null &
sleep 5
cat logs/hw8_dl.log
venv/bin/python -c 'import vllm, torch, transformers; print("versions", vllm.__version__, torch.__version__, transformers.__version__)' 2>/dev/null | tail -1
