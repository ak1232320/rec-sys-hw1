#!/bin/bash
# Wait for the model download to finish, then start the generation run.
cd /home/mstarchenko/xai_x || exit 1
for i in $(seq 1 120); do
  test -f models/qwen3-1.7b/.ok && break
  sleep 10
done
date +%H:%M:%S
bash ~/hw8_upload/start_llm.sh
