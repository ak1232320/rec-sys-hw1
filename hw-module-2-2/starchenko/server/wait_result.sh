#!/bin/bash
# Block until the generation run writes run_meta.json (or fails), then print the outcome.
cd /home/mstarchenko/xai_x || exit 1
for i in $(seq 1 90); do
  test -f hw8/out/run_meta.json && break
  if test -f logs/hw8_llm.log && ! pgrep -f run_prefix_llm.py > /dev/null && test $i -gt 6; then
    echo 'generation process is not running'; break
  fi
  sleep 10
done
cat logs/hw8_wait.log
tail -n 25 logs/hw8_llm.log | cut -c1-300
