#!/bin/bash
# Run one command under /usr/bin/time -v with an nvidia-smi sampler (200 ms) on GPU $GPU; write outputs/resource/<label>.*
#   GPU=0 bash scripts/resource_sample.sh <label> <command ...>
label=$1; shift; GPU=${GPU:-0}; out=outputs/resource; mkdir -p $out
nvidia-smi -i $GPU --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 200 > $out/$label.util.csv 2>/dev/null &
SP=$!
/usr/bin/time -v -o $out/$label.time.txt "$@" > $out/$label.log 2>&1; rc=$?
kill $SP 2>/dev/null; wait $SP 2>/dev/null
[ $rc -ne 0 ] && echo "  [$label] command failed (rc=$rc): $(grep -m1 -iE 'error|out of memory' $out/$label.log | cut -c1-120)"
python scripts/resource_summary.py $label
