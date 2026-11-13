#!/bin/bash
# report_arm.sh <jobid> <tag> — block until the job leaves the queue, then print its curve.
id=$1; tag=$2
cd /home/doyoonkim/projects/distIL
while squeue -j "$id" -h -o "%t" 2>/dev/null | grep -qE "R|PD|CG"; do sleep 300; done
echo "=== $(date '+%m-%d %H:%M')  $tag ($id)  $(sacct -j $id --format=State -n 2>/dev/null|head -1|tr -d ' ') ==="
L=$(ls -t logs/*_${id}.log 2>/dev/null | head -1)
[ -z "$L" ] && { echo "  no log"; exit 0; }
python3 -I -c "
import re
for line in open('$L',errors='ignore'):
    if 'val-core' not in line: continue
    a=re.search(r'acc/mean@16:np\.float64\(([0-9.]+)\)',line)
    b=re.search(r'acc/maj@16/mean:np\.float64\(([0-9.]+)\)',line)
    s=re.search(r'step:(\d+)',line)
    if a and b: print(f'  step {s.group(1) if s else \"?\":>3}  mean@16 {float(a.group(1))*100:5.2f}  maj@16 {float(b.group(1))*100:5.2f}')
"
E=$(ls -t logs/*_${id}.err 2>/dev/null | head -1)
grep -iE "Traceback|CUDA out of memory|TIME LIMIT|CANCELLED" "$E" 2>/dev/null | tail -3 | cut -c1-160
echo "  valdump: $(ls valdump/${tag} 2>/dev/null | wc -l) files"
