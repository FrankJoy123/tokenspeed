#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$VENV/bin/python" - <<'CHECK'
import os, requests
s=requests.Session();s.trust_env=False
u=os.environ['BENCH_URL']
r=s.get(u+'/workers',timeout=30);r.raise_for_status()
w=r.json()['workers'];assert len(w)==2 and all(x['is_healthy'] for x in w), w
r=s.post(u+'/v1/chat/completions',json={'model':'deepseek-v41-flash','messages':[{'role':'user','content':'What is 1 + 1? Answer briefly.'}],'temperature':0,'max_tokens':32},timeout=300)
r.raise_for_status();print(r.json())
CHECK
