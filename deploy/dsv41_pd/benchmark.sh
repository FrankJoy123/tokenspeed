#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require SESSIONS_FILE SESSIONS_SHA256 OUTPUT_TOKENS ROUNDS PRIME_TOKENS CHUNK_TOKENS WARMUP_PASSES CONCURRENCIES
export BENCH_OUTPUT="${BENCH_OUTPUT:-$STATE_DIR/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$BENCH_OUTPUT" ]] || { echo 'Use a new BENCH_OUTPUT directory' >&2; exit 2; }
mkdir -p "$BENCH_OUTPUT"
"$VENV/bin/python" - <<'CHECK'
import hashlib,json,os,pathlib
p=pathlib.Path(os.environ['SESSIONS_FILE']);assert hashlib.sha256(p.read_bytes()).hexdigest()==os.environ['SESSIONS_SHA256'], 'Dataset hash mismatch'
a=json.loads(p.read_text());cs=[int(c) for c in os.environ['CONCURRENCIES'].split()]
assert all(1<=c<=32 for c in cs)
assert len(a)>=max(cs) and len({x['id'] for x in a[:max(cs)]})==max(cs)
assert int(os.environ['OUTPUT_TOKENS'])>1 and int(os.environ['ROUNDS'])>0 and int(os.environ['WARMUP_PASSES'])>=1
keys=['SESSIONS_SHA256','OUTPUT_TOKENS','ROUNDS','PRIME_TOKENS','CHUNK_TOKENS','WARMUP_PASSES','CONCURRENCIES','SIMULATED_ACCEPT_LEN']
(pathlib.Path(os.environ['BENCH_OUTPUT'])/'protocol.json').write_text(json.dumps({k:os.getenv(k) for k in keys},indent=2))
CHECK
for c in $CONCURRENCIES; do
 for ((pass=1;pass<=WARMUP_PASSES;pass++)); do
  "$VENV/bin/python" "$DEPLOY_DIR/flush.py" > "$BENCH_OUTPUT/warm${pass}-c${c}-flush.json"
  "$VENV/bin/python" "$DEPLOY_DIR/bench.py" --concurrency "$c" --tag "warm${pass}-c$c"
 done
 "$VENV/bin/python" "$DEPLOY_DIR/flush.py" > "$BENCH_OUTPUT/measured-c${c}-flush.json"
 "$VENV/bin/python" "$DEPLOY_DIR/bench.py" --concurrency "$c" --tag "measured-c$c"
done
"$VENV/bin/python" "$DEPLOY_DIR/summarize.py" "$BENCH_OUTPUT"
