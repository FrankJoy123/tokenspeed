"""Whole-request TPOT and measured output-window throughput, prime excluded."""
import json
import statistics
import sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for folder in sorted((root/'results').glob('measured-c*')):
    requests=[json.loads(p.read_text()) for d in folder.glob('turn*-*') if not d.name.startswith('turn0-') for p in d.glob('request-*.json')]
    assert requests and all(r['success'] for r in requests)
    window=max(r['events'][-1][0] for r in requests)-min(r['events'][0][0] for r in requests)
    row={'concurrency':int(folder.name.split('-c')[1]),'requests':len(requests),'ttft_mean_s':statistics.mean(r['ttft_s'] for r in requests),'tpot_mean_ms':statistics.mean(r['tpot_ms'] for r in requests),'output_window_s':window,'output_tokens':sum(r['meta']['completion_tokens'] for r in requests)}
    row['measured_output_tokens_per_s']=row['output_tokens']/window
    rows.append(row)
(root/'SUMMARY.json').write_text(json.dumps(sorted(rows,key=lambda r:r['concurrency']),indent=2))
print(json.dumps(rows,indent=2))
