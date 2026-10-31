"""CPU checks for reporting and cache flush validation; no service restart."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from flush import validate_flush

class MetricsTest(unittest.TestCase):
    def test_flush_rejects_skipped_worker(self):
        good=dict(status='success',workers_flushed=2,total_workers=2,total_zmq_workers_skipped=0)
        validate_flush(good)
        for update in [dict(workers_flushed=1),dict(total_zmq_workers_skipped=1),dict(status='error')]:
            with self.assertRaises(ValueError): validate_flush(good | update)

    def test_output_window_excludes_prime_includes_between_turn_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for turn, first, last in [(0,1,2),(1,10,20),(2,30,40)]:
                folder=root/'results/measured-c1'/f'turn{turn}-100';folder.mkdir(parents=True)
                row=dict(success=True,events=[[first,1],[last,11]],ttft_s=2,tpot_ms=1000,meta=dict(completion_tokens=11))
                (folder/'request-0.json').write_text(json.dumps(row))
            subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),str(root)],check=True,capture_output=True)
            result=json.loads((root/'SUMMARY.json').read_text())[0]
            self.assertEqual(result['output_tokens'],22)
            self.assertEqual(result['output_window_s'],30)
            self.assertAlmostEqual(result['measured_output_tokens_per_s'],22/30)

if __name__=='__main__': unittest.main()
