# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU coverage for the K3 Slurm lifecycle and native AgentX protocol."""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "kimi_k3/tokenspeed"


STUB = r"""#!/usr/bin/env python3
import json, os, pathlib, signal, sys, time
root=pathlib.Path(os.environ['FAKE_ROOT']);name=pathlib.Path(sys.argv[0]).name;args=sys.argv[1:]
def mark(stage):
    (root/'stage').write_text(stage)
def block():
    (root/'child-pid').write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    while True: time.sleep(.02)
if name=='squeue':
    if '-s' in args:
        print('123.999 unrelated_service')
        if os.environ.get('FAKE_REGISTERED') and (root/'registered').exists():print('123.1 '+(root/'registered').read_text())
    elif args[-1]=='%u': print(os.environ['USER'])
    elif args[-1]=='%T': print('RUNNING')
    else: print('node[1-2]')
elif name=='scontrol':
    print('node1\nnode2' if 'hostnames' in args else 'Fake allocation')
elif name=='scancel':
    with (root/'cancelled').open('a') as f:f.write(' '.join(args)+'\n')
elif name=='nvidia-smi':
    if '--query-gpu=uuid' in args[0]: print('gpu1\ngpu2\ngpu3\ngpu4')
    elif '--query-gpu=name,uuid,driver_version' in args[0]: print('GB300, gpu, driver')
elif name=='srun':
    stage='preflight' if any('_preflight' in a for a in args) else 'server'
    mark(stage)
    if os.environ.get('FAKE_REGISTERED'):
        (root/'registered').write_text(next(a.split('=',1)[1] for a in args if a.startswith('--job-name=')))
    if os.environ.get('FAKE_FAIL')==stage:sys.exit(37)
    if os.environ.get('FAKE_BLOCK')==stage:block()
    i=args.index('bash');os.execvp('bash',args[i:])
elif name=='ts':
    with (root/'server-args.jsonl').open('a') as f:f.write(json.dumps(args)+'\n')
    (root/'ready').write_text(args[len(args)-1-args[::-1].index('--port')+1])
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
    try:
        while True:time.sleep(.02)
    finally:(root/'ready').unlink(missing_ok=True)
elif name=='curl':
    if not (root/'ready').exists():sys.exit(7)
    from urllib.parse import urlparse
    if args[-1].endswith('/metrics'):
        sys.exit(7 if os.environ.get('FAKE_BAD_ENDPOINT') else 0)
    if str(urlparse(args[-1]).port)!=(root/'ready').read_text():sys.exit(7)
    if args[-1].endswith('/v1/models'):print(json.dumps({'data':[{'id':'kimi-k3'}]}))
elif name=='python':
    assert args == ['-m', 'infx.results.agentic.process_agentic_result']
    assert os.environ['IS_MULTINODE']=='false'
    assert os.environ['TP']=='8'
    (pathlib.Path(os.environ['RESULT_DIR'])/'inferencex_result.json').write_text('{"num_gpus":8,"sentinel":123}')
else:raise RuntimeError(name)
"""


NATIVE_STUB = r"""
AIPERF_PYTHON="$CLIENT_ROOT/venv/bin/python"
resolve_trace_source() { TRACE_SOURCE_FLAG="--dataset native-trace"; }
build_replay_cmd() { REPLAY_CMD="aiperf profile --scenario agentx $TRACE_SOURCE_FLAG"; }
run_agentic_replay_and_write_outputs() {
    printf run > "$FAKE_ROOT/stage"
    if [[ "${FAKE_FAIL:-}" = run ]]; then return 38; fi
    if [[ "${FAKE_BLOCK:-}" = run ]]; then
        python3 - <<'PYTHON'
import os, pathlib, signal, sys, time
(pathlib.Path(os.environ['FAKE_ROOT'])/'child-pid').write_text(str(os.getpid()))
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
while True: time.sleep(.02)
PYTHON
    fi
    export REPLAY_CMD
    python3 - <<'PYTHON'
import json, os
from pathlib import Path
keys = ['EP_SIZE', 'DURATION', 'CONCURRENCY', 'AIPERF_WARMUP_REQUESTS_PER_LANE', 'AIPERF_EXTRA_INPUTS']
result = {key: os.environ[key] for key in keys}
result['command'] = os.environ['REPLAY_CMD']
result['loader'] = os.environ['WEKA_LOADER_OVERRIDE']
result['context'] = os.environ['MAX_MODEL_LEN']
(Path(os.environ['RESULT_DIR'])/'captured.json').write_text(json.dumps(result))
PYTHON
    if [[ "$SERVER_METRICS" = true && "${FAKE_MISSING_METRICS:-}" != 1 ]]; then
        python3 - <<'METRICS'
import json, os
from pathlib import Path
metrics = {}
for name, total in [('prompt_tokens',10000),('generation_tokens',370),('spec_decode_num_drafts',100),('spec_decode_num_accepted_tokens',270)]:
    metrics['tokenspeed:'+name]={'series':[{'endpoint_url':os.environ['AIPERF_SERVER_METRICS_URLS'],'stats':{'total':total}}]}
p=Path(os.environ['RESULT_DIR'])/'aiperf_artifacts';p.mkdir()
(p/'server_metrics_export.json').write_text('' if os.environ.get('FAKE_EMPTY_METRICS') else json.dumps({'metrics':metrics}))
METRICS
    fi
    write_agentic_result_json "$RESULT_DIR"
}
"""


class SlurmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        scripts = source / "test/agentic_benchmark/kimi_k3/tokenspeed"
        shutil.copytree(SCRIPTS, scripts)
        self.launcher = scripts / "agentx.slurm"
        tools = self.root / "bin"
        tools.mkdir()
        for name in (
            "squeue",
            "scontrol",
            "scancel",
            "nvidia-smi",
            "srun",
            "ts",
            "curl",
        ):
            path = tools / name
            path.write_text(STUB)
            path.chmod(0o755)
        client_bin = self.root / "native/venv/bin"
        client_bin.mkdir(parents=True)
        python = client_bin / "python"
        python.write_text(STUB)
        python.chmod(0o755)
        library = (
            self.root / "native/inferencex/inferencex-e2e/benchmarks/benchmark_lib.sh"
        )
        library.parent.mkdir(parents=True)
        library.write_text(NATIVE_STUB)
        self.env = {
            **os.environ,
            "PATH": f'{tools}:{os.environ["PATH"]}',
            "FAKE_ROOT": str(self.root),
            "SOURCE_ROOT": str(source),
            "CLIENT_ROOT": str(self.root / "native"),
            "INFERENCEX_WORKSPACE": str(library.parent.parent),
            "AGENTX_PROTOCOL": "weka-062126-256k",
            "ACCEPTANCE_MODE": "natural",
            "AL_TARGET": "",
            "SPEC_DECODING": "eagle3",
            "SPEC_STEPS": "3",
            "PRECISION": "nvfp4",
            "MAX_MODEL_LEN": "262144",
            "MAX_NUM_SEQS": "16",
            "SERVER_METRICS": "false",
            "CLIENT_PYTHON": str(python),
            "SLURM_JOB_ID": "123",
            "CONFIG_NAMES": "attn_tp8_moe_tp8 attn_tp8_moe_ep8",
            "SERVER_VENV": "",
            "CONTAINER_IMAGE": "fake.sqsh",
            "CONTAINER_MOUNTS": "/shared:/shared",
            "MODEL_NAME": "kimi-k3",
            "TOKENIZER_PATH": "/model",
            "CPUS_PER_TASK": "64",
            "CONCURRENCY": "1",
            "DURATION": "180",
            "BENCHMARK_GRACE_SECONDS": "30",
            "REQUEST_TIMEOUT_SECONDS": "21600",
            "CLIENT_SEED": "42",
            "AIPERF_WARMUP_REQUESTS_PER_LANE": "1",
            "AGENTIC_WARMUP_GRACE_PERIOD": "1800",
            "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS": "300",
            "AIPERF_FAILED_REQUEST_THRESHOLD": "0.1",
            "AIPERF_EXTRA_INPUTS": '{"ignore_eos":true}',
            "AIPERF_UNSAFE_OVERRIDE": "true",
            "API_PORT": "8000",
            "READINESS_PATH": "/readiness",
            "READINESS_TIMEOUT": "30",
            "CLIENT_TIMEOUT": "120",
            "RUN_ROOT": str(self.root / "runs"),
        }

    def launch(self):
        return subprocess.Popen(
            ["bash", str(self.launcher), "--model", "/weights/model with spaces"],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def wait_stage(self, name):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            p = self.root / "stage"
            if p.exists() and p.read_text() == name:
                return
            time.sleep(0.02)
        self.fail(f"Did not reach {name}")

    def test_sweep_preserves_native_settings_and_server_arguments(self):
        self.env["API_PORT"] = "18048"
        process = self.launch()
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stdout + stderr)
        for config, ep in (("attn_tp8_moe_tp8", "1"), ("attn_tp8_moe_ep8", "8")):
            run = self.root / "runs" / config
            captured = json.loads((run / "client/captured.json").read_text())
            self.assertEqual(captured["EP_SIZE"], ep)
            for key in (
                "DURATION",
                "CONCURRENCY",
                "AIPERF_WARMUP_REQUESTS_PER_LANE",
                "AIPERF_EXTRA_INPUTS",
            ):
                self.assertEqual(captured[key], self.env[key])
            self.assertIn("--benchmark-grace-period 30", captured["command"])
            self.assertIn("--random-seed 42", captured["command"])
            self.assertIn("--no-server-metrics", captured["command"])
            self.assertFalse(
                (run / "client/aiperf_artifacts/server_metrics_export.json").exists()
            )
            self.assertEqual((run / "client-exit-code.txt").read_text(), "0\n")
            self.assertEqual(
                json.loads((run / "client/inferencex_result.json").read_text()),
                {"num_gpus": 8, "sentinel": 123},
            )
            self.assertEqual(
                json.loads((run / "client/deployment.json").read_text())[
                    "physical_nodes"
                ],
                2,
            )
        commands = [
            json.loads(line)
            for line in (self.root / "server-args.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(commands), 2)
        for args in commands:
            model_index = max(i for i, arg in enumerate(args) if arg == "--model")
            self.assertEqual(args[model_index + 1], "/weights/model with spaces")
            self.assertEqual(args[-2:], ["--port", self.env["API_PORT"]])
        self.assertFalse((self.root / "cancelled").exists())

    def test_client_inputs_fail_before_startup(self):
        for key, value, diagnostic in (
            ("DURATION", "", "Missing DURATION"),
            (
                "TOKENIZER_PATH",
                "/model with spaces",
                "TOKENIZER_PATH must not contain whitespace",
            ),
            (
                "AIPERF_UNSAFE_OVERRIDE",
                "invalid",
                "AIPERF_UNSAFE_OVERRIDE must be true or false",
            ),
            ("CONFIG_NAMES", "   ", "CONFIG_NAMES must select at least one config"),
            (
                "RUN_ROOT",
                str(self.root / "run with spaces"),
                "RUN_ROOT must not contain whitespace",
            ),
        ):
            with self.subTest(key=key):
                result = subprocess.run(
                    ["bash", str(self.launcher)],
                    env={**self.env, key: value},
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse((self.root / "stage").exists())
                self.assertFalse((self.root / "runs").exists())

    def test_missing_paths_report_before_startup(self):
        library = (
            self.root / "native/inferencex/inferencex-e2e/benchmarks/benchmark_lib.sh"
        )
        config = self.launcher.parent / "configs/attn_tp8_moe_tp8.sh"
        for path, diagnostic in (
            (Path(self.env["CLIENT_PYTHON"]), "Client Python is not executable"),
            (library, "Missing InferenceX benchmark_lib.sh"),
            (config, "Missing server config"),
        ):
            with self.subTest(path=path):
                backup = path.with_suffix(".saved")
                path.rename(backup)
                try:
                    process = self.launch()
                    _, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 2, stderr)
                    self.assertIn(diagnostic, stderr)
                    self.assertFalse((self.root / "stage").exists())
                finally:
                    backup.rename(path)

    def test_unexported_server_venv_and_conflicting_port(self):
        self.env["CONFIG_NAMES"] = "attn_tp8_moe_tp8"
        self.env["API_PORT"] = "18048"
        result = subprocess.run(
            [
                "bash",
                "-c",
                'export -n SERVER_VENV; source "$1" --port 9000',
                "test-launcher",
                str(self.launcher),
            ],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = json.loads((self.root / "server-args.jsonl").read_text())
        self.assertIn(
            "9000", [args[i + 1] for i, arg in enumerate(args) if arg == "--port"]
        )
        self.assertEqual(args[-2:], ["--port", "18048"])
        command = (self.root / "runs/attn_tp8_moe_tp8/server-command.txt").read_text()
        self.assertIn("--port 9000", command)
        self.assertTrue(command.rstrip().endswith("--port 18048"))

    def test_preflight_failure_stops_before_server_and_next_config(self):
        self.env["FAKE_FAIL"] = "preflight"
        process = self.launch()
        _, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 37, stderr)
        self.assertFalse((self.root / "server-args.jsonl").exists())
        self.assertFalse((self.root / "runs/attn_tp8_moe_ep8").exists())

    def test_failed_client_preserves_exit_and_stops_sweep(self):
        self.env["FAKE_FAIL"] = "run"
        process = self.launch()
        _, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 38, stderr)
        self.assertFalse((self.root / "runs/attn_tp8_moe_ep8").exists())
        self.assertFalse((self.root / "ready").exists())

    def test_cleanup_cancels_only_owned_registered_steps(self):
        self.env["FAKE_REGISTERED"] = "1"
        self.env["CONFIG_NAMES"] = "attn_tp8_moe_tp8"
        process = self.launch()
        _, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 0, stderr)
        cancelled = (self.root / "cancelled").read_text().splitlines()
        self.assertTrue(cancelled)
        self.assertEqual(set(cancelled), {"123.1"})

    def test_interrupt_client_reaps_child_and_server(self):
        self.env["FAKE_BLOCK"] = "run"
        process = self.launch()
        try:
            self.wait_stage("run")
            time.sleep(0.1)
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=8)
            self.assertEqual(process.returncode, 143)
            child = int((self.root / "child-pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
            self.assertFalse((self.root / "ready").exists())
            self.assertFalse((self.root / "runs/attn_tp8_moe_ep8").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_native_bridge_rejects_missing_or_split_arguments(self):
        for key, value in (("DURATION", ""), ("TOKENIZER_PATH", "/model with spaces")):
            with self.subTest(key=key):
                result = subprocess.run(
                    ["bash", str(self.launcher.with_name("agentx.sh"))],
                    env={
                        **self.env,
                        "RESULT_DIR": str(self.root / "result"),
                        "EP_SIZE": "8",
                        "AIPERF_SERVER_URL": "http://node1:8000",
                        key: value,
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((self.root / "result").exists())

    def test_full_protocol_synthetic_metrics(self):
        self.env.update(
            CONFIG_NAMES="attn_tp8_moe_tp8",
            AGENTX_PROTOCOL="weka-062126-full",
            MAX_MODEL_LEN="1048576",
            MAX_NUM_SEQS="2",
            ACCEPTANCE_MODE="synthetic",
            AL_TARGET="3.7",
            SERVER_METRICS="true",
            METRICS_PORT="18090",
            PRECISION="fp4",
        )
        process = self.launch()
        stdout, stderr = process.communicate(timeout=25)
        self.assertEqual(process.returncode, 0, stdout + stderr)
        root = self.root / "runs/attn_tp8_moe_tp8/client"
        self.assertTrue(
            (root / "aiperf_artifacts/server_metrics_export.json").is_file()
        )
        self.assertFalse((root / "protocol.json").exists())
        args = json.loads((self.root / "server-args.jsonl").read_text().splitlines()[0])
        self.assertEqual(args[args.index("--synthetic-acceptance-length") + 1], "3.7")
        self.assertIn("--enable-metrics", args)
        captured = json.loads((root / "captured.json").read_text())
        self.assertEqual(captured["context"], "1048576")
        self.assertEqual(captured["loader"], "semianalysis_cc_traces_weka_062126")
        self.assertNotIn("--no-server-metrics", captured["command"])

    def test_required_metrics_failure_stops_run(self):
        for missing, bad_endpoint, empty in (
            ("1", "", ""),
            ("", "1", ""),
            ("", "", "1"),
        ):
            with self.subTest(missing=missing, bad_endpoint=bad_endpoint, empty=empty):
                self.env.update(
                    CONFIG_NAMES="attn_tp8_moe_tp8",
                    SERVER_METRICS="true",
                    METRICS_PORT="18090",
                    FAKE_MISSING_METRICS=missing,
                    FAKE_BAD_ENDPOINT=bad_endpoint,
                    FAKE_EMPTY_METRICS=empty,
                    RUN_ROOT=str(
                        self.root
                        / ("missing" if missing else "empty" if empty else "endpoint")
                    ),
                )
                process = self.launch()
                stdout, stderr = process.communicate(timeout=25)
                self.assertNotEqual(process.returncode, 0, stdout + stderr)
                self.assertFalse((self.root / "ready").exists())

    def test_protocol_mismatch_fails_before_gpu_startup(self):
        for override in (
            {"AL_TARGET": "nan", "ACCEPTANCE_MODE": "synthetic"},
            {"AGENTX_PROTOCOL": "weka-062126-full"},
            {"MAX_NUM_SEQS": "0"},
            {"AL_TARGET": "4.1", "ACCEPTANCE_MODE": "synthetic"},
            {"SPEC_DECODING": "unknown"},
            {"METRICS_PORT": "8000", "SERVER_METRICS": "true"},
        ):
            with self.subTest(override=override):
                result = subprocess.run(
                    ["bash", str(self.launcher)],
                    env={**self.env, **override},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse((self.root / "stage").exists())

    def test_interrupt_before_preflight_step_registration(self):
        self.env["FAKE_BLOCK"] = "preflight"
        process = self.launch()
        try:
            self.wait_stage("preflight")
            time.sleep(0.05)
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=8)
            self.assertEqual(process.returncode, 143)
            child = int((self.root / "child-pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
            self.assertFalse((self.root / "cancelled").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_interrupt_before_server_step_registration(self):
        self.env["FAKE_BLOCK"] = "server"
        process = self.launch()
        try:
            self.wait_stage("server")
            time.sleep(0.05)
            process.send_signal(signal.SIGINT)
            process.communicate(timeout=8)
            self.assertEqual(process.returncode, 130)
            child = int((self.root / "child-pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
            self.assertFalse((self.root / "cancelled").exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main()
