#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ "$ROLE" != gateway ]]; then
 for tool in nvcc ptxas cuobjdump; do
  [[ -x "$CUDA_HOME/bin/$tool" ]] || { echo "Missing CUDA tool: $CUDA_HOME/bin/$tool (use a complete CUDA toolkit)" >&2; exit 2; }
 done
fi
if [[ "$ROLE" == prefill ]]; then
 export PYTHONPATH="$P_DEEPEP_SOURCE:$P_FLASHINFER_SOURCE:$PYTHONPATH"
 export LD_PRELOAD="$P_NVSHMEM_LIB_DIR/libnvshmem_host.so.3"
 export LD_LIBRARY_PATH="$P_NVSHMEM_LIB_DIR:$LD_LIBRARY_PATH"
fi
"$VENV/bin/python" - <<'CHECK'
import importlib, importlib.metadata, json, os, pathlib, subprocess
modules=['torch','tokenspeed','tokenspeed_kernel','tokenspeed_engine','tokenspeed_scheduler','flashinfer','deep_ep']
if os.environ['ROLE']=='decode': modules.append('deep_gemm')
if os.environ['ROLE']=='gateway': modules=['smg']
result={'role':os.environ['ROLE'],'cuda_launch_blocking':os.getenv('CUDA_LAUNCH_BLOCKING'),'modules':{}}
for name in modules:
 module=importlib.import_module(name)
 result['modules'][name]=str(module.__file__)
if os.environ['ROLE']!='gateway':
 import torch, tokenspeed_engine
 assert torch.cuda.device_count()==8, 'Expected eight visible GPUs'
 assert torch.cuda.get_device_capability()==(12,0), 'This recipe targets SM120'
 binary=pathlib.Path(tokenspeed_engine.__file__).parent/'bin/ts-host'
 assert binary.is_file(), binary
 result['host_binary']=str(binary)
result['versions']={}
if os.environ['ROLE']=='decode':
 for package, expected in {
  'torch':'2.14.0+cu130', 'flashinfer-python':'0.7.2',
  'tokenspeed-deepep':'2.1.0.post20260810',
  'tokenspeed-mooncake':'0.3.13.post20261026',
 }.items():
  actual=importlib.metadata.version(package)
  if actual != expected: raise RuntimeError(f'{package}: expected {expected}, got {actual}')
  result['versions'][package]=actual
 import deep_ep
 dependency_root=pathlib.Path(deep_ep.__file__).parent.parent
 patch=pathlib.Path('deploy/dsv41_pd/patches/elastic-rdma-tuner.patch').resolve()
 subprocess.run(['patch','--batch','--dry-run','--reverse','-p0','-d',str(dependency_root),'-i',str(patch)],check=True,stdout=subprocess.DEVNULL)
 result['deep_ep_patch']='verified'
result['commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
print(json.dumps(result,indent=2))
CHECK
