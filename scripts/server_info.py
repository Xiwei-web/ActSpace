"""Server info -> JSON (stdout). Run with the python of the model server's environment, so the versions of that environment are recorded.

    <env>/bin/python scripts/server_info.py <model> <GPUs (comma-separated)> <tensor parallelism> <port> <vllm|cambrian> <start timestamp> <ready JSON> <server args…>
"""
import json
import platform
import socket
import subprocess
import sys
import time

m, dev, tp, port, kind, t0, ready = sys.argv[1:8]
info = {"model": m, "kind": kind, "host": socket.gethostname(), "gpus": dev.split(","), "n_gpus": len(dev.split(",")),
        "tensor_parallel_size": int(tp), "port": int(port), "args": sys.argv[8:], "python": platform.python_version(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(int(t0))), "ready": json.loads(ready)}
for mod in ("vllm", "torch", "transformers"):
    try:
        info[mod] = __import__(mod).__version__
    except Exception as e:  # noqa: BLE001
        info[mod] = f"unavailable: {e}"
try:
    import torch
    info["cuda_runtime"] = torch.version.cuda
    info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception:  # noqa: BLE001
    pass
q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
info["nvidia_smi"] = q.stdout.strip().splitlines()
print(json.dumps(info, ensure_ascii=False, indent=1))
