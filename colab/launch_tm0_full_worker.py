"""Launch the long TM-0 GPU worker without tying it to a CLI websocket."""

from pathlib import Path
import subprocess

log = Path("/content/tm0_full_worker.log").open("ab", buffering=0)
process = subprocess.Popen(
    ["/usr/bin/python3", "-u", "/content/run_tm0_full_worker.py"],
    cwd="/content/tm0_repo",
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
Path("/content/tm0_full_worker.pid").write_text(str(process.pid) + "\n")
print(f"tm0_full_worker_pid={process.pid}")
