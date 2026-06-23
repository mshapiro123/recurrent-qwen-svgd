"""Copy this into Colab to run the held-out ARC validation tail depth sweep."""

import base64
import json
import os
import time
import urllib.request

from google.colab import userdata

# Non-overlapping tail after the first 256 validation examples used in the
# initial loop-depth sweep. ARC-Challenge has only a small tail; ARC-Easy has a
# larger one. Keep loops to 1,2,3 because loop 4 was consistently damaging.
os.environ["STAGE5_DEPTH_SWEEP_LOOPS"] = "1,2,3"
os.environ["STAGE5_DEPTH_SWEEP_ARC_EASY_OFFSET"] = "256"
os.environ["STAGE5_DEPTH_SWEEP_ARC_EASY_LIMIT"] = "full"
os.environ["STAGE5_DEPTH_SWEEP_ARC_CHALLENGE_OFFSET"] = "256"
os.environ["STAGE5_DEPTH_SWEEP_ARC_CHALLENGE_LIMIT"] = "full"
os.environ["STAGE5_DEPTH_SWEEP_RUN_ID"] = time.strftime(
    "stage5_depth_sweep_arc_heldout_tail_loop123_%Y%m%d_%H%M%S"
)
os.environ["STAGE5_DEPTH_SWEEP_DISCONNECT"] = "0"

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "main"
CELL_PATH = "colab/STAGE5_DEPTH_SWEEP_BENCHMARK_CELL.py"

gh = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN")
assert gh, "Missing GH_TOKEN in Colab secrets."

url = (
    f"https://api.github.com/repos/{REPO}/contents/{CELL_PATH}"
    f"?ref={REF}&cache_bust={int(time.time())}"
)
req = urllib.request.Request(
    url,
    headers={
        "Authorization": f"Bearer {gh}",
        "Accept": "application/vnd.github+json",
        "Cache-Control": "no-cache",
    },
)

payload = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
code = base64.b64decode(payload["content"]).decode("utf-8")

print("Fetched:", CELL_PATH)
print("sha:", payload.get("sha"))
print("heldout_run_id:", os.environ["STAGE5_DEPTH_SWEEP_RUN_ID"])
assert "STAGE5_DEPTH_SWEEP_BENCHMARK_CELL_VERSION" in code
assert "STAGE5_BENCHMARK_ARC_EASY_OFFSET" in code

exec(compile(code, CELL_PATH, "exec"))
