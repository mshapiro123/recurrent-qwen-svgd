"""Copy this whole file into one Colab code cell to run the next A100 experiment.

This intentionally bypasses the generic Stage 5 router. It fetches and runs only
the direct max_loops=1 preservation probe.
"""

import base64
import json
import os
import time
import urllib.request

from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "main"
CELL_PATH = "colab/STAGE5_DIRECT_PRESERVATION_PROBE_CELL.py"

gh = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN")
assert gh, "Missing GH_TOKEN in Colab secrets."

# Real training experiment config. No planner/router.
os.environ["STAGE5_DIRECT_PRESERVE_DISCONNECT"] = "1"
os.environ["STAGE5_DIRECT_PRESERVE_MAX_STEPS"] = "75"
os.environ["STAGE5_DIRECT_PRESERVE_ARC_TRAIN_LIMIT"] = "512"
os.environ["STAGE5_DIRECT_PRESERVE_ARC_EVAL_LIMIT"] = "128"
os.environ["STAGE5_DIRECT_PRESERVE_LR"] = "5e-7"
os.environ["STAGE5_DIRECT_PRESERVE_DISTILL_WEIGHT"] = "1.0"
os.environ["STAGE5_DIRECT_PRESERVE_MIN_BASE_MARGIN"] = "1.0"

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
assert "STAGE5_DIRECT_PRESERVATION_PROBE_CELL_VERSION" in code
assert "colab/run_stage5_direct_preservation_probe.py" in code

exec(compile(code, CELL_PATH, "exec"))
