import base64, json, os, urllib.request
from google.colab import userdata

REPO = "mshapiro123/recurrent-qwen-svgd"
REF = "main"
LAUNCHER_PATH = "colab/STAGE5_ARC_MIX_RECOVERY_CELL.py"

def secret(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return value
    return None

GH_TOKEN = secret("GH_TOKEN", "GITHUB_TOKEN")
assert GH_TOKEN, "Missing GH_TOKEN or GITHUB_TOKEN in Colab secrets."

url = f"https://api.github.com/repos/{REPO}/contents/{LAUNCHER_PATH}?ref={REF}"
request = urllib.request.Request(
    url,
    headers={
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    },
)
with urllib.request.urlopen(request) as response:
    payload = json.loads(response.read().decode("utf-8"))

code = base64.b64decode(payload["content"]).decode("utf-8")
required_markers = [
    "colab/run_stage5_arc_mix_recovery_once.py",
    "colab/check_stage5_a100_go_no_go.py",
    "arc_mix_response_w01_lr2e6",
    "STAGE5_ARC_MIX_ONCE_AUTO_DISCONNECT",
]
missing = [marker for marker in required_markers if marker not in code]
assert not missing, f"Fetched launcher is missing expected safety markers: {missing}"

print(f"Fetched {LAUNCHER_PATH} from {REPO}@{REF} sha={payload.get('sha')}", flush=True)
exec(compile(code, LAUNCHER_PATH, "exec"))
