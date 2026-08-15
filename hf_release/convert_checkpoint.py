"""Convert one receipt-bound Paper One checkpoint into a release safetensors delta."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "hf_release" / "release_manifest.json"
ACTIVE_BRIDGE_KEYS = {
    "bridge.bridge_gate",
    "bridge.prelude_norm.weight",
    "bridge.prelude_norm.bias",
    "bridge.prelude_proj.weight",
    "bridge.state_proj.weight",
    "bridge.state_proj.bias",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_key(key: str) -> str:
    return f"backbone.{key[len('base_model.'):]}" if key.startswith("base_model.") else key


def validate_adapter_keys(state: dict[str, torch.Tensor]) -> None:
    invalid = sorted(
        key
        for key in state
        if key not in ACTIVE_BRIDGE_KEYS and ".lora_a." not in key and ".lora_b." not in key
    )
    if invalid:
        raise RuntimeError(f"Adapter release contains non-adapter/base tensors: {invalid}")


def convert(repo_name: str, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spec = manifest["repos"][repo_name]
    source_sha = sha256_file(checkpoint)
    if source_sha != spec["source_checkpoint_sha256"]:
        raise RuntimeError(
            f"Source checkpoint hash mismatch for {repo_name}: "
            f"expected={spec['source_checkpoint_sha256']} actual={source_sha}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_state = payload.get("trainable_state_dict")
    if not isinstance(raw_state, dict) or not raw_state:
        raise RuntimeError("Checkpoint has no nonempty trainable_state_dict")
    source_state = {str(key): value.detach().contiguous().cpu() for key, value in raw_state.items()}
    if spec["checkpoint_kind"] == "lora_adapter":
        validate_adapter_keys(source_state)
    state = {remap_key(key): value for key, value in source_state.items()}
    if len(state) != len(source_state):
        raise RuntimeError("Key remapping created a collision")

    tensor_count = len(state)
    total_parameters = sum(int(tensor.numel()) for tensor in state.values())
    if total_parameters != int(spec["expected_delta_parameters"]):
        raise RuntimeError(
            f"Parameter mismatch for {repo_name}: "
            f"expected={spec['expected_delta_parameters']} actual={total_parameters}"
        )
    lora_parameters = sum(
        int(tensor.numel())
        for key, tensor in state.items()
        if ".lora_a." in key or ".lora_b." in key
    )
    bridge_parameters = sum(
        int(tensor.numel()) for key, tensor in state.items() if key.startswith("bridge.")
    )
    if spec["checkpoint_kind"] == "lora_adapter":
        if lora_parameters != int(spec["expected_lora_parameters"]):
            raise RuntimeError(f"LoRA parameter mismatch: {lora_parameters}")
        if bridge_parameters != int(spec["expected_bridge_parameters"]):
            raise RuntimeError(f"Bridge parameter mismatch: {bridge_parameters}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = output_dir / "recurrent_delta.safetensors"
    save_file(
        state,
        str(safetensors_path),
        metadata={
            "format": "pt",
            "checkpoint_kind": str(spec["checkpoint_kind"]),
            "source_checkpoint_sha256": source_sha,
        },
    )
    dtype_counts = Counter(str(tensor.dtype).removeprefix("torch.") for tensor in state.values())
    receipt = {
        "schema_version": 1,
        "kind": "paper_one_hf_checkpoint_conversion",
        "status": "green",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_name": repo_name,
        "checkpoint_kind": spec["checkpoint_kind"],
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256_expected": spec["source_checkpoint_sha256"],
        "source_checkpoint_sha256": source_sha,
        "source_phase": payload.get("phase"),
        "source_step": payload.get("step"),
        "safetensors_file": safetensors_path.name,
        "safetensors_sha256": sha256_file(safetensors_path),
        "tensor_count": tensor_count,
        "total_parameters": total_parameters,
        "lora_parameters": lora_parameters,
        "bridge_parameters": bridge_parameters,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "key_transform": "base_model.* -> backbone.*; all other keys unchanged",
    }
    receipt_path = output_dir / "conversion_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = convert(args.repo, args.checkpoint, args.output_dir)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

