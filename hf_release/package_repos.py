"""Assemble the three local Hugging Face repository directories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "hf_release"
ARCHITECTURE_FIGURES = (
    ROOT / "docs/figures/figure1_architecture_comparison.svg",
    ROOT / "docs/figures/figure1_architecture_comparison.png",
)
LFS_ATTRIBUTES = """*.safetensors filter=lfs diff=lfs merge=lfs -text
*.json text eol=lf
*.md text eol=lf
*.py text eol=lf
*.svg text eol=lf
*.png -text
LICENSE text eol=lf
"""


def config_for(spec: dict[str, object]) -> dict[str, object]:
    return {
        "architectures": ["RecurrentQwenForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_recurrent_qwen.RecurrentQwenConfig",
            "AutoModelForCausalLM": "modeling_recurrent_qwen.RecurrentQwenForCausalLM",
        },
        "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "base_model_revision": "main",
        "checkpoint_kind": spec["checkpoint_kind"],
        "delta_filename": "recurrent_delta.safetensors",
        "lora_alpha": 32,
        "lora_rank": 16,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "model_type": "recurrent_qwen",
        "prelude_end": 6,
        "recurrent_end": 18,
        "transformers_version": "4.53.0",
        "use_cache": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-missing-weights", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((RELEASE / "release_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (RELEASE / "verification_assets/verification_manifest.json").read_text(encoding="utf-8")
    )["repos"]

    for repo_name, spec in manifest["repos"].items():
        destination = args.output_root / repo_name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / spec["card"], destination / "README.md")
        shutil.copy2(RELEASE / "LICENSE", destination / "LICENSE")
        for figure in ARCHITECTURE_FIGURES:
            shutil.copy2(figure, destination / figure.name)
        shutil.copy2(RELEASE / "configuration_recurrent_qwen.py", destination / "configuration_recurrent_qwen.py")
        shutil.copy2(RELEASE / "modeling_recurrent_qwen.py", destination / "modeling_recurrent_qwen.py")
        (destination / ".gitattributes").write_text(LFS_ATTRIBUTES, encoding="utf-8", newline="\n")
        (destination / "config.json").write_text(
            json.dumps(config_for(spec), indent=2) + "\n", encoding="utf-8", newline="\n"
        )

        verification_spec = verification[repo_name]
        source_data = ROOT / verification_spec["verification_data"]
        shutil.copy2(source_data, destination / "verification_subset.jsonl")
        (destination / "verification_spec.json").write_text(
            json.dumps(verification_spec, indent=2) + "\n", encoding="utf-8", newline="\n"
        )

        converted = args.converted_root / repo_name
        required = ["recurrent_delta.safetensors", "conversion_receipt.json"]
        missing = [name for name in required if not (converted / name).is_file()]
        if missing and not args.allow_missing_weights:
            raise FileNotFoundError(f"{repo_name} missing converted files: {missing}")
        for name in required:
            if (converted / name).is_file():
                shutil.copy2(converted / name, destination / name)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

