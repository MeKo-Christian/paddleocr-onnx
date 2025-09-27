#!/usr/bin/env python3
"""
Streamlined PP-OCRv5 export script for CI/CD workflows.
Fixes path issues and handles the new PaddlePaddle 3.x JSON format.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


# PP-OCRv5 model configurations
PPOCRV5_CONFIGS = {
    "PP-OCRv5_server_det": {
        "config": "configs/det/PP-OCRv5/PP-OCRv5_server_det.yml",
        "description": "PP-OCRv5 server detection model",
        "model_type": "detection",
        "pretrained_url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams"
    },
    "PP-OCRv5_mobile_det": {
        "config": "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml",
        "description": "PP-OCRv5 mobile detection model",
        "model_type": "detection",
        "pretrained_url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams"
    },
    "PP-OCRv5_server_rec": {
        "config": "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
        "description": "PP-OCRv5 server recognition model",
        "model_type": "recognition",
        "pretrained_url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams"
    },
    "PP-OCRv5_mobile_rec": {
        "config": "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
        "description": "PP-OCRv5 mobile recognition model",
        "model_type": "recognition",
        "pretrained_url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"
    }
}


def download_file(url: str, local_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        print(f"Downloading {local_path.name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"✓ Downloaded: {local_path}")
        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def setup_paddleocr_repo(repo_path: Path) -> bool:
    """Setup PaddleOCR repository."""
    if repo_path.exists() and (repo_path / "tools" / "export_model.py").exists():
        print("✓ PaddleOCR repository already exists")
        return True

    print("Setting up PaddleOCR repository...")
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git", str(repo_path)]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("✓ PaddleOCR repository cloned")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone repository: {e}")
        return False


def download_pretrained_model(model_name: str, models_dir: Path) -> Optional[Path]:
    """Download pretrained model."""
    if model_name not in PPOCRV5_CONFIGS:
        print(f"Unknown model: {model_name}")
        return None

    config = PPOCRV5_CONFIGS[model_name]
    filename = Path(config["pretrained_url"]).name
    local_path = models_dir / "pretrained" / filename

    if local_path.exists():
        print(f"✓ Model already exists: {local_path}")
        return local_path

    success = download_file(config["pretrained_url"], local_path)
    return local_path if success else None


def export_paddle_model(repo_path: Path, model_name: str, pretrained_path: Path, output_dir: Path) -> bool:
    """Export PaddlePaddle model using official tools."""
    if model_name not in PPOCRV5_CONFIGS:
        print(f"Unknown model: {model_name}")
        return False

    config = PPOCRV5_CONFIGS[model_name]
    config_path = repo_path / config["config"]

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use absolute paths to avoid path construction issues
    abs_config_path = config_path.resolve()
    abs_pretrained_path = pretrained_path.resolve()
    abs_output_dir = output_dir.resolve()

    cmd = [
        "python", "tools/export_model.py",
        "-c", str(abs_config_path),
        "-o", f"Global.pretrained_model={abs_pretrained_path}",
        f"Global.save_inference_dir={abs_output_dir}"
    ]

    print(f"Exporting {model_name}...")
    print(f"Working directory: {repo_path}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Model export successful")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Model export failed: {e}")
        print(f"stderr: {e.stderr}")
        print(f"stdout: {e.stdout}")
        return False


def convert_to_onnx(inference_dir: Path, onnx_output: Path, opset_version: int = 11) -> bool:
    """Convert exported model to ONNX format."""

    # Check for new JSON format first (PaddlePaddle 3.x)
    json_file = inference_dir / "inference.json"
    params_file = inference_dir / "inference.pdiparams"

    if json_file.exists() and params_file.exists():
        print(f"Using new JSON format: {json_file}")
        cmd = [
            "paddle2onnx",
            "--model_dir", str(inference_dir),
            "--model_filename", "inference.json",
            "--params_filename", "inference.pdiparams",
            "--save_file", str(onnx_output),
            "--opset_version", str(opset_version),
            "--enable_auto_update_opset", "True",
            "--enable_onnx_checker", "True"
        ]
    else:
        # Fallback to old format
        pdmodel_file = None
        pdiparams_file = None

        for file_path in inference_dir.iterdir():
            if file_path.suffix == ".pdmodel":
                pdmodel_file = file_path
            elif file_path.suffix == ".pdiparams":
                pdiparams_file = file_path

        if not pdmodel_file or not pdiparams_file:
            print(f"✗ No valid model files found in {inference_dir}")
            return False

        cmd = [
            "paddle2onnx",
            "--model_dir", str(inference_dir),
            "--model_filename", pdmodel_file.name,
            "--params_filename", pdiparams_file.name,
            "--save_file", str(onnx_output),
            "--opset_version", str(opset_version),
            "--enable_auto_update_opset", "True",
            "--enable_onnx_checker", "True"
        ]

    onnx_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting to ONNX: {onnx_output}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ ONNX conversion successful: {onnx_output}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ ONNX conversion failed: {e}")
        print(f"stderr: {e.stderr}")
        return False


def export_single_model(model_name: str, output_path: Path, base_dir: Path = Path("official_pipeline")) -> bool:
    """Export a single model end-to-end."""
    print(f"\n{'='*60}")
    print(f"Exporting {model_name}")
    print(f"{'='*60}")

    # Setup paths
    repo_path = base_dir / "PaddleOCR"
    models_dir = base_dir / "models"
    exports_dir = base_dir / "exports"

    # Step 1: Setup repository
    if not setup_paddleocr_repo(repo_path):
        return False

    # Step 2: Download pretrained model
    pretrained_path = download_pretrained_model(model_name, models_dir)
    if not pretrained_path:
        return False

    # Step 3: Export PaddlePaddle model
    export_dir = exports_dir / model_name
    if not export_paddle_model(repo_path, model_name, pretrained_path, export_dir):
        return False

    # Step 4: Convert to ONNX
    if not convert_to_onnx(export_dir, output_path):
        return False

    print(f"✓ {model_name} exported successfully to {output_path}")
    return True


def download_all_models(base_dir: Path = Path("official_pipeline")) -> bool:
    """Download all pretrained models without exporting."""
    print("Downloading all pretrained models...")

    repo_path = base_dir / "PaddleOCR"
    models_dir = base_dir / "models"

    # Setup repository first
    if not setup_paddleocr_repo(repo_path):
        return False

    success = True
    for model_name in PPOCRV5_CONFIGS.keys():
        print(f"\nDownloading {model_name}...")
        pretrained_path = download_pretrained_model(model_name, models_dir)
        if not pretrained_path:
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Streamlined PP-OCRv5 model export tool"
    )

    parser.add_argument("--model", choices=list(PPOCRV5_CONFIGS.keys()),
                        help="Model to export")
    parser.add_argument("--output", type=Path,
                        help="Output ONNX file path")
    parser.add_argument("--base-dir", type=Path, default=Path("official_pipeline"),
                        help="Base working directory")
    parser.add_argument("--download-all", action="store_true",
                        help="Download all pretrained models")
    parser.add_argument("--opset-version", type=int, default=11,
                        help="ONNX opset version")

    args = parser.parse_args()

    if args.download_all:
        success = download_all_models(args.base_dir)
        sys.exit(0 if success else 1)

    if not args.model or not args.output:
        print("Error: --model and --output are required")
        print("\nAvailable models:")
        for name, config in PPOCRV5_CONFIGS.items():
            print(f"  {name}: {config['description']}")
        sys.exit(1)

    success = export_single_model(args.model, args.output, args.base_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()