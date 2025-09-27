# PP-OCRv5 ONNX Export Pipeline

Streamlined pipeline for exporting PP-OCRv5 models to ONNX format using the official PaddleOCR method. Perfect for CI/CD workflows and automated releases.

## 🚀 Quick Start

```bash
# Prerequisites: Install just command runner
# curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash

# Complete pipeline - one command does it all
just quick-start

# Or step by step
just deps        # Install Python dependencies
just export-all  # Export all 4 models to onnx_models/
```

## Features

- ✅ **Official Method**: Uses PaddleOCR's official `tools/export_model.py`
- ✅ **Auto Setup**: Downloads PaddleOCR repo and pretrained models automatically
- ✅ **Fixed Issues**: Resolves path and JSON format problems from original scripts
- ✅ **CI/CD Ready**: GitHub Action included for automated releases
- ✅ **Proper Naming**: Preserves original PP-OCRv5 model names

## Available Models

| Model | Type | Size | Description |
|-------|------|------|-------------|
| PP-OCRv5_server_det | Detection | ~88MB | High accuracy detection model |
| PP-OCRv5_server_rec | Recognition | ~84MB | High accuracy recognition model |
| PP-OCRv5_mobile_det | Detection | ~5MB | Optimized mobile detection |
| PP-OCRv5_mobile_rec | Recognition | ~16MB | Optimized mobile recognition |

## Commands

```bash
just info           # Show pipeline information
just deps           # Install Python dependencies
just setup          # Setup PaddleOCR repository and download models
just export-all     # Export all models
just validate       # Validate exported ONNX models
just clean          # Clean intermediate files
just clean-all      # Full cleanup
just quick-start    # Complete setup and export
```

## Individual Model Export

```bash
just export-server-det    # PP-OCRv5_server_det.onnx
just export-server-rec    # PP-OCRv5_server_rec.onnx
just export-mobile-det    # PP-OCRv5_mobile_det.onnx
just export-mobile-rec    # PP-OCRv5_mobile_rec.onnx
```

## GitHub Actions

The repository includes a GitHub Action that:

- Automatically exports models on tag pushes (`v*`)
- Can be triggered manually with `workflow_dispatch`
- Uploads models as release artifacts
- Creates GitHub releases with model files

### Manual Trigger

1. Go to Actions tab in GitHub
2. Select "Export PP-OCRv5 Models"
3. Click "Run workflow"
4. Optionally specify a release tag

### Automatic Release

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Using Exported Models

```python
import onnxruntime as ort
import numpy as np

# Load detection model
det_session = ort.InferenceSession('onnx_models/PP-OCRv5_server_det.onnx')

# Load recognition model
rec_session = ort.InferenceSession('onnx_models/PP-OCRv5_server_rec.onnx')

# Run inference
det_result = det_session.run(None, {"x": input_image})
rec_result = rec_session.run(None, {"x": cropped_text})
```

## File Structure

```
├── onnx_models/                    # Exported ONNX models
│   ├── PP-OCRv5_server_det.onnx
│   ├── PP-OCRv5_server_rec.onnx
│   ├── PP-OCRv5_mobile_det.onnx
│   └── PP-OCRv5_mobile_rec.onnx
├── scripts/
│   └── export_official_fixed.py   # Fixed export script
├── .github/workflows/
│   └── export-models.yml          # GitHub Action
├── justfile                        # Command definitions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Requirements

- Python 3.8+
- [just](https://github.com/casey/just) command runner
- Git

Dependencies are automatically installed with `just deps`.

## Technical Details

- **Export Method**: Official PaddleOCR `tools/export_model.py`
- **Format Support**: Handles PaddlePaddle 3.x JSON export format
- **Path Handling**: Uses absolute paths to avoid construction issues
- **ONNX Version**: Auto-selected opset (typically 14)
- **Optimization**: Includes onnxoptimizer for model optimization

## License

This project follows the same license as the original PaddleOCR project.

## Credits

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Original OCR framework
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - Deep learning framework
- [paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX) - Model conversion tool