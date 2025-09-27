# PP-OCRv5 Model Export Pipeline
# Streamlined commands for exporting PaddleOCR models to ONNX format

# Install Python dependencies
deps:
    @echo "Installing Python dependencies..."
    pip install -r requirements.txt
    @echo "✓ Dependencies installed"

# Download all pretrained models
download-all:
    python scripts/download.py all

# Download specific model
download-mobile-det:
    python scripts/download.py PP-OCRv5_mobile_det

download-mobile-rec:
    python scripts/download.py PP-OCRv5_mobile_rec

download-server-det:
    python scripts/download.py PP-OCRv5_server_det

download-server-rec:
    python scripts/download.py PP-OCRv5_server_rec

# Setup PaddleOCR repository (one-time setup)
setup: deps download-all
    @echo "✓ Setup complete"

# Export PP-OCRv5 server detection model
export-server-det:
    @echo "Exporting PP-OCRv5 server detection model..."
    python scripts/export.py --model PP-OCRv5_server_det --output onnx_models/PP-OCRv5_server_det.onnx

# Export PP-OCRv5 server recognition model
export-server-rec:
    @echo "Exporting PP-OCRv5 server recognition model..."
    python scripts/export.py --model PP-OCRv5_server_rec --output onnx_models/PP-OCRv5_server_rec.onnx

# Export PP-OCRv5 mobile detection model
export-mobile-det:
    @echo "Exporting PP-OCRv5 mobile detection model..."
    python scripts/export.py --model PP-OCRv5_mobile_det --output onnx_models/PP-OCRv5_mobile_det.onnx

# Export PP-OCRv5 mobile recognition model
export-mobile-rec:
    @echo "Exporting PP-OCRv5 mobile recognition model..."
    python scripts/export.py --model PP-OCRv5_mobile_rec --output onnx_models/PP-OCRv5_mobile_rec.onnx

# Export all models in parallel (requires GNU parallel)
export-all-parallel:
    @echo "Exporting all models in parallel..."
    @if command -v parallel >/dev/null 2>&1; then \
        parallel -j4 --bar just ::: export-server-det export-server-rec export-mobile-det export-mobile-rec; \
    else \
        echo "GNU parallel not found, running sequentially..."; \
        just export-all; \
    fi

# Export all models sequentially
export-all: export-server-det export-server-rec export-mobile-det export-mobile-rec
    @echo "✓ All models exported successfully!"
    @echo "Output files:"
    @ls -lh onnx_models/*.onnx

# Validate specific ONNX model
validate-mobile-det:
    @echo "Validating PP-OCRv5 mobile detection model..."
    @if [ -f "onnx_models/PP-OCRv5_mobile_det.onnx" ]; then \
        python3 -c "import onnx; onnx.checker.check_model(onnx.load('onnx_models/PP-OCRv5_mobile_det.onnx')); print('✅ PP-OCRv5_mobile_det.onnx - Valid')"; \
    else \
        echo "❌ PP-OCRv5_mobile_det.onnx not found"; \
        exit 1; \
    fi

validate-mobile-rec:
    @echo "Validating PP-OCRv5 mobile recognition model..."
    @if [ -f "onnx_models/PP-OCRv5_mobile_rec.onnx" ]; then \
        python3 -c "import onnx; onnx.checker.check_model(onnx.load('onnx_models/PP-OCRv5_mobile_rec.onnx')); print('✅ PP-OCRv5_mobile_rec.onnx - Valid')"; \
    else \
        echo "❌ PP-OCRv5_mobile_rec.onnx not found"; \
        exit 1; \
    fi

validate-server-det:
    @echo "Validating PP-OCRv5 server detection model..."
    @if [ -f "onnx_models/PP-OCRv5_server_det.onnx" ]; then \
        python3 -c "import onnx; onnx.checker.check_model(onnx.load('onnx_models/PP-OCRv5_server_det.onnx')); print('✅ PP-OCRv5_server_det.onnx - Valid')"; \
    else \
        echo "❌ PP-OCRv5_server_det.onnx not found"; \
        exit 1; \
    fi

validate-server-rec:
    @echo "Validating PP-OCRv5 server recognition model..."
    @if [ -f "onnx_models/PP-OCRv5_server_rec.onnx" ]; then \
        python3 -c "import onnx; onnx.checker.check_model(onnx.load('onnx_models/PP-OCRv5_server_rec.onnx')); print('✅ PP-OCRv5_server_rec.onnx - Valid')"; \
    else \
        echo "❌ PP-OCRv5_server_rec.onnx not found"; \
        exit 1; \
    fi

# Validate all exported ONNX models
validate-all:
    @echo "Validating ONNX models..."
    @if [ -d "onnx_models" ]; then \
        for model in onnx_models/*.onnx; do \
            if [ -f "$model" ]; then \
                echo "Checking $(basename $model)..."; \
                python3 -c "import onnx; onnx.checker.check_model(onnx.load('$model')); print('✅ $(basename $model) - Valid')" || echo "❌ $(basename $model) - Invalid"; \
            fi; \
        done; \
    else \
        echo "❌ No onnx_models directory found"; \
        exit 1; \
    fi

# Show model information
info:
    @echo "PP-OCRv5 Model Export Pipeline"
    @echo "=============================="
    @echo ""
    @echo "Available models:"
    @echo "  • PP-OCRv5_server_det  - Server detection model (high accuracy)"
    @echo "  • PP-OCRv5_server_rec  - Server recognition model (high accuracy)"
    @echo "  • PP-OCRv5_mobile_det  - Mobile detection model (optimized)"
    @echo "  • PP-OCRv5_mobile_rec  - Mobile recognition model (optimized)"
    @echo ""
    @echo "Usage:"
    @echo "  just deps           # Install dependencies"
    @echo "  just setup          # Setup repository and download models"
    @echo "  just export-all     # Export all models"
    @echo "  just validate       # Validate exported ONNX models"
    @echo ""
    @if [ -d "onnx_models" ]; then \
        echo "Current exported models:"; \
        ls -lh onnx_models/*.onnx 2>/dev/null || echo "  No ONNX models found"; \
    else \
        echo "No models exported yet. Run 'just export-all' to get started."; \
    fi

# Clean intermediate files (keep ONNX models)
clean:
    @echo "Cleaning intermediate files..."
    rm -rf official_pipeline/exports
    @echo "✓ Intermediate files cleaned"

# Full clean (including ONNX models and downloaded dependencies)
clean-all:
    @echo "Performing full cleanup..."
    rm -rf official_pipeline onnx_models
    @echo "✓ Full cleanup complete"

# Quick start: setup, export all models, and validate
quick-start: deps setup export-all validate-all
    @echo "🎉 Quick start complete! All models exported and validated."

# Default target (show help)
default: info