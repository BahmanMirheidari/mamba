#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# install_mamba.sh
#
# Installs pre-built mamba-ssm and causal-conv1d wheels.
# No source compilation.
#
# Tested target:
#   Python 3.12
#   PyTorch 2.6
#   CUDA 12.x
#   Linux x86_64
# =============================================================================

echo
echo "============================================================"
echo " Mamba / causal-conv1d pre-built wheel installer"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Check Python
# -----------------------------------------------------------------------------

PY_TAG=$(python - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)

PY_VERSION=$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)

echo "Python version : $PY_VERSION"
echo "Python tag     : $PY_TAG"


# -----------------------------------------------------------------------------
# 2. Check PyTorch
# -----------------------------------------------------------------------------

TORCH_VERSION=$(python - <<'PY'
import torch
v = torch.__version__.split('+')[0]
print(v)
PY
)

TORCH_MAJOR_MINOR=$(python - <<'PY'
import torch
v = torch.__version__.split('+')[0]
parts = v.split('.')
print(f"{parts[0]}.{parts[1]}")
PY
)

echo "PyTorch        : $TORCH_VERSION"
echo "Torch version  : $TORCH_MAJOR_MINOR"


# -----------------------------------------------------------------------------
# 3. Check CUDA
# -----------------------------------------------------------------------------

CUDA_VERSION=$(python - <<'PY'
import torch

if torch.version.cuda is None:
    raise SystemExit("ERROR: PyTorch was not built with CUDA support.")

print(torch.version.cuda)
PY
)

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

echo "Torch CUDA     : $CUDA_VERSION"
echo "CUDA wheel tag : cu${CUDA_MAJOR}"


# -----------------------------------------------------------------------------
# 4. Check C++11 ABI
# -----------------------------------------------------------------------------

CXX11_ABI=$(python - <<'PY'
import torch
print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())
PY
)

if [[ "$CXX11_ABI" == "TRUE" ]]; then
    ABI_TAG="cxx11abiTRUE"
else
    ABI_TAG="cxx11abiFALSE"
fi

echo "C++11 ABI      : $CXX11_ABI"
echo "ABI wheel tag  : $ABI_TAG"


# -----------------------------------------------------------------------------
# 5. Check architecture
# -----------------------------------------------------------------------------

ARCH=$(uname -m)

case "$ARCH" in

    x86_64)
        PLATFORM_TAG="linux_x86_64"
        ;;

    aarch64)
        PLATFORM_TAG="linux_aarch64"
        ;;

    *)
        echo
        echo "ERROR: Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "Architecture   : $ARCH"
echo "Platform tag   : $PLATFORM_TAG"


# -----------------------------------------------------------------------------
# 6. Select package versions
#
# These versions work with the user's current stack:
#
# Python 3.12
# PyTorch 2.6
# CUDA 12.x
#
# -----------------------------------------------------------------------------

if [[ "$TORCH_MAJOR_MINOR" == "2.6" && "$PY_TAG" == "cp312" && "$CUDA_MAJOR" == "12" ]]; then

    CAUSAL_VERSION="1.7.0"
    MAMBA_VERSION="2.3.2.post1"

else

    echo
    echo "============================================================"
    echo " WARNING"
    echo "============================================================"
    echo
    echo "This script currently has a tested wheel configuration for:"
    echo
    echo "  Python 3.12"
    echo "  PyTorch 2.6"
    echo "  CUDA 12.x"
    echo
    echo "Detected:"
    echo "  Python : $PY_VERSION"
    echo "  Torch  : $TORCH_VERSION"
    echo "  CUDA   : $CUDA_VERSION"
    echo
    echo "No package versions have been selected."
    echo "This prevents accidental source compilation."
    echo
    exit 1
fi


# -----------------------------------------------------------------------------
# 7. Construct wheel names
# -----------------------------------------------------------------------------

WHEEL_TAG="cu${CUDA_MAJOR}torch${TORCH_MAJOR_MINOR}${ABI_TAG}-${PY_TAG}-${PY_TAG}-${PLATFORM_TAG}"

CAUSAL_WHEEL="causal_conv1d-${CAUSAL_VERSION}+${WHEEL_TAG}.whl"

MAMBA_WHEEL="mamba_ssm-${MAMBA_VERSION}+${WHEEL_TAG}.whl"

CAUSAL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_VERSION}/${CAUSAL_WHEEL}"

MAMBA_URL="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VERSION}/${MAMBA_WHEEL}"


# -----------------------------------------------------------------------------
# 8. Display what will be installed
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo " Wheel selection"
echo "============================================================"

echo
echo "causal-conv1d:"
echo "  Version : $CAUSAL_VERSION"
echo "  Wheel   : $CAUSAL_WHEEL"
echo

echo "mamba-ssm:"
echo "  Version : $MAMBA_VERSION"
echo "  Wheel   : $MAMBA_WHEEL"
echo

echo "============================================================"


# -----------------------------------------------------------------------------
# 9. Create temporary directory
# -----------------------------------------------------------------------------

TMP_DIR="${TMPDIR:-/tmp}/mamba_install_$$"

mkdir -p "$TMP_DIR"

trap 'rm -rf "$TMP_DIR"' EXIT

echo
echo "Temporary directory:"
echo "  $TMP_DIR"


# -----------------------------------------------------------------------------
# 10. Download causal-conv1d
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo " Downloading causal-conv1d"
echo "============================================================"

if command -v wget >/dev/null 2>&1; then

    wget -q --show-progress \
        "$CAUSAL_URL" \
        -O "$TMP_DIR/$CAUSAL_WHEEL"

elif command -v curl >/dev/null 2>&1; then

    curl -L --fail \
        "$CAUSAL_URL" \
        -o "$TMP_DIR/$CAUSAL_WHEEL"

else

    echo "ERROR: Neither wget nor curl is available."
    exit 1

fi


# -----------------------------------------------------------------------------
# 11. Download Mamba
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo " Downloading mamba-ssm"
echo "============================================================"

if command -v wget >/dev/null 2>&1; then

    wget -q --show-progress \
        "$MAMBA_URL" \
        -O "$TMP_DIR/$MAMBA_WHEEL"

else

    curl -L --fail \
        "$MAMBA_URL" \
        -o "$TMP_DIR/$MAMBA_WHEEL"

fi


# -----------------------------------------------------------------------------
# 12. Uninstall previous versions
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo " Removing previous installations"
echo "============================================================"

python -m pip uninstall -y \
    mamba-ssm \
    causal-conv1d \
    2>/dev/null || true


# -----------------------------------------------------------------------------
# 13. Install wheels
#
# --no-index prevents pip from looking elsewhere.
# --no-deps prevents pip from replacing your existing PyTorch.
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo " Installing causal-conv1d"
echo "============================================================"

python -m pip install \
    --no-index \
    --no-deps \
    "$TMP_DIR/$CAUSAL_WHEEL"


echo
echo "============================================================"
echo " Installing mamba-ssm"
echo "============================================================"

python -m pip install \
    --no-index \
    --no-deps \
    "$TMP_DIR/$MAMBA_WHEEL"


# -----------------------------------------------------------------------------
# 14. Verify installation
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo " Verifying installation"
echo "============================================================"

python - <<'PY'

import torch

print()
print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()

try:
    import causal_conv1d
    print("causal-conv1d: OK")
except Exception as e:
    print("causal-conv1d: FAILED")
    print(e)
    raise

try:
    from mamba_ssm import Mamba
    print("mamba-ssm: OK")
except Exception as e:
    print("mamba-ssm: FAILED")
    print(e)
    raise

print()
print("============================================================")
print(" MAMBA INSTALLATION SUCCESSFUL")
print("============================================================")

PY