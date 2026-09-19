#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# install_mamba.sh
#
# Installs pre-built Mamba SSM and causal-conv1d wheels from GitHub.
# Never compiles from source.
#
# Fixes:
#  - Python urllib for downloads (no curl URL-encoding problems).
#  - Release JSON written to a file (no env var size limit).
#  - Informational output goes to stderr; only the URL reaches stdout.
#
# Override versions:
#   CAUSAL_VERSION=1.5.0.post8 MAMBA_VERSION=2.2.4 ./install_mamba.sh
# =============================================================================

CAUSAL_VERSION="${CAUSAL_VERSION:-1.5.0.post8}"
MAMBA_VERSION="${MAMBA_VERSION:-2.2.4}"

CAUSAL_REPO="Dao-AILab/causal-conv1d"
MAMBA_REPO="state-spaces/mamba"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mamba_install.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

die() { echo; echo "ERROR: $*" >&2; exit 1; }

echo
echo "============================================================"
echo " Mamba pre-built wheel installer"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Required commands
# -----------------------------------------------------------------------------
command -v python >/dev/null 2>&1 || die "python not found"
PYTHON="$(command -v python)"
echo "Python executable: $PYTHON"

# -----------------------------------------------------------------------------
# 2. Detect environment
# -----------------------------------------------------------------------------
readarray -t ENV_INFO < <("$PYTHON" - <<'PY'
import sys, platform, torch
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print(torch.__version__.split("+")[0])
print(torch.version.cuda or "none")
print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())
print(platform.machine())
PY
)

PY_TAG="${ENV_INFO[0]}"
PY_VERSION="${ENV_INFO[1]}"
TORCH_VERSION="${ENV_INFO[2]}"
CUDA_VERSION="${ENV_INFO[3]}"
CXX11_ABI="${ENV_INFO[4]}"
ARCH="${ENV_INFO[5]}"

[[ "$CUDA_VERSION" != "none" ]] || die "PyTorch has no CUDA support"

CUDA_MAJOR="${CUDA_VERSION%%.*}"
TORCH_MAJOR_MINOR="$("$PYTHON" -c 'import torch; p=torch.__version__.split("+")[0].split("."); print(f"{p[0]}.{p[1]}")')"

if [[ "$CXX11_ABI" == "TRUE" ]]; then ABI_TAG="cxx11abiTRUE"; else ABI_TAG="cxx11abiFALSE"; fi

case "$ARCH" in
    x86_64)  PLATFORM_TAG="linux_x86_64"  ;;
    aarch64) PLATFORM_TAG="linux_aarch64" ;;
    *)       die "Unsupported architecture: $ARCH" ;;
esac

WHEEL_TAG="cu${CUDA_MAJOR}torch${TORCH_MAJOR_MINOR}${ABI_TAG}-${PY_TAG}-${PY_TAG}-${PLATFORM_TAG}"

# -----------------------------------------------------------------------------
# 3. Show what we detected
# -----------------------------------------------------------------------------
echo
echo "Detected environment:"
echo "  Python       : $PY_VERSION"
echo "  Python tag   : $PY_TAG"
echo "  PyTorch      : $TORCH_VERSION"
echo "  Torch tag    : $TORCH_MAJOR_MINOR"
echo "  CUDA         : $CUDA_VERSION"
echo "  CUDA tag     : cu${CUDA_MAJOR}"
echo "  C++11 ABI    : $CXX11_ABI"
echo "  Architecture : $ARCH"
echo "  Platform     : $PLATFORM_TAG"
echo "  Wheel tag    : $WHEEL_TAG"

# -----------------------------------------------------------------------------
# 4. Optional safety check
# -----------------------------------------------------------------------------
if [[ "${ALLOW_OTHER_STACKS:-0}" != "1" ]]; then
    [[ "$PY_TAG"            == "cp312"   ]] || die "Expected Python 3.12; detected $PY_TAG"
    [[ "$TORCH_MAJOR_MINOR" == "2.6"     ]] || die "Expected PyTorch 2.6; detected $TORCH_MAJOR_MINOR"
    [[ "$CUDA_MAJOR"        == "12"      ]] || die "Expected CUDA 12.x; detected $CUDA_VERSION"
    [[ "$CXX11_ABI"         == "FALSE"   ]] || die "Expected CXX11 ABI FALSE; detected $CXX11_ABI"
    [[ "$ARCH"              == "x86_64"  ]] || die "Expected x86_64; detected $ARCH"
fi

# -----------------------------------------------------------------------------
# 5. Helpers
# -----------------------------------------------------------------------------

# Download URL -> path using Python's urllib. Only writes to stderr.
download() {
    local url="$1" out="$2"
    URL="$url" OUT="$out" "$PYTHON" - <<'PY'
import os, sys, urllib.request, shutil, ssl

url = os.environ["URL"]
out = os.environ["OUT"]

print(f"  GET {url}", file=sys.stderr)

ctx = ssl.create_default_context()
try:
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r, \
         open(out, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 64)
except Exception as e:
    print(f"  download failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

# Query GitHub releases API; print ONLY the download URL to stdout.
# All informational output goes to stderr so `$(...)` captures the URL alone.
find_asset_url() {
    local repo="$1" version="$2" prefix="$3"
    local api_url json_file safe_repo

    api_url="https://api.github.com/repos/${repo}/releases/tags/v${version}"
    safe_repo="${repo//\//_}"
    json_file="${TMP_DIR}/release_${safe_repo}_${version}.json"

    # --- all info to stderr (>&2), never stdout ---
    {
        echo
        echo "Searching GitHub release:"
        echo "  Repository : $repo"
        echo "  Version    : $version"
        echo "  Prefix     : $prefix"
        echo "  Tag        : $WHEEL_TAG"
    } >&2

    download "$api_url" "$json_file" >&2 \
        || die "Could not retrieve GitHub release: $api_url"

    JSON_FILE="$json_file" \
    PREFIX="$prefix" \
    TAG="$WHEEL_TAG" \
    REPO="$repo" \
    "$PYTHON" - <<'PY' || die "No compatible wheel for $repo v$version"
import json, os, sys

with open(os.environ["JSON_FILE"], encoding="utf-8") as f:
    release = json.load(f)

prefix = os.environ["PREFIX"]
tag    = os.environ["TAG"]
repo   = os.environ["REPO"]

exact, near = [], []
for asset in release.get("assets", []):
    name = asset.get("name", "")
    if not name.endswith(".whl") or not name.startswith(prefix + "-"):
        continue
    (exact if f"+{tag}.whl" in name else near).append(name)

if len(exact) != 1:
    if len(exact) > 1:
        print("Multiple exact matches:", file=sys.stderr)
        for n in exact:
            print("  " + n, file=sys.stderr)
    else:
        print(f"No wheel matching tag: {tag}", file=sys.stderr)
        print(f"Available {prefix} wheels in {release.get('tag_name','?')}:",
              file=sys.stderr)
        for n in sorted(near):
            print("  " + n, file=sys.stderr)
        print("", file=sys.stderr)
        print("Try a different version, e.g.:", file=sys.stderr)
        print("  CAUSAL_VERSION=1.6.1.post4 MAMBA_VERSION=2.3.1 ./install_mamba.sh",
              file=sys.stderr)
    sys.exit(1)

# Only the URL reaches stdout.
tag_name = release["tag_name"]
filename = exact[0]
print(f"https://github.com/{repo}/releases/download/{tag_name}/{filename}")
PY
}

# -----------------------------------------------------------------------------
# 6. Resolve wheel URLs (stdout from find_asset_url is just the URL)
# -----------------------------------------------------------------------------
CAUSAL_URL="$(find_asset_url "$CAUSAL_REPO" "$CAUSAL_VERSION" "causal_conv1d")"
MAMBA_URL="$( find_asset_url "$MAMBA_REPO"  "$MAMBA_VERSION"  "mamba_ssm")"

# Strip any trailing newlines (defensive).
CAUSAL_URL="${CAUSAL_URL//$'\n'/}"
MAMBA_URL="${MAMBA_URL//$'\n'/}"

CAUSAL_WHEEL="${CAUSAL_URL##*/}"
MAMBA_WHEEL="${MAMBA_URL##*/}"

CAUSAL_WHEEL="${CAUSAL_WHEEL//%2B/+}"
CAUSAL_WHEEL="${CAUSAL_WHEEL//%2b/+}"
MAMBA_WHEEL="${MAMBA_WHEEL//%2B/+}"
MAMBA_WHEEL="${MAMBA_WHEEL//%2b/+}"

echo
echo "Selected wheels:"
echo "  causal-conv1d: $CAUSAL_WHEEL"
echo "  mamba-ssm    : $MAMBA_WHEEL"

# -----------------------------------------------------------------------------
# 7. Download wheels
# -----------------------------------------------------------------------------
echo
echo "Downloading causal-conv1d..."
download "$CAUSAL_URL" "$TMP_DIR/$CAUSAL_WHEEL" \
    || die "Failed to download causal-conv1d"

echo
echo "Downloading mamba-ssm..."
download "$MAMBA_URL" "$TMP_DIR/$MAMBA_WHEEL" \
    || die "Failed to download mamba-ssm"

# -----------------------------------------------------------------------------
# 8. Remove existing installs
# -----------------------------------------------------------------------------
echo
echo "Removing existing installations..."
"$PYTHON" -m pip uninstall -y mamba-ssm causal-conv1d >/dev/null 2>&1 || true

# -----------------------------------------------------------------------------
# 9. Install wheels only
# -----------------------------------------------------------------------------
echo
echo "Installing causal-conv1d..."
"$PYTHON" -m pip install --no-index --no-deps "$TMP_DIR/$CAUSAL_WHEEL" \
    || die "causal-conv1d installation failed"

echo
echo "Installing mamba-ssm..."
"$PYTHON" -m pip install --no-index --no-deps "$TMP_DIR/$MAMBA_WHEEL" \
    || die "mamba-ssm installation failed"

# -----------------------------------------------------------------------------
# 10. Verify
# -----------------------------------------------------------------------------
echo
echo "============================================================"
echo " Verifying installation"
echo "============================================================"

"$PYTHON" - <<'PY'
import sys, torch
print("Python     :", sys.version.split()[0])
print("PyTorch    :", torch.__version__)
print("Torch CUDA :", torch.version.cuda)
print("CUDA avail :", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("GPU        :", torch.cuda.get_device_name(0))

import causal_conv1d                # noqa: F401
from mamba_ssm import Mamba

print("causal-conv1d: OK")
print("mamba-ssm    : OK")

model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 8, 16, device="cuda")
with torch.no_grad():
    y = model(x)
print("Mamba forward pass: OK, output shape:", tuple(y.shape))
PY

echo
echo "============================================================"
echo " Installation completed successfully"
echo "============================================================"