#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build the exact third-party runtime used by the formal zero-shot suite.
# Native Ubuntu build dependencies must already be installed.

ROOT="${VOXELIBRE_DIR:-${PROJECT_ROOT}/.voxelibre}"
CRAFTIUM_COMMIT="8fb5a13eb8690fe3efde72cc7ae55da34f71f1ae"
CRAFTIUM_TAG="v0.0.1"
VOXELIBRE_COMMIT="23cd983750b787bef33b6969192a653ae1b15189"
VOXELIBRE_TAG="0.89.4"
CRAFTIUM_DIR="$ROOT/craftium-v0.0.1"
CRAFTIUM_PYTHON_DIR="$ROOT/craftium-python"
VOXELIBRE_GAME_DIR="$ROOT/VoxeLibre-0.89.4"
CRAFTIUM_PATCH="$PROJECT_ROOT/src/gamewam/envs/voxelibre/patches/craftium_v0.0.1_virtual_place_edge.diff"
CRAFTIUM_PACKAGING_PATCH="$PROJECT_ROOT/src/gamewam/envs/voxelibre/patches/craftium_v0.0.1_optional_package_dirs.diff"
CRAFTIUM_EXTERNAL_RUNTIME_PATCH="$PROJECT_ROOT/src/gamewam/envs/voxelibre/patches/craftium_v0.0.1_external_runtime_package.diff"
CRAFTIUM_XVFB_FRAMEBUFFER_PATCH="$PROJECT_ROOT/src/gamewam/envs/voxelibre/patches/craftium_v0.0.1_xvfb_framebuffer.diff"
BUILD_JOBS="$(python -c 'import os; print(min(os.cpu_count() or 1, 8))')"

ensure_pinned_tag() {
    local repository="$1"
    local tag="$2"
    local expected_commit="$3"

    if ! git -C "$repository" cat-file -e "${expected_commit}^{commit}" 2>/dev/null || \
            ! git -C "$repository" show-ref --verify --quiet "refs/tags/${tag}"; then
        git -C "$repository" fetch --depth 1 origin \
            "+refs/tags/${tag}:refs/tags/${tag}"
    fi
    local tag_commit
    tag_commit="$(git -C "$repository" rev-list -n 1 "$tag")"
    if [[ "$tag_commit" != "$expected_commit" ]]; then
        echo "ERROR: $repository tag $tag resolved to $tag_commit; expected $expected_commit" >&2
        exit 1
    fi
}

mkdir -p "$ROOT"
if [[ ! -d "$CRAFTIUM_DIR/.git" ]]; then
    git clone --branch "$CRAFTIUM_TAG" --depth 1 --no-checkout \
        https://github.com/mikelma/craftium.git "$CRAFTIUM_DIR"
fi
ensure_pinned_tag "$CRAFTIUM_DIR" "$CRAFTIUM_TAG" "$CRAFTIUM_COMMIT"
git -C "$CRAFTIUM_DIR" checkout --detach "$CRAFTIUM_COMMIT"
git -C "$CRAFTIUM_DIR" submodule update --init --recursive

# Restore only files owned by our patches before reapplying them. The packaging
# patch builds on the setup.py change in the PLACE-edge patch, so checking each
# complete patch in reverse is not sufficient after both have been applied.
# Build outputs and all other checkout contents remain untouched.
git -C "$CRAFTIUM_DIR" checkout "$CRAFTIUM_COMMIT" -- \
    pyproject.toml \
    setup.py \
    irr/src/COpenGLDriver.cpp \
    src/client/client.cpp \
    src/client/craftium.h \
    src/client/inputhandler.h

# Craftium v0.0.1 omits virtual PLACE from wasKeyPressed(), so a one-frame
# Minecraft right-click is lost. The packaging patch also skips output folders
# for optional features disabled by this evaluation build (such as Gettext).
if ! git -C "$CRAFTIUM_DIR" apply --check "$CRAFTIUM_PATCH"; then
    echo "ERROR: Craftium source does not match the pinned PLACE-edge patch" >&2
    exit 1
fi
git -C "$CRAFTIUM_DIR" apply "$CRAFTIUM_PATCH"

if ! git -C "$CRAFTIUM_DIR" apply --check "$CRAFTIUM_PACKAGING_PATCH"; then
    echo "ERROR: Craftium source does not match the pinned packaging patch" >&2
    exit 1
fi
git -C "$CRAFTIUM_DIR" apply "$CRAFTIUM_PACKAGING_PATCH"

if ! git -C "$CRAFTIUM_DIR" apply --check "$CRAFTIUM_EXTERNAL_RUNTIME_PATCH"; then
    echo "ERROR: Craftium source does not match the external-runtime packaging patch" >&2
    exit 1
fi
git -C "$CRAFTIUM_DIR" apply "$CRAFTIUM_EXTERNAL_RUNTIME_PATCH"

# Craftium v0.0.1 reads a permanently black GL front buffer under SDL/Xvfb.
# Read the rendered back buffer used by the evaluation launcher instead.
if ! git -C "$CRAFTIUM_DIR" apply --check "$CRAFTIUM_XVFB_FRAMEBUFFER_PATCH"; then
    echo "ERROR: Craftium source does not match the Xvfb framebuffer patch" >&2
    exit 1
fi
git -C "$CRAFTIUM_DIR" apply "$CRAFTIUM_XVFB_FRAMEBUFFER_PATCH"

if [[ ! -d "$VOXELIBRE_GAME_DIR/.git" ]]; then
    git clone --branch "$VOXELIBRE_TAG" --depth 1 --no-checkout \
        https://github.com/VoxeLibre/VoxeLibre.git "$VOXELIBRE_GAME_DIR"
fi
ensure_pinned_tag "$VOXELIBRE_GAME_DIR" "$VOXELIBRE_TAG" "$VOXELIBRE_COMMIT"
git -C "$VOXELIBRE_GAME_DIR" checkout --detach "$VOXELIBRE_COMMIT"

# Configure only the client features used by visual evaluation. In particular,
# sound is disabled so a headless evaluator does not require OpenAL/Vorbis.
# The Python wheel step below reuses this external build.
CMAKE_ARGS=(
    -S "$CRAFTIUM_DIR"
    -B "$CRAFTIUM_DIR"
    -DRUN_IN_PLACE=TRUE
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_UNITTESTS=FALSE
    -DBUILD_DOCUMENTATION=FALSE
    -DENABLE_LTO=FALSE
    -DENABLE_SOUND=FALSE
    -DENABLE_CURL=FALSE
    -DENABLE_GETTEXT=FALSE
)

# Some GPU containers expose NVIDIA's GL dispatch before Mesa. Make CMake use
# one internally consistent GLVND provider when that node layout is present.
if [[ -f /usr/local/nvidia/lib64/libOpenGL.so && \
      -f /usr/local/nvidia/lib64/libGLX.so && \
      -f /usr/local/nvidia/lib64/libEGL.so ]]; then
    CMAKE_ARGS+=(
        -DOPENGL_opengl_LIBRARY=/usr/local/nvidia/lib64/libOpenGL.so
        -DOPENGL_glx_LIBRARY=/usr/local/nvidia/lib64/libGLX.so
        -DOPENGL_egl_LIBRARY=/usr/local/nvidia/lib64/libEGL.so
        -DOPENGL_INCLUDE_DIR=/usr/include
    )
fi
cmake "${CMAKE_ARGS[@]}"
cmake --build "$CRAFTIUM_DIR" --parallel "$BUILD_JOBS"
printf '%s\n' 'craftium-v0.0.1-gamewam-xvfb-back-buffer-v1' > \
    "$CRAFTIUM_DIR/GAMEWAM_XVFB_FRAMEBUFFER"

# Discard setuptools output from an earlier full-runtime wheel. CMake builds
# in the source tree, so these directories contain only Python package output.
rm -rf "$CRAFTIUM_DIR/build" "$CRAFTIUM_DIR/craftium.egg-info"
rm -rf "$CRAFTIUM_PYTHON_DIR"
mkdir -p "$CRAFTIUM_PYTHON_DIR"

(cd "$CRAFTIUM_DIR" && \
    CRAFTIUM_BUILD_MINETEST=0 CRAFTIUM_PACKAGE_RUNTIME=0 \
    python -m pip install \
        --no-deps --no-build-isolation --target "$CRAFTIUM_PYTHON_DIR" .)

CRAFTIUM_PYTHON_DIR="$CRAFTIUM_PYTHON_DIR" \
PYTHONPATH="$CRAFTIUM_PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python - <<'PY'
import os
from importlib import metadata, resources
from pathlib import Path

from craftium.craftium_env import ACTION_ORDER

assert metadata.version("craftium") == "0.0.1"
package_dir = Path(str(resources.files("craftium"))).resolve()
expected_dir = (Path(os.environ["CRAFTIUM_PYTHON_DIR"]) / "craftium").resolve()
assert package_dir == expected_dir, (package_dir, expected_dir)
marker = resources.files("craftium").joinpath("GAMEWAM_VIRTUAL_PLACE_EDGE")
assert marker.is_file()
assert marker.read_text(encoding="utf-8").strip() == \
    "craftium-v0.0.1-gamewam-place-edge-v1"
assert len(ACTION_ORDER) == 22 and ACTION_ORDER[-1] == "mouse"
print("Craftium 0.0.1 import/action contract verified.")
PY

echo "VoxeLibre game directory: $VOXELIBRE_GAME_DIR"
echo "Craftium Python directory: $CRAFTIUM_PYTHON_DIR"
