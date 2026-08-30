#!/usr/bin/env bash
# Bootstrap the hash-pinned WEFT-1 P-A runtime on a fresh Colab Linux VM.
set -Eeuo pipefail
IFS=$'\n\t'
umask 022

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly BUILDER="${REPOSITORY_ROOT}/scripts/build_weft1_pa_runtime.py"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "WEFT-1 P-A runtime builds require Linux" >&2
  exit 2
fi
case "$(uname -m)" in
  x86_64|amd64) ;;
  *)
    echo "WEFT-1 P-A runtime lock is authorized only for x86-64 Linux" >&2
    exit 2
    ;;
esac
if [[ ! -f "${BUILDER}" || -L "${BUILDER}" ]]; then
  echo "governed runtime builder is absent or is a symlink" >&2
  exit 2
fi

readonly -a BUILD_PACKAGES=(
  build-essential
  ca-certificates
  libbz2-dev
  libdb-dev
  libffi-dev
  libgdbm-dev
  liblzma-dev
  libncurses-dev
  libnsl-dev
  libreadline-dev
  libssl-dev
  pkgconf
  tk-dev
  uuid-dev
  xz-utils
  zlib1g-dev
)

if [[ "${EUID}" -eq 0 ]]; then
  readonly -a APT=(apt-get)
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  readonly -a APT=(sudo -n apt-get)
else
  echo "root or passwordless sudo is required to install build dependencies" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export TOKENIZERS_PARALLELISM=false
export TZ=UTC

"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends "${BUILD_PACKAGES[@]}"

exec env -u LD_LIBRARY_PATH -u PYTHONHOME -u PYTHONPATH \
  python3 -I -B "${BUILDER}" --repository-root "${REPOSITORY_ROOT}" "$@"
