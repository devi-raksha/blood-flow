#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="devi-raksha/blood-flow"
RELEASE_TAG="v0.1.0"
ASSET_NAME="blood-flow-datasets-v0.1.0.tar.gz"
ASSET_SHA256="a9bef4aa466cb263d8c0bdf7503f175a659c7b22a3a812d2a9d95d3ed0988f47"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DESTINATION="${1:-${REPOSITORY_ROOT}}"
ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/blood-flow-datasets.XXXXXX")"

cleanup() {
  rm -f "${ARCHIVE}"
}
trap cleanup EXIT

DOWNLOAD_URL="https://github.com/${REPOSITORY}/releases/download/${RELEASE_TAG}/${ASSET_NAME}"

echo "Downloading network datasets from ${RELEASE_TAG}..."
curl --fail --location --retry 3 --output "${ARCHIVE}" "${DOWNLOAD_URL}"

if command -v shasum >/dev/null 2>&1; then
  printf '%s  %s\n' "${ASSET_SHA256}" "${ARCHIVE}" | shasum -a 256 -c -
elif command -v sha256sum >/dev/null 2>&1; then
  printf '%s  %s\n' "${ASSET_SHA256}" "${ARCHIVE}" | sha256sum -c -
fi

mkdir -p "${DESTINATION}"
tar -xzf "${ARCHIVE}" -C "${DESTINATION}"
echo "Datasets extracted to ${DESTINATION}."
