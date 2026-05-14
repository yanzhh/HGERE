#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="$SCRIPT_DIR/.pypi"

if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "Error: $TOKEN_FILE not found." >&2
    exit 1
fi

API_TOKEN="$(cat "$TOKEN_FILE")"

# Verify the token file is gitignored before doing anything
if git -C "$SCRIPT_DIR" check-ignore -q "$TOKEN_FILE" 2>/dev/null; then
    echo ".pypi is gitignored — safe to proceed."
else
    echo "Error: $TOKEN_FILE is NOT gitignored. Aborting to protect the token." >&2
    exit 1
fi

cd "$SCRIPT_DIR"

echo "Cleaning previous builds..."
rm -rf dist/ build/ src/*.egg-info

echo "Building..."
uv build

echo "Uploading to PyPI..."
uv run twine upload \
    --username __token__ \
    --password "$API_TOKEN" \
    dist/*

echo "Done."
