#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 EXPECTED_CYTOSCAPE_MIN_JS_SHA256" >&2
  exit 2
fi

expected_sha256=$1
case "$expected_sha256" in
  *[!0-9a-f]*)
    echo "expected checksum must be exactly 64 lowercase hexadecimal characters" >&2
    exit 2
    ;;
esac
checksum_length=$(printf %s "$expected_sha256" | wc -c)
if [ "$checksum_length" -ne 64 ]; then
  echo "expected checksum must be exactly 64 lowercase hexadecimal characters" >&2
  exit 2
fi

version=3.30.4
source_url="https://registry.npmjs.org/cytoscape/-/cytoscape-${version}.tgz"
script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
temporary_directory=$(mktemp -d)
trap 'rm -rf -- "$temporary_directory"' EXIT HUP INT TERM

curl --fail --location --proto '=https' --tlsv1.2 --max-filesize 5000000 \
  --output "$temporary_directory/cytoscape.tgz" "$source_url"
tar -xOzf "$temporary_directory/cytoscape.tgz" package/dist/cytoscape.min.js \
  > "$temporary_directory/cytoscape.min.js"
tar -xOzf "$temporary_directory/cytoscape.tgz" package/LICENSE \
  > "$temporary_directory/CYTOSCAPE_LICENSE"

asset_bytes=$(wc -c < "$temporary_directory/cytoscape.min.js")
if [ "$asset_bytes" -gt 2000000 ]; then
  echo "refusing unexpectedly large Cytoscape asset: $asset_bytes bytes" >&2
  exit 1
fi

actual_sha256=$(sha256sum "$temporary_directory/cytoscape.min.js" | awk '{print $1}')
if [ "$actual_sha256" != "$expected_sha256" ]; then
  echo "checksum mismatch: expected $expected_sha256, observed $actual_sha256" >&2
  exit 1
fi

license_sha256=$(sha256sum "$temporary_directory/CYTOSCAPE_LICENSE" | awk '{print $1}')
license_bytes=$(wc -c < "$temporary_directory/CYTOSCAPE_LICENSE")

install -m 0644 "$temporary_directory/cytoscape.min.js" "$script_directory/cytoscape.min.js"
install -m 0644 "$temporary_directory/CYTOSCAPE_LICENSE" "$script_directory/CYTOSCAPE_LICENSE"
cat > "$script_directory/cytoscape.lock.json" <<EOF
{
  "package": "cytoscape",
  "version": "$version",
  "source": "$source_url",
  "sha256": "$actual_sha256",
  "bytes": $asset_bytes,
  "license_file": "CYTOSCAPE_LICENSE",
  "license_sha256": "$license_sha256",
  "license_bytes": $license_bytes
}
EOF

echo "installed and verified cytoscape.min.js ($actual_sha256)"
