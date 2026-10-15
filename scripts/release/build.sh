#!/usr/bin/env bash
# build.sh <version> <sha> <out.zip>
# Zip the worker's files at <sha> (a commit on the checked-out main) with a
# release.json naming <version>.
set -euo pipefail

version=$1
sha=$(git rev-parse --verify "$2^{commit}")
out=$3

if ! git merge-base --is-ancestor "$sha" HEAD; then
  echo "::error::$sha is not a commit on main"
  exit 1
fi

tmp=$(mktemp -d)
jq -n --arg version "$version" --arg commit "$sha" \
  '{version: $version, commit: $commit, ref: "production"}' > "$tmp/release.json"
git archive --format=zip --add-file="$tmp/release.json" -o "$out" "$sha" \
  src comfy_nodes manifests assets scripts deploy ops pyproject.toml
unzip -l "$out" | tail -1
