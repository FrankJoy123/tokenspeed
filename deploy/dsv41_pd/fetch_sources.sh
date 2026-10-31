#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
mkdir -p "$SOURCE_DIR"
for manifest in "$DEPLOY_DIR/sources.tsv" "$DEPLOY_DIR/sources.local.tsv"; do
 [[ -f "$manifest" ]] || continue
 while read -r name url rev extra; do
  [[ -n "$name" && "$name" != \#* ]] || continue
  [[ "$name" =~ ^[a-zA-Z0-9_-]+$ && "$rev" =~ ^[0-9a-f]{40}$ && -z "$extra" ]] || { echo "Invalid source: $name" >&2; exit 2; }
  [[ "$url" != artifacts/* ]] || url="$DEPLOY_DIR/$url"
  dest="$SOURCE_DIR/$name"
  if [[ ! -d "$dest" ]]; then
   git clone --no-checkout "$url" "$dest"
  else
   [[ -z "$(git -C "$dest" status --porcelain)" ]] || { echo "Dirty source: $dest; preserve it and resolve manually" >&2; exit 2; }
  fi
  git -C "$dest" fetch origin "$rev"
  git -C "$dest" checkout --detach "$rev"
  git -C "$dest" submodule update --init --recursive
 done < "$manifest"
done
