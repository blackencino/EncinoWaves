#!/usr/bin/env bash
# Publish the committed web/ app to GitHub Pages without rewriting branch history.
set -euo pipefail

source_root=$(git rev-parse --show-toplevel)
source_commit=$(git -C "$source_root" rev-parse HEAD)
remote_url=$(git -C "$source_root" remote get-url origin)
branch=${PAGES_BRANCH:-gh-pages}

if ! git -C "$source_root" diff --quiet HEAD -- web; then
  echo 'Commit web/ changes before deployment so the published app is reproducible.' >&2
  exit 1
fi
if [ -n "$(git -C "$source_root" ls-files --others --exclude-standard web)" ]; then
  echo 'Commit the untracked web/ files before deployment.' >&2
  exit 1
fi
git -C "$source_root" cat-file -e HEAD:web/index.html
git -C "$source_root" cat-file -e HEAD:web/assets/sky.json

scratch=$(mktemp -d "${TMPDIR:-/tmp}/encinowaves-pages.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/site"
git -C "$source_root" archive HEAD:web | tar -x -C "$scratch/site"

if git ls-remote --exit-code --heads "$remote_url" "$branch" > "$scratch/branch-ref"; then
  git clone --quiet --depth 1 --branch "$branch" "$remote_url" "$scratch/publish"
else
  status=$?
  if [ "$status" -ne 2 ]; then
    echo 'Could not inspect the remote Pages branch; deployment stopped.' >&2
    exit "$status"
  fi
  git init --quiet -b "$branch" "$scratch/publish"
  git -C "$scratch/publish" remote add origin "$remote_url"
fi

# Only the static app is published. Tests and local reports stay in the source.
rsync -a --delete --exclude=.git/ --exclude=tests/ "$scratch/site/" "$scratch/publish/"
touch "$scratch/publish/.nojekyll"
printf '%s\n' "$source_commit" > "$scratch/publish/source-commit.txt"
git -C "$scratch/publish" add --all
if git -C "$scratch/publish" diff --cached --quiet; then
  echo "Pages already contains source $source_commit"
else
  git -C "$scratch/publish" commit -m "Publish School web preview from $source_commit"
  git -C "$scratch/publish" push origin "HEAD:refs/heads/$branch"
fi
echo "Published source $source_commit on $branch."
echo "GitHub Pages must be configured to deploy from branch $branch, folder / (root)."
