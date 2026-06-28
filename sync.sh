#!/usr/bin/env bash
# Sync THIS fork with the official Poker44 upstream and publish to your repo.
# Self-contained: run it from anywhere inside the repo.  Usage:  ./sync.sh
set -euo pipefail

cd "$(dirname "$0")"
UPSTREAM_URL="https://github.com/Poker44/Poker44-subnet.git"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Auto-configure the upstream remote on a fresh clone (fetch-only, push blocked).
if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "==> Adding 'upstream' remote ($UPSTREAM_URL)"
  git remote add upstream "$UPSTREAM_URL"
  git remote set-url --push upstream DISABLED_NO_PUSH
fi

echo "==> Fetching upstream..."
git fetch upstream

echo "==> Incoming upstream commits:"
git log --oneline "HEAD..upstream/$BRANCH" || true

echo "==> Merging upstream/$BRANCH..."
if ! git merge --no-edit "upstream/$BRANCH"; then
  echo
  echo "!! Merge conflicts. Likely: neurons/miner.py, requirements.txt, scripts/miner/run/run_miner.sh"
  echo "   Your model is in poker44_model/, so conflicts here are usually small."
  echo "   Fix the files, then:  git add -A && git commit  &&  ./sync.sh"
  exit 1
fi

echo "==> Running tests..."
PYTHONPATH="$(pwd)" python3 -m unittest discover -s tests

echo "==> Pushing to your repo (origin/$BRANCH)..."
git push origin "$BRANCH"

echo
echo "==> Done. HEAD is now $(git rev-parse --short HEAD)."
echo "    Restart the miner to serve it:  ./run_p44_miner.sh"
