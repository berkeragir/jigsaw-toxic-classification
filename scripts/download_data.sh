#!/usr/bin/env bash
# Download the Jigsaw Toxic Comment Classification dataset into the repo root.
#
# Prerequisites:
#   1. Install the Kaggle CLI:        pip install kaggle
#   2. Place your API token at:       ~/.kaggle/kaggle.json   (chmod 600)
#      Get the token from:            https://www.kaggle.com/settings -> "Create New Token"
#   3. Accept the competition rules at:
#      https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules
#
# After running this script, the repo root will contain:
#   train.csv, test.csv, test_labels.csv, sample_submission.csv

set -euo pipefail

COMPETITION="jigsaw-toxic-comment-classification-challenge"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Downloading $COMPETITION into $REPO_ROOT ..."
cd "$REPO_ROOT"

kaggle competitions download -c "$COMPETITION" -p .

# The download is a single zip containing per-file zips.
unzip -o "${COMPETITION}.zip"
for f in train.csv.zip test.csv.zip test_labels.csv.zip sample_submission.csv.zip; do
  if [ -f "$f" ]; then
    unzip -o "$f"
    rm -f "$f"
  fi
done
rm -f "${COMPETITION}.zip"

echo "Done. Files now in $REPO_ROOT :"
ls -lh train.csv test.csv test_labels.csv sample_submission.csv
