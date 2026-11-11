#!/bin/bash
# Builds tests/atomic_sanity.hip with Proteus and runs it in double, with
# demote-fp-atomics, and with all three passes. Each run checks its own
# expected results and fails the script on a mismatch.
# Usage: run_atomic_sanity.sh <libDemoteFP.so> [build dir]
source "$(dirname "$0")/sanity_common.sh"
build_sanity atomic_sanity

echo "== double (no plugin)"
"$OUT/atomic_sanity"
echo "== demote-fp-atomics"
QS_DEMOTE_VERBOSE=1 FPDEMOTE_PLUGIN=$PLUGIN \
  FPDEMOTE_PIPELINE=demote-fp-atomics "$OUT/atomic_sanity"
echo "== demote-fp,demote-fp-math,demote-fp-atomics"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE=demote-fp,demote-fp-math,demote-fp-atomics \
  "$OUT/atomic_sanity"
