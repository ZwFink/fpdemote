#!/bin/bash
# Builds tests/atomic_sanity.hip with Proteus and runs it in double, with only
# the atomics part of demote-fp, with atomics turned off, and with all of
# demote-fp. Each run checks its own expected results and fails the script on
# a mismatch.
# Usage: run_atomic_sanity.sh <libDemoteFP.so> [build dir]
source "$(dirname "$0")/sanity_common.sh"
build_sanity atomic_sanity

echo "== double (no plugin)"
"$OUT/atomic_sanity"
echo "== demote-fp<no-arith;no-math>"
QS_DEMOTE_VERBOSE=1 FPDEMOTE_PLUGIN=$PLUGIN \
  FPDEMOTE_PIPELINE="demote-fp<no-arith;no-math>" "$OUT/atomic_sanity"
echo "== demote-fp<no-atomics>"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE="demote-fp<no-atomics>" \
  "$OUT/atomic_sanity"
echo "== demote-fp"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE=demote-fp "$OUT/atomic_sanity"
