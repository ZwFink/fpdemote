#!/bin/bash
# Builds tests/math_sanity.hip with Proteus and runs it in double, with only
# the math part of demote-fp, and with all of demote-fp.
# Usage: run_math_sanity.sh <libDemoteFP.so> [build dir]
source "$(dirname "$0")/sanity_common.sh"
build_sanity math_sanity

echo "== double (no plugin)"
"$OUT/math_sanity"
echo "== demote-fp<no-arith;no-atomics>"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE="demote-fp<no-arith;no-atomics>" \
  "$OUT/math_sanity"
echo "== demote-fp"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE=demote-fp "$OUT/math_sanity"
