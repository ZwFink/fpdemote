#!/bin/bash
# Builds tests/math_sanity.hip with Proteus and runs it in double, with
# demote-fp-math, and with both passes.
# Usage: run_math_sanity.sh <libDemoteFP.so> [build dir]
source "$(dirname "$0")/sanity_common.sh"
build_sanity math_sanity

echo "== double (no plugin)"
"$OUT/math_sanity"
echo "== demote-fp-math"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE=demote-fp-math "$OUT/math_sanity"
echo "== demote-fp,demote-fp-math"
FPDEMOTE_PLUGIN=$PLUGIN FPDEMOTE_PIPELINE=demote-fp,demote-fp-math \
  "$OUT/math_sanity"
