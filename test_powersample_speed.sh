#!/bin/bash

# Quick test script to compare power sampling speeds
# This tests with a small subset to show the speedup

set -euo pipefail

export OMP_NUM_THREADS=2
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

echo "=========================================="
echo "Power Sampling Speed Comparison Test"
echo "=========================================="
echo ""

# Test parameters - small subset for quick testing
TEST_EXAMPLES=50
TEST_STEPS=3
TEST_MAX_NEW=128

echo "Test configuration:"
echo "  Examples: $TEST_EXAMPLES"
echo "  MCMC steps: $TEST_STEPS"
echo "  Max new tokens: $TEST_MAX_NEW"
echo ""

# Test 1: Single GPU (original way)
echo "=========================================="
echo "Test 1: Single GPU (original)"
echo "=========================================="
export POWERSAMPLE_MAX_EXAMPLES=$TEST_EXAMPLES
export POWERSAMPLE_STEPS=$TEST_STEPS
export POWERSAMPLE_MAX_NEW=$TEST_MAX_NEW
export POWERSAMPLE_NUM_GPUS=1
export POWERSAMPLE_USE_OPTIMIZED=0

START_TIME=$SECONDS
bash speedrun_powersample.sh
SINGLE_TIME=$((SECONDS - START_TIME))
echo "Single GPU time: ${SINGLE_TIME}s"
echo ""

# Test 2: Multi-GPU (8 GPUs)
echo "=========================================="
echo "Test 2: Multi-GPU (8 GPUs)"
echo "=========================================="
export POWERSAMPLE_NUM_GPUS=8
export POWERSAMPLE_USE_OPTIMIZED=0

START_TIME=$SECONDS
bash speedrun_powersample.sh
MULTI_TIME=$((SECONDS - START_TIME))
echo "Multi-GPU time: ${MULTI_TIME}s"
echo ""

# Test 3: Multi-GPU with batching (optimized)
echo "=========================================="
echo "Test 3: Multi-GPU + Batching (Optimized)"
echo "=========================================="
export POWERSAMPLE_NUM_GPUS=8
export POWERSAMPLE_BATCH_SIZE=4
export POWERSAMPLE_USE_OPTIMIZED=1

START_TIME=$SECONDS
bash speedrun_powersample.sh
OPTIMIZED_TIME=$((SECONDS - START_TIME))
echo "Optimized time: ${OPTIMIZED_TIME}s"
echo ""

# Calculate speedups
MULTI_SPEEDUP=$(echo "scale=2; $SINGLE_TIME / $MULTI_TIME" | bc)
OPTIMIZED_SPEEDUP=$(echo "scale=2; $SINGLE_TIME / $OPTIMIZED_TIME" | bc)

echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo "Single GPU:        ${SINGLE_TIME}s (baseline)"
echo "Multi-GPU (8x):    ${MULTI_TIME}s (${MULTI_SPEEDUP}x speedup)"
echo "Optimized (8x+4b): ${OPTIMIZED_TIME}s (${OPTIMIZED_SPEEDUP}x speedup)"
echo ""
echo "With 8 GPUs and batching, you should see ~8-32x speedup!"
echo "For full GSM8K (1319 problems), estimated times:"
echo "  Single GPU:    ~$(($SINGLE_TIME * 1319 / $TEST_EXAMPLES / 60)) minutes"
echo "  Multi-GPU:     ~$(($MULTI_TIME * 1319 / $TEST_EXAMPLES / 60)) minutes"
echo "  Optimized:     ~$(($OPTIMIZED_TIME * 1319 / $TEST_EXAMPLES / 60)) minutes"
