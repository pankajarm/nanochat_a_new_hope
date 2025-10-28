#!/bin/bash

# Quick test to verify report generation is working correctly
set -euo pipefail

export OMP_NUM_THREADS=2
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

echo "Testing report generation fix..."
echo ""

# Super minimal test - just 1 example to quickly generate a report
export POWERSAMPLE_MAX_EXAMPLES=1
export POWERSAMPLE_STEPS=1
export POWERSAMPLE_MAX_NEW=32
export POWERSAMPLE_NUM_GPUS=1

echo "Running minimal power sampling to generate report..."
source .venv/bin/activate
python -m scripts.eval_gsm8k_powersample \
    --source sft \
    --max-examples 1 \
    --num-steps 1 \
    --max-new-tokens 32 \
    --seed 0

echo ""
echo "Generating report..."
python -m nanochat.report generate

echo ""
echo "Checking if report files exist..."
if [ -f "$NANOCHAT_BASE_DIR/report/report.md" ]; then
    echo "✅ Report generated at: $NANOCHAT_BASE_DIR/report/report.md"
else
    echo "❌ Report not found at: $NANOCHAT_BASE_DIR/report/report.md"
fi

if [ -f "$NANOCHAT_BASE_DIR/report/power-sampling-evaluation.md" ]; then
    echo "✅ Power sampling section created at: $NANOCHAT_BASE_DIR/report/power-sampling-evaluation.md"
else
    echo "❌ Power sampling section not found at: $NANOCHAT_BASE_DIR/report/power-sampling-evaluation.md"
fi

echo ""
echo "Copying report to current directory..."
cp "$NANOCHAT_BASE_DIR/report/report.md" ./report_test.md && echo "✅ Report copied to ./report_test.md" || echo "❌ Failed to copy report"

echo ""
echo "Report contents preview:"
echo "========================"
head -20 ./report_test.md 2>/dev/null || echo "Could not preview report"
echo "========================"
