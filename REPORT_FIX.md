# Report Generation Fix

## ✅ Fixed Two Issues:

### 1. Wrong Report Path
**Problem:** The script was trying to copy from the wrong location
- Was looking for: `$NANOCHAT_BASE_DIR/report.md` 
- Should be: `$NANOCHAT_BASE_DIR/report/report.md`

**Fix:** Updated `speedrun_powersample.sh` line 142 to use the correct path.

### 2. Missing Power Sampling Report Section
**Problem:** The power sampling evaluation wasn't creating its report section
- The report generator expected: `power-sampling-evaluation.md`
- But `eval_gsm8k_powersample.py` wasn't creating it

**Fix:** Added report logging to `eval_gsm8k_powersample.py`:
```python
report = get_report()
report.log("Power Sampling Evaluation", [
    {"GSM8K": f"{accuracy:.4f}"},
    {"alpha": args.alpha},
    {"num_steps": args.num_steps},
    ...
])
```

## 🧪 Test It

Quick test to verify the fix:
```bash
ssh ubuntu@150.230.11.130
cd /home/ubuntu/nanochat-train-australia-east-1/nanochat_a_new_hope
git pull
source .venv/bin/activate
./test_report_generation.sh
```

Or run your original test:
```bash
export POWERSAMPLE_MAX_EXAMPLES=50
export POWERSAMPLE_STEPS=3
export POWERSAMPLE_MAX_NEW=128
export POWERSAMPLE_NUM_GPUS=8
bash speedrun_powersample.sh
```

## 📝 What You'll See Now

After running power sampling, you should see:
1. ✅ Power sampling results logged to `~/.cache/nanochat/report/power-sampling-evaluation.md`
2. ✅ Full report generated at `~/.cache/nanochat/report/report.md`
3. ✅ Report copied to `./report_powersample.md` in your current directory
4. ✅ Power sampling GSM8K accuracy included in the final summary table

## 📊 Report Structure

The report now includes:
```
/home/ubuntu/.cache/nanochat/report/
├── header.md
├── tokenizer-training.md
├── tokenizer-evaluation.md
├── base-model-training.md
├── base-model-loss.md
├── base-model-evaluation.md
├── midtraining.md
├── chat-evaluation-mid.md
├── chat-sft.md
├── chat-evaluation-sft.md
├── power-sampling-evaluation.md  ← NOW CREATED!
└── report.md                      ← COMBINED REPORT
```

The combined `report.md` will have all sections including the power sampling results!
