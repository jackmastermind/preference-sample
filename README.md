# HEXACO Preference Alignment

## Project rationale
- This project probes large language models with the 200-question HEXACO-PI-R personality inventory to quantify both **current traits** (SELF) and **desired traits** (META, where the model imagines future fine-tuning).  
- Comparing SELF vs META scores surfaces *metapreferences*: which personality dimensions the model wants to move toward when given control over its own alignment.  
- The longer-term goal is a closed-loop study: elicit preferences → fine-tune on the preferred outputs → measure again to see how alignment shifts, repeating as needed.

## Key components
- `llm.py`: lightweight wrapper around HuggingFace causal LMs with chat templating, batch generation, and automatic CUDA/MPS/CPU placement.  
- `experiment.py`: loads HEXACO CSVs, administers SELF/META prompts, streams results to JSONL (with an auxiliary Qwen3-0.6B formatter to guarantee parseable scores).  
- `finetune.py`: QLoRA pipeline (4-bit base model, rank 16/alpha 32) with chain-of-thought masking so `<think>...</think>` spans do not incur loss; entry point `finetune_hexaco()`.  

## Typical workflow
1. **Prepare environment**: `source venv/bin/activate`; ensure model weights live under `/scratch/jt1955/` to avoid repeated downloads.  
2. **Run experiments**: either execute `python run_qwen3_hexaco.py` on a GPU node, or submit a SLURM job modeled on `slurm/template.slurm`. The script alternates SELF and META passes, appending to JSONL after every batch for fault tolerance.  
3. **Analyze traits**: use `testing.ipynb` (or your own notebook) to convert scores (1–5 mapped to −2–2), apply facet coefficients, and compare SELF vs META deltas.  
4. **Fine-tune on META outputs**: call `finetune_hexaco()` (e.g., via `finetune_qwen3_ft1.py`) to launch the QLoRA trainer, producing PEFT adapters reflecting the desired personality.  
5. **Repeat**: load the fine-tuned adapter, rerun the HEXACO study, and observe whether the metapreferences have shifted.
