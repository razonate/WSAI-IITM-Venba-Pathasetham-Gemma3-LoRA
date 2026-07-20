# Venba Pathasetham — Automated Word Segmentation of Classical Tamil Poetry

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Model](https://img.shields.io/badge/Model-Gemma%203%20270M-orange)](https://huggingface.co/google/gemma-3-270m-it)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%2B%20TRL-green)](https://unsloth.ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Razon2006-yellow)](https://huggingface.co/Razon2006)
[![Research](https://img.shields.io/badge/Research-IIT%20Madras%20WSAI-purple)](https://cerai.iitm.ac.in)
[![BLEU](https://img.shields.io/badge/BLEU--4-37%2B-brightgreen)]()

> **Research internship project at Wadhwani School of AI and Data Science, IIT Madras**
> Mentor: Dr. Sudarsun Santhiappan

---

## What Is This?

Classical Tamil **Venba** (வெண்பா) poetry compresses multiple words into single orthographic units through **sandhi** — phonological fusion at word boundaries. **Pathasetham** (பதசேதம்) is the traditional scholarly practice of decomposing a Venba verse back into its constituent words.

**Example:**
```
Input  (Venba)    : வண்மைதரு மாகமநூன் வைத்த பொருள்வழுவா உண்மை விளக்க முரைசெய்யத்
Output (Pathaseth): வண்மை தரும் ஆகம நூல் வைத்த பொருள் வழுவா உண்மை விளக்கம் உரை செய்யத்
```

This project builds the **first known ML pipeline** for automated Tamil Venba Pathasetham using supervised fine-tuning of Gemma 3 270M using LoRA, with a novel **grapheme-level tokenization scheme** that addresses the fundamental incompatibility between BPE tokenizers and Tamil's morphophonemic structure.

---

## Why This Project Is Unique

### 1. Novel Task
No prior ML work exists for classical Tamil Venba Pathasetham. This project establishes the first baseline metrics (BLEU, Word Accuracy, Exact Match) for the task.

### 2. Tokenizer Insight
Standard BPE tokenizers split Tamil aksharas (grapheme clusters) sub-phonemically. We empirically demonstrate this across 412,228 akshara occurrences and implement a fix — directly validated by the concurrent **Agathiyam paper (ICLR 2026 under review)** on sandhi-aware Tamil tokenization.

### 3. Low-Resource NLP
~3,700 training examples for a morphologically complex transformation task. The project explores what a 270M model can learn about classical Tamil grammar from minimal data.

### 4. Critical Bugs Discovered and Fixed
The project documents and resolves non-obvious bugs in the Unsloth/TRL SFT pipeline that silently corrupt training — useful reference for anyone doing Tamil LLM fine-tuning.

---

## Results

| Configuration | BLEU-4 |
|---|---|
| BPE, r=16, 50/50 split (baseline) | 18.44 |
| BPE, r=16, 80/20 split | 22.68 |
| **Grapheme tokenization, r=16, 80/20** | 37 |
| Grapheme + no-space Venba  | 30 | 

---

## Key Technical Contributions

### Grapheme-Level Tokenization
```python
# Instead of letting BPE merge Tamil aksharas arbitrarily:
# விருந்தோம்பல் → ['▁விருந்தோம்', 'பல்']  ← sandhi boundary HIDDEN
#
# We force akshara-level tokenization:
# விருந்தோம்பல் → வி|ரு|ந்|தோ|ம்|ப|ல்  ← every akshara = 1 token
#                  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#                  sandhi boundary now VISIBLE between tokens

def clean_grapheme_split(text, tokenizer):
    words = text.split(' ')
    result_words = []
    for word in words:
        clusters = regex.findall(r'\X', word)
        clusters = [unicodedata.normalize('NFC', c) for c in clusters
                    if c.strip() and any('\u0B80' <= ch <= '\u0BFF' for ch in c)]
        # Recursive merge for the 2.3% of aksharas BPE would still split
        merged = list(clusters)
        changed = True
        while changed:
            changed = False
            new_merged = []
            i = 0
            while i < len(merged):
                unit = merged[i]
                if len(tokenizer.tokenize(unit)) > 1 and i + 1 < len(merged):
                    new_merged.append(unit + merged[i+1])
                    i += 2
                    changed = True
                else:
                    new_merged.append(unit)
                    i += 1
            merged = new_merged
        result_words.append('|'.join(merged))
    return ' '.join(result_words)
```

### No-Space Venba Experiment 
```
Normal input : வண்மைதரு மாகமநூன் வைத்த
No-space input: வ|ண்|மை|த|ரு மா|க|ம|நூன் வை|த்|த  →  வ|ண்|மை|த|ரு|மா|க|ம|நூன்|வை|த்|த
```
Removing all spaces forces the model to learn sandhi splitting purely from character-level patterns — a significantly harder task that tests genuine linguistic generalization.

### Critical Bugs Fixed

| Bug | Symptom | Fix |
|---|---|---|
| Double `<bos>` token | Near-zero BLEU despite healthy loss | Strip `<bos>` from `apply_chat_template` output before SFTTrainer tokenization |
| Evaluation slicing | Predicted text includes prompt leakage | Decode full sequence, strip prompt as string prefix |
| `bad_words_ids` vocab overflow | `ValueError: vocabulary size is 262144, but [262144]` | Filter `tid < model.config.vocab_size` |
| Merged model NaN weights | All logits = `nan`, zero output tokens | Load LoRA adapter separately in bfloat16; skip `save_pretrained_merged` |
| Loss over full sequence | Inflated loss (~2.97 vs real ~0.37) | Use `train_on_responses_only` to mask instruction tokens |
| TRL 0.24.0 breaking change | `DataCollatorForCompletionOnlyLM` import fails | Migrate to `train_on_responses_only` from `unsloth.chat_templates` |

---

## Dataset

**3,729 Venba–Pathasetham pairs** curated from classical Tamil texts:
- Pazhamozhi Nanooru (400 venbas, 34 chapters) — scraped from sangathamizh.com
- Ulladu Narpadu (43 verses) — Ramana Maharshi
- Additional classical Tamil Venba sources

```
Columns : Venba (input), Pathavurai (output/pathasetham)
Size    : 3,729 rows (after deduplication)
Language: Classical Tamil (Unicode, Tamil script)
Split   : 80/20 train/test (random_state=42)
```

**Dataset statistics:**

| Metric | Venba | Pathavurai |
|---|---|---|
| Mean characters | 105 | 112 |
| Mean word count | 12 | 16 |
| Max characters | 199 | 221 |

---

## Model Architecture

```
Base Model : Gemma 3 270M (unsloth/gemma-3-270m-it)
Method     : LoRA fine-tuning (r=16, alpha=16)
Framework  : Unsloth + TRL SFTTrainer
Masking    : train_on_responses_only (loss on Pathasetham tokens only)
Tokenizer  : Gemma BPE + grapheme pre-processing (| separator)
Platform   : Kaggle T4 GPU
```

**Training configuration:**
```python
r               = 16
lora_alpha      = 16
lora_dropout    = 0
learning_rate   = 1e-4
num_epochs      = 8
batch_size      = 2 (effective = 8 with grad_accum=4)
max_seq_length  = 512
scheduler       = linear
```

---

## Quick Start

### Installation
```bash
git clone https://github.com/Razon2006/tamil-venba-pathasetham
cd tamil-venba-pathasetham
pip install unsloth trl transformers peft sacrebleu regex
```

### Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "unsloth/gemma-3-270m-it"
LORA_PATH  = "Razon2006/gemma3-tamil-pathasetham-grapheme-r32-lr1e4"

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
```

### Run Tokenizer Analysis
```bash
# On Kaggle — add dataset and run:
jupyter nbconvert --to notebook --execute notebooks/tokenizer_analysis.ipynb
```

---

## Tokenizer Analysis

Dataset-wide BPE tokenization scan (412,228 akshara occurrences):

```
Clean aksharas (1 token each) : 402,562 (97.7%)
Problematic aksharas (split)  : 9,666   (2.3%)
Unique problematic aksharas   : 93
```

Most frequent problematic aksharas: `யு` (1973×), `றா` (1600×), `ஞா` (649×), `யே` (539×), `யை` (443×)

These correspond to Tamil-specific consonants (`ய`, `ற`, `ழ`, `ஞ`) with vowel diacritics that BPE never learned as merged units due to low frequency in multilingual pre-training data.

---

## Related Work

| Work | Relevance |
|---|---|
| **Agathiyam** (ICLR 2026, under review) | Sandhi-aware GPE tokenizer for Tamil; directly validates our tokenization approach |
| **CHiPSAL 2025** (Dasari et al.) | Seq2seq sandhi splitting for Tamil/Telugu on 15K corpus |
| **GPE** (Velayuthan & Sarveswaran, 2024) | Grapheme Pair Encoding for Tamil |
| **ThamizhiMorph** (Sarveswaran et al.) | Rule-based FST morphological analyser for Tamil |
| **IndicBARTSS** (AI4Bharat) | Seq2seq model on 9B Indic tokens; planned future work |

---

## Future Work

- [ ] Web application on HuggingFace Spaces (Gradio)
- [ ] Research paper submission (COLING / LREC-COLING / Indian NLP workshop)
- [ ] Expand corpus to 10,000+ pairs from Thirukkural, Purananuru, Kambaramayanam
- [ ] IndicBARTSS fine-tuning (expected BLEU 40-60)
- [ ] Agathiyam tokenizer integration once publicly released
- [ ] Venba Synthesis (inverse task: prose → valid Venba)
- [ ] Public dataset release on HuggingFace Datasets

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{savio2026venba,
  title     = {Tamil Venba Pathasetham: Automated Word Segmentation of Classical Tamil Poetry using Grapheme-Level Tokenization and Supervised Fine-Tuning},
  author    = {Robin Savio},
  year      = {2026},
  note      = {Research Internship, WSAI, IIT Madras. Mentor: Dr. Sudarsun Santhiappan},
  url       = {https://github.com/Razon2006/tamil-venba-pathasetham}
}
```

---

## Acknowledgements

This work was conducted as a research internship at the **Wadhwani School of Data Science and Artificial Intelligence (WSAI), IIT Madras** under the supervision of **Dr. Sudarsun Santhiappan**.
---

*Project ID: 2026/WSAI/033 | IIT Madras*
