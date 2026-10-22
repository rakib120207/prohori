# Adding more data to Prohori: what, where from, where to put it

## 1. Do you need more data? Yes, and the evidence says which kind

The cross-validation report (`models/cv_report.json`, shown in the app) measures what happens on a kind of human text the model never saw:

| Held-out domain | Accuracy when seen in training | Accuracy when never seen | Human text flagged |
|---|---|---|---|
| textbook | 81% | **62%** | 10% |
| wikinews | 67% | **57%** | **32%** |

The model has only ever seen two kinds of human writing, so it does not generalize. **More rows of the same kind will not fix this. More different domains will.** The same report shows weakness on unseen AI models (DeepSeek caught 28% when unseen), so AI-side variety matters too.

You also reported that it over-indexes on "good grammar = AI". The canonicalization layer removes the cheap tells (spelling, typography), but the vocabulary-register signal remains until the training data contains **polished human writing and casual AI writing**. That is why the priorities below ask for both.

## 2. Where files go (so nothing gets mixed up)

| Folder / file | What goes there | Used for |
|---|---|---|
| `data/raw/` | The original Mendeley benchmark `.xlsx` (Bangla version) | Training. Merged automatically by `prepare_dataset.py` |
| `data/external/` | Any labeled dataset you only want to **test** on | Never trained on. Use `eval_external.py` |
| `data/own_human_texts.csv` | Human passages waiting for AI rewriting (made by `import_corpus.py`) | Input of `generate_ai_rewrites.py` |
| `data/own/` | Ready-to-train files: `own_pairs.csv` (from Groq) and anything made by `add_dataset.py` | Training. Merged automatically |
| `data/processed/dataset.csv` | Built by `prepare_dataset.py` | Training input. Never edit by hand |

None of `data/` is committed to GitHub (it is in `.gitignore`), and **do not upload student texts or datasets to the public repo through the browser**.

## 3. The rules that keep the data honest

1. **Human text must be pre-2022** (before LLM text flooded the web) or your own consented writing. One AI-written "human" sample teaches the wrong lesson.
2. **Test before you train.** Run `eval_external.py` on a new labeled dataset first. If it scores badly there, that is exactly the weakness the data will fix; if it scores well, you may not need it.
3. **Every new human domain needs both human and AI rows**, and the AI rows should come in several styles (`--styles formal,casual,student`), so grammar quality is not the dividing line.
4. After adding data, rerun `evaluate_cv.py` and look at the **unseen-domain** table, not just overall accuracy. Target: unseen-domain accuracy of 75% or more with under 5% of human text flagged.
5. Keep `--exclude-sources wikisource` unless you also add other archaic-spelling human text to balance it (old spelling that always means "human" is a shortcut).

## 4. Priorities (do them in this order)

### P0: test on existing labeled Bangla datasets (cost: 30 minutes, no training)

Verified to exist:

| Dataset | What it is | Notes |
|---|---|---|
| **Bangla Human & AI Text Classification**, Mendeley Data, DOI 10.17632/tvtm9tkvdn.1 | 6,000 texts (3,000 human, 3,000 AI), columns `source_text`, `Source` (Human/AI), `Source Name` (the human origin, or the LLM: Gemini, GPT, Qwen, Grok) | Other generators (Qwen, Grok) = a clean unseen-generator test. Not paired |
| **Bangla AI-Generated and Human-Written Text Dataset**, IEEE DataPort, DOI 10.21227/8cmg-6267 | Excel with `Text`, `Generated_By` (AI/Human), `Text_Generation` (1 = AI). Human text from the internet and social media, AI from GPT models | Casual and web domain. DataPort may ask for a free account before you can download |

```powershell
# put the downloaded files in data\external\
python scripts\eval_external.py --file data\external\tvtm9tkvdn.xlsx --text-col source_text --label-col Source --ai-label AI --source-col "Source Name"
python scripts\eval_external.py --file data\external\ieee_bangla.xlsx --text-col Text --label-col Text_Generation --ai-label 1 --source-col Generated_By
```

Column names are as described on each dataset page; if a file differs, the script prints the columns it found and you override with the flags. Read the per-source lines at the bottom: they show which generator or origin the model fails on.

### P1: polished human text from other domains (the biggest fix for "good grammar = AI")

| Source | Domain | Where | Notes |
|---|---|---|---|
| **Potrika**, Mendeley Data DOI 10.17632/v362rp78dc | News, 2014-2020, six Bangladeshi portals, eight topics (sports, economy, education, science and tech, politics ...); 664,880 articles in the raw set | data.mendeley.com | Pre-LLM, so safely human. Articles are organized in one folder per category. Use several topics, not just one |
| **Bangla Wikipedia** | Encyclopedic, polished | dumps.wikimedia.org (the `bnwiki` dumps; take one from before 2022) | A standard source, not re-verified here. Extract plain text first (for example with the `wikiextractor` tool), then import the extracted files |

```powershell
python scripts\import_corpus.py --input "D:\downloads\potrika\Education" --domain news_education --n 300
python scripts\import_corpus.py --input "D:\downloads\potrika\Sports"    --domain news_sports    --n 300
python scripts\import_corpus.py --input "D:\downloads\bnwiki_text"        --domain wikipedia      --n 300
$env:GROQ_API_KEY = "your-key"
python scripts\generate_ai_rewrites.py --input data\own_human_texts.csv --rewrites 2 --styles formal,casual
```

Suggested size: about 300 human passages per domain, each with 2 AI rewrites (formal + casual). Start with 3 domains (about 900 humans, 1,800 AI rewrites) and measure before adding more. The Groq script resumes if interrupted and respects rate limits with retries; check Groq's current free limits and model names on their site, since both change.

### P2: casual human text

The human half of the IEEE DataPort dataset (internet and social media) is the quickest source. To train on it instead of only testing:

```powershell
python scripts\add_dataset.py --file data\external\ieee_bangla.xlsx --name ieee_social --text-col Text --label-col Text_Generation --ai-label 1 --source-col Generated_By --preview
python scripts\add_dataset.py --file data\external\ieee_bangla.xlsx --name ieee_social --text-col Text --label-col Text_Generation --ai-label 1 --source-col Generated_By
```

(`--preview` prints the label counts and two samples per label so you can confirm the direction; the second command writes `data/own/ieee_social.csv`.) Check the dataset's licence first. Keep only text from before 2022 if the page gives dates.

### P3: your target domain, student writing (the one that decides whether it works for you)

1. Collect **100-150 genuine student texts** of 60-150 words with consent (a simple form works). Remove names and IDs. Never upload them to GitHub.
2. Save as `data\own_human_texts.csv` with columns `text,source` (use `source` = `student_assignment`), or import a CSV with `import_corpus.py --domain student_assignment`.
3. Rewrite with `--styles formal,casual,student` so the AI side includes writing that looks like a student's.
4. Keep roughly 30 of the student texts **out of training** as a final check: put them in `data\external\students.csv` with `text,label` columns (and AI versions you generate separately) and run `eval_external.py`. This is the number to report.

### P4: more AI generators

`unseen_generator` in the report shows how much is missed when a model is new. Every few months, add fresh rewrites from the current versions of ChatGPT, Gemini, Claude, DeepSeek, Qwen, Grok and others. Without an API key, paste outputs by hand into a CSV with the columns `text,label,pair_id,source,generator` (label 1, `pair_id` matching the human row, `source` = the model's name) and drop it in `data/own/`.

## 5. Rebuild, retrain, re-measure

```powershell
python scripts\prepare_dataset.py
python scripts\train.py --exclude-sources wikisource --refit-all
python scripts\evaluate_cv.py --exclude-sources wikisource
```

Then upload the five files in `models/` (`model.joblib`, `lm.joblib`, `surface.joblib`, `meta.json`, `cv_report.json`) to your GitHub repo; Render redeploys on its own.

## 6. If your laptop blocks Python: run all of this on Google Colab (free, Linux)

In a new Colab notebook, one cell per step:

```python
!git clone https://github.com/rakib120207/prohori.git
%cd prohori
!pip install -q -r requirements-dev.txt
```

Upload your files into the Colab file browser under `prohori/data/raw/`, `prohori/data/external/`, `prohori/data/own/` (drag and drop), then run the same `python scripts/...` commands with a `!` in front and forward slashes in paths. When training finishes, download the `models/` folder and upload its five files to GitHub. Colab is also where to run `scripts/finetune_banglabert.py` on a free GPU.

## 7. How much will it help? (honest expectations)

I have not run any of the new datasets, so I cannot promise numbers. Realistic expectations: the unseen-domain accuracy should rise noticeably once the model has seen five or more distinct human domains, and the "polished = AI" bias should shrink once polished human text and casual AI text are both present. Whether it reaches your target is measured by the unseen-domain table, not guessed. If it stalls, the next lever is a fine-tuned Bangla transformer (`scripts/finetune_banglabert.py`), evaluated on the same held-out data.
