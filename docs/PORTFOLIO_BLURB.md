# Portfolio copy for Prohori (pick the length you need)

## One line
Prohori is a Bangla AI-text detector that explains its scores, abstains when unsure, and publishes its own error rates; it ships as a web app and a Chrome extension.

## Short card (about 60 words)
Prohori scores how much Bangla text reads like AI writing and shows the measurable reasons, passage by passage. It was built with leakage-safe evaluation (pair-grouped splits, cross-fitted features), exposed and fixed a "polished writing = AI" bias with a counterfactual stress test, and reports its failure modes openly: 76.5% cross-validated accuracy, 4.1% of human text wrongly flagged.

## Project page paragraph (about 130 words)
Most AI-text detectors give a verdict and hide their error rates; for a student, a false accusation is the costly mistake. Prohori is built around that. It combines 20 explainable stylometric features with character and word n-gram models in an XGBoost classifier, scores long texts in 60-word passages, and refuses to flag short or ambiguous text. Evaluation never lets a human text and its AI rewrite straddle a split. A counterfactual test revealed the first model treated polished spelling and typography as AI (human false flags jumped from 7.6% to 18.5% after cosmetic edits), so I added a canonicalization layer and published the resulting trade-off. The app reports leave-one-generator-out and leave-one-domain-out results, including where it is weak. Stack: Python, scikit-learn, XGBoost, FastAPI, Docker on Render, Chrome Manifest V3.

## CV bullets
- Built and deployed a Bangla AI-text detector (FastAPI, XGBoost, Docker) with a Chrome extension; 76.5% cross-validated accuracy, 4.1% human false-flag rate, 88.4% precision when it answers.
- Found and fixed a "polished writing = AI" shortcut via counterfactual testing (cosmetic edits raised human false flags from 7.6% to 18.5%); added a canonicalization layer, verified invariant by unit tests.
- Designed leakage-safe evaluation: pair-grouped splits, cross-fitted features, leave-one-generator-out and leave-one-domain-out tests; published weaknesses in the product itself.
- Cut the runtime from heavy NLP dependencies to a 270 MB-RAM service by replacing bnlp/shap with verified lightweight equivalents (token-identical on 3,987 texts).

## Talking points for interviews
- Why abstention beats a forced binary answer when errors have unequal costs.
- How a dataset confound (spelling era, typography) can inflate accuracy, and how a counterfactual test exposes it.
- Why scoring in passages fixed calibration drift on long texts.
- What I would do next: more human domains, real student essays, a fine-tuned Bangla transformer blended in only if it wins on the same held-out data.
