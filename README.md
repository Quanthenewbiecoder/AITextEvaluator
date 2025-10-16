# Anti-AI Content Detector (academic focus)

Detect AI-like vs human-like text with a simple classical ML pipeline.
Includes collectors for arXiv/PubMed (human academic prose) and a synthetic AI-like generator.

## Quick start

```bash
# install
python -m venv .venv
source .venv/bin/activate  # (Windows: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt

# train on your own data
python src/anti_ai_toolkit.py --train collectors\your_data.csv --cv 5 --dedupe --approx-dedupe

# check a paragraph
python src/anti_ai_toolkit.py --use-model anti_ai_model.joblib --check "Paste text..." --rewrite
