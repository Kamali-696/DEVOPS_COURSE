# MLOps CI/CD — GitHub Actions + Hugging Face Hub

Every push to `main` trains a model, evaluates it, and — only if it passes
an accuracy threshold — deploys it to Hugging Face Hub automatically.

## How it works

```
push/PR to GitHub
        │
        ▼
  ┌──────┐
  │ test │   pytest sanity checks
  └──┬───┘
     ▼
  ┌────────────────────┐
  │ train_and_evaluate  │  prepare → train → evaluate
  └──────┬──────────────┘  ← fails here if accuracy < params.yaml threshold
         ▼
  ┌─────────────────────┐
  │ deploy_to_huggingface │  only on push to main, only if gate passed
  └─────────────────────┘  re-trains + pushes model to your HF repo
```


## Setup

### 1. Create a Hugging Face account + token
- Go to https://huggingface.co/settings/tokens
- Create a token with **Write** access
- Copy it — you'll paste it into GitHub next

### 2. Create the GitHub repo and push this code
```bash
cd mlops-simple
git init
git add .
git commit -m "Initial MLOps pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 3. Add your HF token as a GitHub secret
In your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**
- Name: `HF_TOKEN`
- Value: the token from step 1

### 4. Add your target HF repo as a GitHub variable
Same page, **Variables** tab → **New repository variable**
- Name: `HF_REPO_ID`
- Value: `your-hf-username/your-model-name` (e.g. `mupalaniappan/breast-cancer-rf`)
  — it'll be created automatically on first successful run if it doesn't exist yet.

### 5. Push to main
That's it — the workflow runs automatically. Check the **Actions** tab in
GitHub to watch it, and once `deploy_to_huggingface` finishes, your model is
live at `https://huggingface.co/your-hf-username/your-model-name`.

## Adjusting the quality gate

Edit `evaluate.min_accuracy` in `params.yaml`. Anything scoring below that
never reaches Hugging Face — the `deploy_to_huggingface` job simply doesn't run.

## Try it locally first (optional but recommended)

```bash
pip install -r requirements.txt
python src/prepare.py
python src/train.py
python src/evaluate.py
cat metrics.json
```
