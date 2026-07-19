# Deploying a Live Demo (Streamlit Community Cloud)

Right now this project is just a folder someone has to download and run
themselves, which is a lot to ask of anyone skimming a resume or GitHub repo.
Putting up a live link fixes that, and Streamlit Community Cloud makes it
free and pretty painless since it deploys straight from a GitHub repo.

Notes below are what I had to figure out getting this working.

## 1. What actually needs to be in the repo

The cloud app can't run the training pipeline itself — there's no way to
interactively pull the Kaggle dataset there. So instead of just pushing code,
I had to include the already-computed outputs too:

```
data/subset/                    (the ~2000 curated images + subset_metadata.csv)
embeddings/                     (baseline_embeddings.npy, finetuned_embeddings.npy,
                                  siamese_embeddings.npy, embeddings_index.csv)
models/finetuned_backbone.h5
models/siamese_embedding_model.h5
```

Didn't push `data/raw/` at all — that's the full Kaggle dataset and it's not
needed once the subset's already built.

## 2. GitHub's file size limit will bite you

GitHub flat out rejects anything over 100MB in a normal commit. My `.h5`
model files are right around 90MB, close enough to the limit that it's worth
just using Git LFS from the start instead of hoping a plain push works:

```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.npy"
git add .gitattributes
git add models/ embeddings/
git commit -m "Add trained models and embeddings via LFS"
```

## 3. .gitignore

Nothing fancy, just keeping the repo from filling up with stuff that
shouldn't be there:

```
venv/
__pycache__/
*.pyc
data/raw/
.streamlit/secrets.toml
```

## 4. packages.txt

Streamlit Cloud runs on a stripped-down Debian image, and OpenCV sometimes
wants a system graphics library even with the "headless" pip package. One
line fixes it — save this as `packages.txt` in the repo root:

```
libgl1
```

## 5. Push it

```bash
git init
git add .
git commit -m "Visual product recommender"
git branch -M main
git remote add origin https://github.com/<your-username>/visual-product-recommender.git
git push -u origin main
```

## 6. Deploy

1. Go to share.streamlit.io, sign in with GitHub
2. New app → pick the repo, branch `main`
3. Main file path: `app/streamlit_app.py`
4. Deploy

First build takes a few minutes since it's installing TensorFlow from
scratch. After that you get a URL like `your-app-name.streamlit.app` — stick
that at the top of the README and wherever else people will actually see it.

## 7. Watch out for the RAM limit

The free tier only gives you about 1GB of RAM, and loading three ResNet50
models at once gets close to that. If it starts throwing memory errors:
- Cut it down to two models instead of three (Baseline + Siamese is a decent
  pair, since Siamese's embeddings are the smallest)
- Or just leave all three in and accept that the first time each model gets
  used it's slow while it loads, then fast after that once it's cached

## 8. After it's live

Drop this at the very top of the README:
```markdown
🔗 **[Live Demo](https://your-app-name.streamlit.app)**
```
Honestly the one line most people will actually click.

