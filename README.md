# LANESIGHT_UI (Streamlit)

Thin UI layer for the existing **LANESIGHT** vision engine.

- UI responsibilities: video upload, frame iteration, user controls, visualization, optional export.
- Vision responsibilities: all inference and mask generation in `LANESIGHT/video/infer_video.py`.

## Run

From the repo root:

```bash
python -m pip install --upgrade pip

# Install PyTorch (CPU) (required by LANESIGHT/UFLD/ViT)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install UI + remaining runtime deps
python -m pip install -r LANESIGHT_UI/requirements.txt

python -m streamlit run LANESIGHT_UI/app.py
```

Then open the Streamlit URL shown in the terminal.
