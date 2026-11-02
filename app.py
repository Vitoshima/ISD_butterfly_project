# app.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import io, torch, torchvision.transforms as T, torch.nn.functional as F
import json

BASE = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE / "templates"))

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
MAX_UPLOAD_SIZE = 5 * 1024 * 1024   # 5 MB
TOP_K = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SCRIPTED = BASE / "model_scripted.pt"
CLASS_NAMES_FILE = BASE / "class_names.json"
# ----------------------------------------

model = None
class_names = []  # Will be loaded from JSON file
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

@app.on_event("startup")
def startup_load_model():
    global model, class_names
    
    # Load class names from JSON file
    if CLASS_NAMES_FILE.exists():
        with open(CLASS_NAMES_FILE, 'r') as f:
            meta = json.load(f)
            class_names = meta['class_names']
        print(f"✅ Loaded {len(class_names)} butterfly species")
        print(f"First 5 species: {class_names[:5]}")
    else:
        raise RuntimeError("class_names.json not found. Place it in the project folder.")
    
    # Load the model
    if MODEL_SCRIPTED.exists():
        model = torch.jit.load(str(MODEL_SCRIPTED), map_location=DEVICE)
        model.to(DEVICE).eval()
        print("✅ Loaded TorchScript model.")
    else:
        raise RuntimeError("model_scripted.pt not found. Place it in the project folder.")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 5MB)")
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Can't read the image. Upload a valid image file.")
    
    # transform and run
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        topk = torch.topk(probs, k=min(TOP_K, probs.shape[1]), dim=1)
        top_probs = topk.values[0].cpu().tolist()
        top_idx = topk.indices[0].cpu().tolist()
    
    predictions = [
        {"name": class_names[int(i)], "prob": float(p)}
        for i, p in zip(top_idx, top_probs)
    ]

    out = {
        "predictions": predictions,  # <--- what your frontend expects
        "predicted_class": predictions[0]["name"],
        "probability": predictions[0]["prob"]
    }

    return JSONResponse(out)
