from pathlib import Path
import csv
import numpy as np
import cv2
import torch
from insightface.app import FaceAnalysis

root = Path(r"C:\Users\antho\scikit_learn_data\lfw_home\lfw_funneled")

pairs = [
    ("Aaron_Peirsol", "Aaron_Peirsol_0001.jpg",
     "Aaron_Peirsol", "Aaron_Peirsol_0002.jpg", "same"),
    ("Aaron_Peirsol", "Aaron_Peirsol_0001.jpg",
     "Aaron_Eckhart", "Aaron_Eckhart_0001.jpg", "different"),
]

print("CUDA available:", torch.cuda.is_available())

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

def embedding(person, filename):
    path = root / person / filename
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(str(path))

    faces = app.get(img)
    if not faces:
        raise ValueError(f"No face detected: {path}")

    face = max(faces, key=lambda f: f.det_score)
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    return emb / np.linalg.norm(emb)

results = []

for p1, f1, p2, f2, label in pairs:
    try:
        e1 = embedding(p1, f1)
        e2 = embedding(p2, f2)
        cosine = float(np.dot(e1, e2))
        results.append([f1, f2, label, cosine])
        print(f"{label}: {cosine:.6f}")
    except Exception as exc:
        print(f"FAILED: {f1} / {f2}: {exc}")

out = Path("results/lfw_arcface_baseline.csv")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image1", "image2", "pair_type", "cosine"])
    writer.writerows(results)

print("Saved:", out.resolve())
print("Successful pairs:", len(results))
