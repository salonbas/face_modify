from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

root = Path(r"C:\Users\antho\scikit_learn_data\lfw_home\lfw_funneled")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
    keep_all=False,
)

model = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)

pairs = [
    ("Aaron_Peirsol", "Aaron_Peirsol_0001.jpg",
     "Aaron_Peirsol", "Aaron_Peirsol_0002.jpg", "same"),
    ("Aaron_Peirsol", "Aaron_Peirsol_0001.jpg",
     "Aaron_Eckhart", "Aaron_Eckhart_0001.jpg", "different"),
]

def embedding(person, filename):
    path = root / person / filename

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(str(path))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = mtcnn(Image.fromarray(image_rgb))

    if face is None:
        raise ValueError(f"No face detected: {path}")

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(face)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    return emb.cpu().numpy().reshape(-1)

results = []

for p1, f1, p2, f2, label in pairs:
    try:
        e1 = embedding(p1, f1)
        e2 = embedding(p2, f2)

        similarity = float(np.dot(e1, e2))

        results.append([f1, f2, label, similarity])
        print(f"{label}: {similarity:.6f}")

    except Exception as exc:
        print("FAILED:", str(exc))

out = Path("results/lfw_facenet_baseline.csv")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image1", "image2", "pair_type", "cosine"])
    writer.writerows(results)

print("Saved:", out.resolve())
print("Successful pairs:", len(results))
