"""Author-checkpoint loaders for AdaFace and MagFace; CPU inference only."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from advface.config import project_root


def _source_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"official model source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _adaface_state(checkpoint: Path) -> dict:
    """Read the pinned author checkpoint across PyTorch serialization versions."""
    manifest = json.loads((project_root() / "models/checkpoint_manifest.json").read_text())
    with checkpoint.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != manifest["adaface"]["sha256"]:
        raise ValueError(f"AdaFace checkpoint checksum mismatch: {checkpoint}")
    class ModelCheckpoint:
        pass
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([
            (ModelCheckpoint, "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint")
        ]):
            return torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    # PyTorch 2.2 has no safe_globals. Its weights-only loader cannot decode
    # the Lightning callback metadata, so remap that single class after hash
    # validation. The original author pickle is never accepted under another hash.
    class CheckpointUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if (module, name) == ("pytorch_lightning.callbacks.model_checkpoint", "ModelCheckpoint"):
                return ModelCheckpoint
            return super().find_class(module, name)
    pickle_module = types.ModuleType("adaface_checkpoint_pickle")
    for name in dir(pickle):
        setattr(pickle_module, name, getattr(pickle, name))
    pickle_module.Unpickler = CheckpointUnpickler
    return torch.load(checkpoint, map_location="cpu", weights_only=False,
                      pickle_module=pickle_module)["state_dict"]


class AdaFaceEmbedder:
    name = "adaface_ir50_ms1mv2"
    threshold = None

    def __init__(self) -> None:
        base = project_root() / "models/third_party/adaface"
        checkpoint = project_root() / "models/adaface/adaface_ir50_ms1mv2.ckpt"
        net = _source_module("official_adaface_net", base / "net.py")
        self.model = net.build_model("ir_50")
        state = _adaface_state(checkpoint)
        weights = {key[6:]: value for key, value in state.items() if key.startswith("model.")}
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        sys.path.insert(0, str(base))
        try:
            from face_alignment import mtcnn
            self.aligner = mtcnn.MTCNN(device="cpu", crop_size=(112, 112))
        finally:
            sys.path.remove(str(base))

    def get_embedding(self, image_bgr: np.ndarray, label: str = "image") -> np.ndarray:
        from PIL import Image
        rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        _, faces = self.aligner.align_multi(rgb, limit=1)
        if not faces:
            raise ValueError(f"AdaFace alignment failed: {label}")
        bgr = np.asarray(faces[0])[:, :, ::-1].copy()
        tensor = torch.from_numpy(bgr.transpose(2, 0, 1).copy()).unsqueeze(0).float()
        tensor = (tensor / 255.0 - 0.5) / 0.5
        with torch.inference_mode():
            embedding, _ = self.model(tensor)
        result = embedding.squeeze(0).numpy()
        if result.shape != (512,) or not np.isfinite(result).all():
            raise ValueError(f"invalid AdaFace embedding: {label}")
        return result


class MagFaceEmbedder:
    name = "magface_iresnet100_ms1mv2"
    threshold = None

    def __init__(self) -> None:
        source = project_root() / "models/third_party/magface/iresnet.py"
        checkpoint = project_root() / "models/magface/magface_iresnet100_ms1mv2.pth"
        arch = _source_module("official_magface_iresnet", source)
        self.model = arch.iresnet100(pretrained=False, num_classes=512)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        weights = {key.removeprefix("features.module."): value for key, value in state.items()
                   if key.startswith("features.module.")}
        self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        from insightface.model_zoo import get_model
        detector_path = project_root() / "models/insightface/buffalo_sc/det_500m.onnx"
        self.detector = get_model(str(detector_path), providers=["CPUExecutionProvider"])
        self.detector.prepare(ctx_id=-1, input_size=(640, 640))

    def get_embedding(self, image_bgr: np.ndarray, label: str = "image") -> np.ndarray:
        from insightface.utils import face_align
        boxes, keypoints = self.detector.detect(image_bgr, max_num=1)
        if len(boxes) != 1 or keypoints is None:
            raise ValueError(f"MagFace alignment failed: {label}")
        crop = face_align.norm_crop(image_bgr, landmark=keypoints[0], image_size=112)
        # Author inference uses cv2 BGR -> torchvision ToTensor [0,1] and
        # Normalize(mean=0,std=1), without an RGB swap or [-1,1] scaling.
        tensor = torch.from_numpy(crop.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255.0
        with torch.inference_mode():
            result = self.model(tensor).squeeze(0).numpy()
        if result.shape != (512,) or not np.isfinite(result).all():
            raise ValueError(f"invalid MagFace embedding: {label}")
        return result
