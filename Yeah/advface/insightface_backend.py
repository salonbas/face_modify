"""相容層（deprecated）：請改從 advface.models.insightface_app 匯入。"""

from advface.models.insightface_app import (  # noqa: F401
    create_face_app,
    get_embedding_from_bgr,
    get_embedding_from_path,
    pick_best_face,
)
