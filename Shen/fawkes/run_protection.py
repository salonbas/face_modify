"""Runtime wrapper that fixes Fawkes 1.0.3 against modern TensorFlow.

Fawkes 1.0.3 直接呼叫 ``tf.keras.optimizers.Adadelta`` 並把它套用在手動建立的
``tf.Variable`` 上。從 TF 2.11 起，Keras 換成新版優化器，會丟出
``KeyError: 'The optimizer cannot recognize variable Variable:0'``。

這個 wrapper 在 ``import fawkes.protection`` 之前，把 ``Adadelta`` 別名換成
``tf.keras.optimizers.legacy.Adadelta``，恢復原本能直接 ``apply_gradients`` 的行為。
這樣就不必改動 site-packages 內的檔案，環境重建後依然有效。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402
import numpy as np  # noqa: E402

_legacy = getattr(tf.keras.optimizers, "legacy", None)
if _legacy is not None and hasattr(_legacy, "Adadelta"):
    tf.keras.optimizers.Adadelta = _legacy.Adadelta

# Fawkes 1.0.3 會對不同尺寸影像的 list 直接呼叫 np.copy，新的 NumPy 會拋
# "setting an array element with a sequence"。這裡只針對該情境做相容處理。
_np_copy = np.copy


def _np_copy_compat(arr, *args, **kwargs):
    if isinstance(arr, list):
        return [_np_copy(x, *args, **kwargs) for x in arr]
    return _np_copy(arr, *args, **kwargs)


np.copy = _np_copy_compat

from fawkes.protection import main  # noqa: E402


if __name__ == "__main__":
    main(*sys.argv)
