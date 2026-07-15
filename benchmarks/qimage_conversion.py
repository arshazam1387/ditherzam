"""Measure and safety-check NumPy -> QImage ownership strategies.

Run from the repository root with::

    QT_QPA_PLATFORM=offscreen .venv/Scripts/python benchmarks/qimage_conversion.py

The benchmark intentionally keeps the production converter unchanged.  The
``borrowed`` strategy demonstrates why simply removing ``QImage.copy()`` is not
an acceptable optimization; ``numpy_owned`` measures the alternative of making
the one required copy in NumPy and letting PySide retain that private buffer.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
import weakref
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ditherzam.ui.convert import numpy_to_qimage, qimage_to_numpy


def _borrowed(array: np.ndarray) -> QImage:
    array = np.ascontiguousarray(array[:, :, :3])
    height, width = array.shape[:2]
    return QImage(
        array.data, width, height, 3 * width, QImage.Format.Format_RGB888
    )


def _numpy_owned(array: np.ndarray) -> QImage:
    owned = np.array(array[:, :, :3], dtype=np.uint8, order="C", copy=True)
    height, width = owned.shape[:2]
    return QImage(
        owned.data, width, height, 3 * width, QImage.Format.Format_RGB888
    )


class _Emitter(QObject):
    ready = Signal(QImage)


class _Receiver(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.image: QImage | None = None

    @Slot(QImage)
    def accept(self, image: QImage) -> None:
        self.image = image


def check_safety(app: QApplication) -> None:
    source = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    expected = source.copy()

    borrowed = _borrowed(source)
    source[:] = 0
    assert not np.array_equal(qimage_to_numpy(borrowed), expected), (
        "borrowed QImage unexpectedly isolated source mutation"
    )

    for converter in (numpy_to_qimage, _numpy_owned):
        source = expected.copy()
        source_ref = weakref.ref(source)
        image = converter(source)
        source[:] = 0
        del source
        gc.collect()
        assert source_ref() is None
        np.testing.assert_array_equal(qimage_to_numpy(image), expected)

        copied = QImage(image)
        del image
        gc.collect()
        np.testing.assert_array_equal(qimage_to_numpy(copied), expected)

        emitter = _Emitter()
        receiver = _Receiver()
        emitter.ready.connect(receiver.accept, Qt.ConnectionType.QueuedConnection)
        emitter.ready.emit(copied)
        del copied
        gc.collect()
        app.processEvents()
        assert receiver.image is not None
        np.testing.assert_array_equal(qimage_to_numpy(receiver.image), expected)

        # QPixmap creation stays on the QApplication/GUI thread.  Dropping the
        # queued QImage afterwards must not invalidate the pixmap's pixels.
        pixmap = QPixmap.fromImage(receiver.image)
        receiver.image = None
        gc.collect()
        assert not pixmap.isNull()
        np.testing.assert_array_equal(qimage_to_numpy(pixmap.toImage()), expected)

    gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
    noncontiguous = expected[:, ::2, :]
    for value in (gray, noncontiguous):
        image = numpy_to_qimage(value)
        assert image.width() == value.shape[1]
        assert image.height() == value.shape[0]


def measure(iterations: int) -> None:
    rng = np.random.default_rng(20260709)
    sizes = (("cap-720", 720, 1280), ("1080p", 1080, 1920), ("4k", 2160, 3840))
    print("size,strategy,median_ms,p95_ms")
    for label, height, width in sizes:
        source = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for name, converter in (
            ("production_qimage_copy", numpy_to_qimage),
            ("private_numpy_copy", _numpy_owned),
        ):
            converter(source)
            samples = []
            for _ in range(iterations):
                started = time.perf_counter()
                image = converter(source)
                samples.append((time.perf_counter() - started) * 1000)
                assert image.width() == width
            ordered = sorted(samples)
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            print(
                f"{label},{name},{statistics.median(samples):.3f},{p95:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    check_safety(app)
    print("safety checks: passed")
    measure(args.iterations)


if __name__ == "__main__":
    main()
