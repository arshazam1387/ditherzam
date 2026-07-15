# QImage conversion experiment — 2026-07-09

Task 3.7 investigated removing the deep copy in `numpy_to_qimage()`.

## Result

The optimization is rejected. The production `QImage.copy()` remains in place.
A directly borrowed NumPy buffer is observably changed when its source is
mutated, so it violates the helper's standalone-image contract. A private NumPy
copy followed by a borrowed `QImage` passed deletion/GC, `QImage` value-copy,
queued-signal, and GUI-thread `QPixmap` checks, but did not provide a reliable
measured improvement over Qt's copy.

Representative medians on the development machine (20 iterations, offscreen
Qt platform) measured Qt-copy versus private-NumPy-copy as 0.81/0.88 ms at
720p, 2.22/2.28 ms at 1080p, and 11.35/14.26 ms at 4K. This is a regression,
not a win. The private-buffer lifetime guarantee would also depend on PySide
binding behavior rather than the explicit ownership provided by
`QImage.copy()`.

## Reproduce

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe benchmarks\qimage_conversion.py --iterations 20
```

The harness also covers source mutation and collection, non-contiguous and
grayscale inputs, queued delivery after local-image deletion, and conversion to
`QPixmap` on the application thread. `QPixmap` creation must remain at the GUI
thread boundary.
