from dataclasses import replace

import numpy as np


def _document(*, layer_id="a", name="Layer", mask=None, revision=0):
    from ditherzam.composition import Look
    from ditherzam.layers import (
        CanvasSpec,
        Layer,
        LayerDocument,
        LayerSource,
    )

    rgba = np.zeros((2, 3, 4), np.uint8)
    rgba[..., 3] = 255
    source = LayerSource(np.zeros((2, 3), np.float32), rgba)
    layer = Layer(
        layer_id, name, Look(name, {"value": name}),
        source=source, raster_mask=mask,
    )
    return LayerDocument(CanvasSpec(3, 2), (layer,), (layer_id,), revision)


def test_history_undo_redo_rebases_live_look_and_source_and_revision():
    from ditherzam.composition import Look
    from ditherzam.layers import LayerDocumentHistory

    before = _document(name="Before", revision=4)
    after = before.replace(0, replace(before.layers[0], name="After"))
    history = LayerDocumentHistory()
    assert history.record(before, after, "Rename Layer") is True

    live = after.replace(
        0, replace(after.layers[0], look=Look("live", {"value": "live"})))
    restored = history.undo(live)
    assert restored.layers[0].name == "Before"
    assert restored.layers[0].look.preset == {"value": "live"}
    assert restored.layers[0].source is live.layers[0].source
    assert restored.selected_ids == live.selected_ids
    assert restored.revision > live.revision
    redone = history.redo(restored)
    assert redone.layers[0].name == "After"
    assert redone.layers[0].look is restored.layers[0].look
    assert redone.revision > restored.revision


def test_history_resurrects_deleted_layer_from_stored_values():
    from ditherzam.layers import LayerDocumentHistory

    before = _document(name="Stored")
    after = before.remove(0)
    history = LayerDocumentHistory()
    history.record(before, after, "Delete Layer")
    restored = history.undo(after)
    assert restored.layers[0] is before.layers[0]
    assert restored.selected_ids == ("a",)


def test_history_branch_clears_redo_and_selection_only_is_noop():
    from ditherzam.layers import LayerDocumentHistory

    before = _document()
    after = before.replace(0, replace(before.layers[0], opacity=40))
    history = LayerDocumentHistory()
    history.record(before, after, "Opacity")
    live = history.undo(after)
    assert history.can_redo
    selected = live.select(0)
    assert history.record(live, selected, "Select") is False
    branched = live.replace(0, replace(live.layers[0], visible=False))
    history.record(live, branched, "Visibility")
    assert not history.can_redo


def test_history_bounds_count_unique_payload_bytes_and_rejects_oversized():
    from ditherzam.layers import LayerDocumentHistory, RasterLayerMask

    base = _document()
    mask = RasterLayerMask(np.zeros((2, 3), np.uint8))
    one = base.replace(0, replace(base.layers[0], raster_mask=mask))
    two = one.replace(0, replace(one.layers[0], opacity=25))
    history = LayerDocumentHistory(max_entries=1, max_bytes=54)
    assert history.record(base, one, "Mask") is True
    assert history.retained_bytes == 54
    assert history.record(one, two, "Opacity") is True
    assert history.entry_count == 1
    assert history.retained_bytes == 54

    too_small = LayerDocumentHistory(max_bytes=53)
    assert too_small.record(base, one, "Huge Mask") is False
    assert too_small.entry_count == 0


def test_history_defaults_and_labels():
    from ditherzam.layers import LayerDocumentHistory

    history = LayerDocumentHistory()
    assert history.max_entries == 64
    assert history.max_bytes == 128 * 1024 * 1024
    assert history.undo_label is None
    before = _document()
    after = before.replace(0, replace(before.layers[0], name="Ink"))
    history.record(before, after, "Rename Layer")
    assert history.undo_label == "Rename Layer"
    assert history.redo_label is None


def test_count_eviction_discards_oldest_entry_deterministically():
    from ditherzam.layers import LayerDocumentHistory

    zero = _document(name="Zero")
    one = zero.replace(0, replace(zero.layers[0], name="One"))
    two = one.replace(0, replace(one.layers[0], name="Two"))
    three = two.replace(0, replace(two.layers[0], name="Three"))
    history = LayerDocumentHistory(max_entries=2)
    history.record(zero, one, "One")
    history.record(one, two, "Two")
    history.record(two, three, "Three")
    assert history.entry_count == 2
    restored = history.undo(three)
    assert restored.layers[0].name == "Two"
    restored = history.undo(restored)
    assert restored.layers[0].name == "One"
    assert history.undo(restored) is restored


def test_byte_eviction_counts_shared_payload_once_across_entries():
    from ditherzam.layers import LayerDocumentHistory, RasterLayerMask

    base = _document()
    masks = [
        RasterLayerMask(np.full((2, 3), value, np.uint8))
        for value in (1, 2, 3, 4)
    ]
    one = base.replace(0, replace(base.layers[0], raster_mask=masks[0]))
    two = one.replace(0, replace(one.layers[0], raster_mask=masks[1]))
    three = two.replace(0, replace(two.layers[0], raster_mask=masks[2]))
    four = three.replace(0, replace(three.layers[0], raster_mask=masks[3]))
    history = LayerDocumentHistory(max_bytes=66)
    assert history.record(base, one, "One")
    assert history.record(one, two, "Two")
    assert history.retained_bytes == 60
    assert history.record(two, three, "Three")
    assert history.entry_count == 3
    assert history.retained_bytes == 66
    assert history.record(three, four, "Four")
    assert history.entry_count == 2
    assert history.retained_bytes == 66
    restored = history.undo(four)
    restored = history.undo(restored)
    assert restored.layers[0].raster_mask is masks[1]
