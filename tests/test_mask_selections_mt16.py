import numpy as np
import pytest

from ditherzam.layers.selection import (
    SelectionError,
    SelectionOperation,
    SelectionShape,
    TemporarySelection,
    combine_selection,
    rasterize_selection,
    restrict_mask_edit,
    selection_to_source,
)


def test_temporary_selection_is_owned_grayscale_document_state():
    pixels = np.array([[0, 128, 255]], np.uint8)
    selection = TemporarySelection(pixels)
    pixels[:] = 0
    assert selection.pixels.tolist() == [[0, 128, 255]]
    assert not selection.pixels.flags.writeable
    with pytest.raises(SelectionError):
        TemporarySelection(np.zeros((1, 1), np.float32))


def test_rectangle_and_ellipse_are_soft_and_deterministic():
    rectangle = rasterize_selection(
        (4, 4), SelectionShape.RECTANGLE, (0.25, 0.25, 2.75, 2.75))
    repeated = rasterize_selection(
        (4, 4), SelectionShape.RECTANGLE, (0.25, 0.25, 2.75, 2.75))
    ellipse = rasterize_selection(
        (5, 5), SelectionShape.ELLIPSE, (0.25, 0.25, 4.75, 4.75))
    assert np.array_equal(rectangle, repeated)
    assert np.any((rectangle > 0) & (rectangle < 255))
    assert ellipse[2, 2] == 255
    assert ellipse[0, 0] < ellipse[2, 2]


def test_selection_operations_are_exact_and_only_three():
    assert [item.value for item in SelectionOperation] == [
        "replace", "add", "subtract"]
    current = TemporarySelection(np.array([[0, 100, 250]], np.uint8))
    candidate = np.array([[20, 200, 20]], np.uint8)
    assert combine_selection(
        current, candidate, SelectionOperation.ADD
    ).pixels.tolist() == [[20, 255, 255]]
    assert combine_selection(
        current, candidate, SelectionOperation.SUBTRACT
    ).pixels.tolist() == [[0, 0, 230]]
    assert combine_selection(
        current, candidate, SelectionOperation.REPLACE
    ).pixels.tolist() == [[20, 200, 20]]
    assert combine_selection(
        None, candidate, SelectionOperation.ADD
    ).pixels.tolist() == [[20, 200, 20]]
    assert combine_selection(
        None, candidate, SelectionOperation.SUBTRACT
    ).pixels.tolist() == [[0, 0, 0]]


def test_inverse_xy_scale_mapping_preserves_soft_coverage():
    selection = TemporarySelection(np.array([
        [0, 0, 0, 0],
        [0, 64, 128, 0],
        [0, 192, 255, 0],
        [0, 0, 0, 0],
    ], np.uint8))
    identity = selection_to_source(
        selection, (4, 4), layer_x=0, layer_y=0, scale_x=1, scale_y=1)
    assert np.array_equal(identity, selection.pixels)
    shifted = selection_to_source(
        selection, (2, 2), layer_x=1, layer_y=1, scale_x=1, scale_y=1)
    assert shifted.tolist() == [[64, 128], [192, 255]]
    scaled = selection_to_source(
        selection, (2, 2), layer_x=0, layer_y=0, scale_x=2, scale_y=2)
    assert scaled.tolist() == [[16, 32], [48, 64]]


def test_selection_restricted_edit_keeps_endpoints_and_soft_blend():
    before = np.array([[20, 20, 20]], np.uint8)
    edited = np.array([[220, 220, 220]], np.uint8)
    coverage = np.array([[0, 128, 255]], np.uint8)
    assert restrict_mask_edit(
        before, edited, coverage).tolist() == [[20, 120, 220]]


def test_selection_state_is_absent_from_document_serialization():
    from ditherzam.layers import LayerStack, layer_stack_to_dict

    payload = layer_stack_to_dict(LayerStack())
    assert "selection" not in repr(payload).lower()


def test_controller_selection_is_temporary_from_selection_and_not_history(
    qapp_fixture,
):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.zeros((4, 4, 4), np.uint8)
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((4, 4), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )
    controller.initialize_source_layer()
    document_before_selection = controller.document
    assert controller.update_temporary_selection(
        SelectionShape.RECTANGLE, (1, 1, 3, 3))
    assert controller.document is document_before_selection
    assert controller.create_mask_from_selection()
    assert controller.document.layers[0].raster_mask.pixels.tolist() == [
        [0, 0, 0, 0],
        [0, 255, 255, 0],
        [0, 255, 255, 0],
        [0, 0, 0, 0],
    ]
    assert controller.undo()
    assert controller.document.layers[0].raster_mask is None
    assert controller.temporary_selection is not None


def test_controller_selection_tracks_layer_transform_mapping(qapp_fixture):
    from dataclasses import replace
    from ditherzam.dithering import registry
    from ditherzam.layers import LayerTransform
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    rgba = np.zeros((2, 2, 4), np.uint8)
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((2, 2), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
    )
    controller.initialize_source_layer()
    layer = controller.document.layers[0]
    controller._document = replace(
        controller.document,
        layers=(replace(
            layer, transform=LayerTransform(x=1, y=1, scale_x=2, scale_y=2)),),
    )
    assert controller.update_temporary_selection(
        SelectionShape.RECTANGLE, (1, 1, 3, 3))
    coverage = controller.selection_coverage_for_active_source()
    assert coverage.tolist() == [[64, 0], [0, 0]]


def test_viewport_selection_drag_emits_document_coordinates_after_zoom_and_pan(
    qapp_fixture,
):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtTest import QTest
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView(enable_inertia=False)
    view.resize(220, 220)
    view.show()
    view.set_pixmap(QPixmap(400, 400), logical_size=(400, 400))
    view.zoom_in()
    view.horizontalScrollBar().setValue(25)
    view.verticalScrollBar().setValue(30)
    view.set_selection_tool("ellipse")
    qapp_fixture.processEvents()
    expected_start = view.mapToScene(QPoint(70, 80))
    expected_end = view.mapToScene(QPoint(150, 160))
    seen = []
    view.selection_dragged.connect(lambda *args: seen.append(args))
    QTest.mousePress(
        view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(70, 80))
    QTest.mouseMove(view.viewport(), QPoint(150, 160))
    QTest.mouseRelease(
        view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(150, 160))
    assert seen and seen[-1][0] == "ellipse"
    assert seen[-1][1:] == pytest.approx((
        expected_start.x(), expected_start.y(),
        expected_end.x(), expected_end.y(),
    ), abs=0.75)


def test_generated_mask_replace_is_restricted_by_soft_selection(qapp_fixture):
    from ditherzam.layers import (
        MaskCandidate, MaskCandidateOrigin, MaskCombinationMode)
    from tests.test_raster_mask_lifecycle import _controller
    from ditherzam.ui.layers_panel import LayersPanel

    controller = _controller(LayersPanel())
    controller.initialize_source_layer()
    controller.propose_active_raster_mask(
        np.full((3, 4), 100, np.uint8), "base")
    controller.update_temporary_selection(
        SelectionShape.RECTANGLE, (1, 0, 3, 3))
    source = controller.document.layers[0].source
    candidate = MaskCandidate(
        np.full((3, 4), 200, np.uint8),
        MaskCandidateOrigin.PATTERN,
        "Pattern",
        source.source_identity,
    )
    assert not controller.propose_mask_candidate(
        candidate, MaskCombinationMode.REPLACE)
    token = controller.mask_replacement_proposal.token
    assert controller.confirm_mask_replacement(token)
    assert controller.document.layers[0].raster_mask.pixels.tolist() == [
        [100, 200, 200, 100],
        [100, 200, 200, 100],
        [100, 200, 200, 100],
    ]


def test_opening_fresh_document_clears_selection_and_overlay(qapp_fixture):
    from ditherzam.dithering import registry
    from ditherzam.ui.layers_controller import LayersController
    from ditherzam.ui.layers_panel import LayersPanel

    clears = []
    rgba = np.zeros((2, 2, 4), np.uint8)
    controller = LayersController(
        LayersPanel(), registry,
        preset_provider=lambda: {
            "adjustments": {"contrast": 50},
            "dither": {"style": "None", "scale": 1, "params": {}}},
        source_provider=lambda: (np.zeros((2, 2), np.float32), rgba, None),
        cap_provider=lambda: 480,
        frame_sink=lambda _frame: None,
        apply_preset=lambda _preset: None,
        selection_overlay_clear=lambda: clears.append(True),
    )
    controller.initialize_source_layer()
    controller.update_temporary_selection(
        SelectionShape.RECTANGLE, (0, 0, 1, 1))
    controller.open_document(np.zeros((2, 2), np.float32), rgba)
    assert controller.temporary_selection is None
    assert clears


def test_completed_selection_overlay_remains_visible(qapp_fixture):
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView()
    view.install_selection_overlay(np.array([[0, 255], [128, 0]], np.uint8))
    assert view._selection_item is not None
    assert view._selection_item.isVisible()
    view.clear_selection_overlay()
    assert not view._selection_item.isVisible()


def test_viewport_gradient_drag_emits_document_geometry(qapp_fixture):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtTest import QTest
    from ditherzam.ui.viewport import CustomGraphicsView

    view = CustomGraphicsView(enable_inertia=False)
    view.resize(200, 200)
    view.show()
    view.set_pixmap(QPixmap(100, 100), logical_size=(100, 100))
    view.set_gradient_tool("linear")
    qapp_fixture.processEvents()
    start, end = QPoint(60, 70), QPoint(140, 130)
    expected_start, expected_end = view.mapToScene(start), view.mapToScene(end)
    seen = []
    view.gradient_dragged.connect(lambda *args: seen.append(args))
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert seen[-1][0] == "linear"
    assert seen[-1][1:] == pytest.approx((
        expected_start.x(), expected_start.y(),
        expected_end.x(), expected_end.y(),
    ), abs=0.75)
