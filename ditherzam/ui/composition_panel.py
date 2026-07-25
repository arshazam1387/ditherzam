"""Compact widget-only surface for editing a still-image Look composition."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class CompositionPanel(QWidget):
    """Look list, transition controls, scrubber, playback, and export intents.

    Composition state and rendering deliberately live in the controller.  This
    class owns only presentation state and translates native widget events into
    stable, index-based signals.
    """

    capture_requested = Signal()
    remove_requested = Signal(int)
    clip_duration_changed = Signal(int, int)
    transition_changed = Signal(int, str, int)
    frame_changed = Signal(int)
    export_requested = Signal(int)
    play_toggled = Signal(bool)

    _EMPTY_TEXT = (
        "Capture the current editor settings to create your first Look."
    )
    _TRANSITIONS = (
        "crossfade",
        "dither-dissolve",
        "spatial-wipe",
        "param-morph",
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, int, int]] = []

        self.empty_label = QLabel(self._EMPTY_TEXT)
        self.empty_label.setWordWrap(True)
        self.empty_label.setAccessibleName("Look Composer empty state")

        self.look_list = QListWidget()
        self.look_list.setAccessibleName("Composition Looks")

        self.capture_btn = QPushButton("Capture Current Look")
        self.remove_btn = QPushButton("Remove Look")

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 3600)
        self.duration_spin.setValue(24)
        self.duration_spin.setAccessibleName("Selected Look duration")
        self.duration_spin.setSuffix(" frames")

        self.transition_combo = QComboBox()
        self.transition_combo.addItems(self._TRANSITIONS)
        self.transition_combo.setAccessibleName("Incoming transition")

        self.transition_duration_spin = QSpinBox()
        self.transition_duration_spin.setRange(1, 3600)
        self.transition_duration_spin.setValue(6)
        self.transition_duration_spin.setAccessibleName(
            "Incoming transition duration"
        )
        self.transition_duration_spin.setSuffix(" frames")

        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setAccessibleName("Composition frame")
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setAccessibleName("Current composition frame")
        self.export_btn = QPushButton("Export Current Frame")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Look Composer status")
        self.status_label.setProperty("error", False)

        self._timer = QTimer(self)
        self._timer.setInterval(1000 // 24)

        self._build_layout()
        self._connect()
        self._refresh_clip_state()

    def _build_layout(self) -> None:
        actions = QHBoxLayout()
        actions.addWidget(self.capture_btn)
        actions.addWidget(self.remove_btn)
        actions.addStretch(1)

        clip_controls = QHBoxLayout()
        duration_label = QLabel("Look duration")
        duration_label.setBuddy(self.duration_spin)
        transition_label = QLabel("Incoming transition")
        transition_label.setBuddy(self.transition_combo)
        transition_duration_label = QLabel("Transition duration")
        transition_duration_label.setBuddy(self.transition_duration_spin)
        clip_controls.addWidget(duration_label)
        clip_controls.addWidget(self.duration_spin)
        clip_controls.addSpacing(8)
        clip_controls.addWidget(transition_label)
        clip_controls.addWidget(self.transition_combo)
        clip_controls.addWidget(transition_duration_label)
        clip_controls.addWidget(self.transition_duration_spin)
        clip_controls.addStretch(1)

        transport = QHBoxLayout()
        transport.addWidget(self.play_btn)
        transport.addWidget(self.frame_slider, 1)
        transport.addWidget(self.frame_label)
        transport.addWidget(self.export_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(6)
        root.addWidget(self.empty_label)
        root.addLayout(actions)
        root.addWidget(self.look_list)
        root.addLayout(clip_controls)
        root.addLayout(transport)
        root.addWidget(self.status_label)

        QWidget.setTabOrder(self.capture_btn, self.remove_btn)
        QWidget.setTabOrder(self.remove_btn, self.look_list)
        QWidget.setTabOrder(self.look_list, self.duration_spin)
        QWidget.setTabOrder(self.duration_spin, self.transition_combo)
        QWidget.setTabOrder(
            self.transition_combo, self.transition_duration_spin
        )
        QWidget.setTabOrder(self.transition_duration_spin, self.play_btn)
        QWidget.setTabOrder(self.play_btn, self.frame_slider)
        QWidget.setTabOrder(self.frame_slider, self.export_btn)

    def _connect(self) -> None:
        self.capture_btn.clicked.connect(self.capture_requested.emit)
        self.remove_btn.clicked.connect(self._request_remove)
        self.look_list.currentRowChanged.connect(self._selection_changed)
        self.duration_spin.valueChanged.connect(self._duration_edited)
        self.transition_combo.currentTextChanged.connect(
            self._transition_edited
        )
        self.transition_duration_spin.valueChanged.connect(
            self._transition_edited
        )
        self.play_btn.toggled.connect(self._play_changed)
        self.frame_slider.valueChanged.connect(self._frame_changed)
        self.export_btn.clicked.connect(
            lambda: self.export_requested.emit(self.frame_slider.value())
        )
        self._timer.timeout.connect(self._advance)

    def set_clips(
        self,
        rows: list[tuple[str, int, int]],
        selected: int | None,
    ) -> None:
        """Replace the displayed clips without publishing user-edit signals."""
        clean_rows: list[tuple[str, int, int]] = []
        for name, start, end in rows:
            clean_rows.append((str(name), int(start), int(end)))
        self._rows = clean_rows

        with QSignalBlocker(self.look_list):
            self.look_list.clear()
            for name, start, end in self._rows:
                self.look_list.addItem(f"{name}  ·  frames {start + 1}–{end}")
            if selected is not None and 0 <= int(selected) < len(self._rows):
                self.look_list.setCurrentRow(int(selected))
            else:
                self.look_list.setCurrentRow(-1)
        self._refresh_clip_state()

    def set_frame_range(self, total: int, current: int) -> None:
        total = max(0, int(total))
        maximum = max(0, total - 1)
        current = max(0, min(int(current), maximum))
        with QSignalBlocker(self.frame_slider):
            self.frame_slider.setRange(0, maximum)
            self.frame_slider.setValue(current)
        self.frame_label.setText(
            f"{current + 1} / {total}" if total else "0 / 0"
        )

    def set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(str(text))
        self.status_label.setProperty("error", bool(error))
        style = self.status_label.style()
        style.unpolish(self.status_label)
        style.polish(self.status_label)
        self.status_label.update()

    def selected_index(self) -> int | None:
        row = self.look_list.currentRow()
        return row if 0 <= row < len(self._rows) else None

    def stop_playback(self) -> None:
        if self.play_btn.isChecked():
            self.play_btn.setChecked(False)
        else:
            self._timer.stop()
            self.play_btn.setText("Play")

    def _refresh_clip_state(self) -> None:
        index = self.selected_index()
        has_clips = bool(self._rows)
        has_selection = index is not None
        self.empty_label.setVisible(not has_clips)
        self.look_list.setVisible(has_clips)
        self.remove_btn.setEnabled(has_selection)
        self.duration_spin.setEnabled(has_selection)
        incoming = has_selection and index != 0
        self.transition_combo.setEnabled(incoming)
        self.transition_duration_spin.setEnabled(incoming)
        self.play_btn.setEnabled(has_clips)
        self.export_btn.setEnabled(has_clips)
        self.frame_slider.setEnabled(has_clips)

        if index is not None:
            _name, start, end = self._rows[index]
            with QSignalBlocker(self.duration_spin):
                self.duration_spin.setValue(max(1, min(3600, end - start)))
        if not has_clips:
            self.stop_playback()

    def _selection_changed(self, _row: int) -> None:
        self._refresh_clip_state()

    def _request_remove(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.remove_requested.emit(index)

    def _duration_edited(self, duration: int) -> None:
        index = self.selected_index()
        if index is not None:
            self.clip_duration_changed.emit(index, int(duration))

    def _transition_edited(self, _value: object = None) -> None:
        index = self.selected_index()
        if index is not None and index > 0:
            self.transition_changed.emit(
                index,
                self.transition_combo.currentText(),
                self.transition_duration_spin.value(),
            )

    def _play_changed(self, playing: bool) -> None:
        playing = bool(playing)
        self.play_btn.setText("Pause" if playing else "Play")
        if playing:
            self._timer.start()
        else:
            self._timer.stop()
        self.play_toggled.emit(playing)

    def _frame_changed(self, frame: int) -> None:
        total = self.frame_slider.maximum() + 1 if self._rows else 0
        self.frame_label.setText(
            f"{int(frame) + 1} / {total}" if total else "0 / 0"
        )
        self.frame_changed.emit(int(frame))

    def _advance(self) -> None:
        total = self.frame_slider.maximum() + 1
        if not self._rows or total <= 0:
            self.stop_playback()
            return
        self.frame_slider.setValue((self.frame_slider.value() + 1) % total)
