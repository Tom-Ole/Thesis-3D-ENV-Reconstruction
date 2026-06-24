"""Capture tab: connect to Spot, pick sources, record images + LiDAR.

Capture is a continuous recording: the user drives Spot with the tablet, presses
"Start capture", and the app records images and point clouds at independent,
configurable rates until "Stop capture" is pressed.

Widget construction and Qt threading wiring live here; all robot I/O and
persistence are delegated to :class:`CaptureController` running on worker
threads so the UI never blocks.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.app_context import AppContext
from src.capture.capture_controller import CaptureController
from src.gui.workers import (
    ContinuousCaptureWorker,
    FunctionWorker,
    StateRecorderWorker,
)
from src.models import (
    CaptureBatchResult,
    ConnectResult,
)

logger = logging.getLogger(__name__)


class CaptureTab(QWidget):
    """GUI for connecting to Spot and recording images / point clouds."""

    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.config = context.config
        self.controller = CaptureController(
            context.config, context.spot, context.sessions
        )
        self.pool = QThreadPool.globalInstance()

        self.connected = False
        self.busy = False
        self.recording = False
        self.has_point_cloud = False
        self._capture_worker: Optional[ContinuousCaptureWorker] = None
        self._state_worker: Optional[StateRecorderWorker] = None

        self._build_ui()
        self._prefill_image_sources()
        self._refresh_enabled_state()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(self._build_connection_group())

        # Main content: sources expand to fill all available space;
        # controls column stays compact on the right.
        content = QHBoxLayout()
        content.setSpacing(8)
        sources = self._build_sources_panel()
        controls = self._build_controls_panel()
        controls.setMaximumWidth(340)
        content.addWidget(sources, stretch=1)
        content.addWidget(controls)
        root.addLayout(content, stretch=1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setPlaceholderText("Status and log output...")
        self.log_view.setMinimumHeight(90)
        self.log_view.setMaximumHeight(200)
        root.addWidget(self.log_view)

    def _build_connection_group(self) -> QGroupBox:
        box = QGroupBox("Connection")
        layout = QHBoxLayout(box)
        layout.setSpacing(16)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)
        self.hostname_edit = QLineEdit(self.config.robot_hostname)
        self.username_edit = QLineEdit(self.config.robot_username)
        self.password_edit = QLineEdit(self.config.robot_password)
        self.password_edit.setEchoMode(QLineEdit.Password)
        for edit in (self.hostname_edit, self.username_edit, self.password_edit):
            edit.setMinimumWidth(180)
        form.addRow("Hostname / IP:", self.hostname_edit)
        form.addRow("Username:", self.username_edit)
        form.addRow("Password:", self.password_edit)
        layout.addLayout(form, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.clicked.connect(self._on_connect)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setObjectName("disconnectBtn")
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        for btn in (self.connect_btn, self.disconnect_btn):
            btn.setMinimumWidth(110)
        btn_col.addWidget(self.connect_btn)
        btn_col.addWidget(self.disconnect_btn)
        btn_col.addStretch()
        layout.addLayout(btn_col)

        status_col = QVBoxLayout()
        status_col.setSpacing(4)
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(14, 14)
        self.status_text = QLabel("Disconnected")
        self.battery_label = QLabel("")
        dot_row = QHBoxLayout()
        dot_row.addWidget(self.status_dot)
        dot_row.addWidget(self.status_text)
        dot_row.addStretch()
        status_col.addLayout(dot_row)
        status_col.addWidget(self.battery_label)
        status_col.addStretch()
        layout.addLayout(status_col, stretch=1)

        self._set_status_indicator(False, "Disconnected")
        return box

    def _build_sources_panel(self) -> QWidget:
        """Left column: image + point-cloud source lists that expand to fill space."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        img_box = QGroupBox("Image sources")
        img_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_layout = QVBoxLayout(img_box)
        img_layout.setSpacing(6)

        img_sel_row = QHBoxLayout()
        img_sel_row.setSpacing(16)
        self.image_select_all = QCheckBox("Select all")
        self.image_select_all.toggled.connect(
            lambda checked: self._set_all_checked(self.image_list, checked)
        )
        self.image_color_only = QCheckBox("Color only")
        self.image_color_only.toggled.connect(self._on_image_color_only_toggled)
        img_sel_row.addWidget(self.image_select_all)
        img_sel_row.addWidget(self.image_color_only)
        img_sel_row.addStretch()

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.image_list.setMinimumHeight(130)
        self.image_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_layout.addLayout(img_sel_row)
        img_layout.addWidget(self.image_list)
        layout.addWidget(img_box, stretch=3)

        pc_box = QGroupBox("Point-cloud / LiDAR sources")
        pc_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pc_layout = QVBoxLayout(pc_box)
        pc_layout.setSpacing(6)
        self.pc_select_all = QCheckBox("Select all")
        self.pc_select_all.toggled.connect(
            lambda checked: self._set_all_checked(self.pc_list, checked)
        )
        self.pc_list = QListWidget()
        self.pc_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.pc_list.setMinimumHeight(80)
        self.pc_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pc_hint = QLabel("Discovered from the robot on connect.")
        self.pc_hint.setWordWrap(True)
        pc_layout.addWidget(self.pc_select_all)
        pc_layout.addWidget(self.pc_list)
        pc_layout.addWidget(self.pc_hint)
        layout.addWidget(pc_box, stretch=2)

        return panel

    def _build_controls_panel(self) -> QWidget:
        """Right column: capture rates, session info, start/stop controls."""
        panel = QWidget()
        panel.setMinimumWidth(260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        rate_box = QGroupBox("Capture rates")
        rate_form = QFormLayout(rate_box)
        rate_form.setSpacing(8)
        rate_form.setLabelAlignment(Qt.AlignRight)
        self.image_hz_spin = self._make_hz_spin(self.config.image_sample_rate)
        self.lidar_hz_spin = self._make_hz_spin(self.config.lidar_sample_rate)
        rate_form.addRow("Image rate (Hz):", self.image_hz_spin)
        rate_form.addRow("LiDAR rate (Hz):", self.lidar_hz_spin)
        self.record_state_check = QCheckBox("Record IMU / odometry log")
        self.record_state_check.setChecked(True)
        self.record_state_check.setToolTip(
            "Continuously log IMU / odometry to imu/state_log.jsonl for time "
            "synchronisation and LiDAR deskewing."
        )
        rate_form.addRow(self.record_state_check)
        layout.addWidget(rate_box)

        session_box = QGroupBox("Capture session")
        session_layout = QVBoxLayout(session_box)
        session_layout.setSpacing(6)
        self.session_label = QLabel("No active session (created on Start).")
        self.session_label.setWordWrap(True)
        self.new_session_btn = QPushButton("Start new session")
        self.new_session_btn.clicked.connect(self._on_new_session)
        session_layout.addWidget(self.session_label)
        session_layout.addWidget(self.new_session_btn)
        layout.addWidget(session_box)

        capture_box = QGroupBox("Recording")
        capture_layout = QVBoxLayout(capture_box)
        capture_layout.setSpacing(8)
        self.start_btn = QPushButton("Start capture")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setMinimumHeight(38)
        self.start_btn.clicked.connect(self._on_start_capture)
        self.stop_btn = QPushButton("Stop capture")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.clicked.connect(self._on_stop_capture)
        self.progress_label = QLabel("Images: 0  |  Scans: 0  |  Elapsed: 0.0 s")
        self.progress_label.setWordWrap(True)
        capture_layout.addWidget(self.start_btn)
        capture_layout.addWidget(self.stop_btn)
        capture_layout.addWidget(self.progress_label)
        layout.addWidget(capture_box)

        layout.addStretch()
        return panel

    @staticmethod
    def _make_hz_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 60.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setSuffix(" Hz")
        spin.setValue(value)
        spin.setToolTip("Set to 0 to disable this stream.")
        return spin

    # -- enable/disable logic ---------------------------------------------

    def _refresh_enabled_state(self) -> None:
        idle = self.connected and not self.busy and not self.recording

        self.connect_btn.setEnabled(
            not self.connected and not self.busy and not self.recording
        )
        self.disconnect_btn.setEnabled(
            self.connected and not self.busy and not self.recording
        )
        for edit in (self.hostname_edit, self.username_edit, self.password_edit):
            edit.setEnabled(not self.connected and not self.busy)

        self.new_session_btn.setEnabled(idle)
        self.start_btn.setEnabled(idle)
        self.stop_btn.setEnabled(self.recording)

        # Source selection and rates are locked while recording.
        editable = not self.recording
        self.image_list.setEnabled(editable)
        self.pc_list.setEnabled(editable)
        self.image_select_all.setEnabled(editable)
        self.image_color_only.setEnabled(editable)
        self.pc_select_all.setEnabled(editable)
        self.image_hz_spin.setEnabled(editable)
        self.lidar_hz_spin.setEnabled(editable)
        self.record_state_check.setEnabled(editable)

    # -- connection actions ------------------------------------------------

    def _on_connect(self) -> None:
        hostname = self.hostname_edit.text().strip()
        if not hostname:
            self._log("Please enter a hostname / IP.")
            return
        self._set_busy(True, "Connecting...")
        worker = FunctionWorker(
            self.controller.connect_and_discover,
            hostname,
            self.username_edit.text(),
            self.password_edit.text(),
        )
        worker.signals.log.connect(self._log)
        worker.signals.finished.connect(self._on_connected)
        worker.signals.error.connect(self._on_connect_error)
        self.pool.start(worker)

    def _on_connected(self, result: ConnectResult) -> None:
        self.connected = result.status.connected
        self.has_point_cloud = result.status.has_point_cloud
        self._populate_sources(result)
        self._set_status_indicator(True, result.status.message or "Connected")
        self._update_battery(result.status.battery_percent)
        self._set_busy(False)

    def _on_connect_error(self, message: str) -> None:
        self.connected = False
        self.has_point_cloud = False
        self._set_status_indicator(False, "Connection failed")
        self._log(f"ERROR: {message}")
        self._set_busy(False)

    def _on_disconnect(self) -> None:
        self.controller.disconnect(log=self._log)
        self.connected = False
        self.has_point_cloud = False
        self._prefill_image_sources()
        self.pc_list.clear()
        self.pc_select_all.setChecked(False)
        self.pc_hint.setText("Discovered from the robot on connect.")
        self._set_status_indicator(False, "Disconnected")
        self._update_battery(None)
        self._refresh_enabled_state()

    # -- session / recording actions --------------------------------------

    def _on_new_session(self) -> None:
        session = self.controller.start_new_session()
        self.session_label.setText(f"Active: {session.name}")
        self._log(f"Started session {session.name}")

    def _on_start_capture(self) -> None:
        image_sources = self._checked_items(self.image_list)
        pc_sources = self._checked_items(self.pc_list)
        if not image_sources and not pc_sources:
            self._log("Select at least one image or point-cloud source first.")
            return

        self.recording = True
        self.progress_label.setText("Images: 0  |  Scans: 0  |  Elapsed: 0.0 s")
        self._set_status_indicator(True, "Recording...")
        self._refresh_enabled_state()

        worker = ContinuousCaptureWorker(
            self.controller,
            image_sources,
            pc_sources,
            self.image_hz_spin.value(),
            self.lidar_hz_spin.value(),
        )
        worker.signals.started.connect(self._on_recording_started)
        worker.signals.tick.connect(self._on_capture_tick)
        worker.signals.progress.connect(self._on_capture_progress)
        worker.signals.finished.connect(self._on_recording_finished)
        worker.signals.error.connect(self._on_capture_error)
        worker.signals.log.connect(self._log)
        self._capture_worker = worker
        self.pool.start(worker)

    def _on_stop_capture(self) -> None:
        self._log("Stopping capture...")
        if self._capture_worker is not None:
            self._capture_worker.stop()
        if self._state_worker is not None:
            self._state_worker.stop()
        self.stop_btn.setEnabled(False)

    def _on_recording_started(self, session_name: str) -> None:
        session = self.controller.active_session
        if session is not None:
            self.session_label.setText(f"Active: {session.name}")
        self._log(f"Recording into {session_name}")
        self._start_state_recorder(session)

    def _start_state_recorder(self, session) -> None:
        """Begin the concurrent IMU / odometry log for this recording."""
        if session is None or not self.record_state_check.isChecked():
            return
        worker = StateRecorderWorker(
            self.controller.spot,
            self.controller.state_log_path(session),
            self.config.state_sample_rate,
        )
        worker.signals.finished.connect(self._on_state_finished)
        worker.signals.error.connect(self._on_capture_error)
        worker.signals.log.connect(self._log)
        self._state_worker = worker
        self.pool.start(worker)

    def _on_state_finished(self, count: int) -> None:
        self._state_worker = None
        self._log(f"IMU / odometry log written: {count} sample(s).")

    def _on_capture_tick(self, batch: CaptureBatchResult) -> None:
        pass  # progress counter is updated via the separate _on_capture_progress signal

    def _on_capture_progress(self, images: int, scans: int, elapsed: float) -> None:
        self.progress_label.setText(
            f"Images: {images}  |  Scans: {scans}  |  Elapsed: {elapsed:.1f} s"
        )

    def _on_recording_finished(self, images: int, scans: int, elapsed: float) -> None:
        self.recording = False
        self._capture_worker = None
        # If the capture loop ended on its own, make sure the IMU log stops too.
        if self._state_worker is not None:
            self._state_worker.stop()
        self.progress_label.setText(
            f"Images: {images}  |  Scans: {scans}  |  Elapsed: {elapsed:.1f} s"
        )
        self._set_status_indicator(self.connected, "Connected")
        self._refresh_enabled_state()
        self._log(
            f"Capture stopped. {images} image(s), {scans} scan(s) "
            f"over {elapsed:.1f}s."
        )

    def _on_capture_error(self, message: str) -> None:
        self._log(f"ERROR: {message}")

    # -- view helpers ------------------------------------------------------

    def _prefill_image_sources(self) -> None:
        """Show expected cameras from config before a robot is connected."""
        self.image_list.clear()
        for name in self.config.available_cameras:
            self._add_checkable(self.image_list, name, name, checked=True)
        self.image_select_all.setChecked(True)

    def _populate_sources(self, result: ConnectResult) -> None:
        # The robot's reported sources are authoritative -- replace the config
        # pre-fill with what is actually available.
        self.image_list.clear()
        for src in result.image_sources:
            label = f"{src.name}  ({src.cols}x{src.rows}, {src.image_type})"
            self._add_checkable(self.image_list, src.name, label, checked=True)
        self.image_select_all.setChecked(True)

        self.pc_list.clear()
        for src in result.point_cloud_sources:
            self._add_checkable(self.pc_list, src.name, src.name, checked=True)

        if result.point_cloud_sources:
            self.pc_hint.setText("")
            self.pc_select_all.setChecked(True)
        elif result.status.has_point_cloud:
            self.pc_hint.setText("Service present but no sources advertised.")
        else:
            self.pc_hint.setText(
                "No point-cloud / LiDAR payload detected on this robot."
            )

    def _update_battery(self, percent: Optional[float]) -> None:
        if percent is None:
            self.battery_label.setText("")
        else:
            self.battery_label.setText(f"Battery: {percent:.0f}%")

    def _set_status_indicator(self, connected: bool, text: str) -> None:
        color = "#a6e3a1" if connected else "#f38ba8"
        self.status_dot.setStyleSheet(
            f"background:{color}; border-radius:7px;"
        )
        self.status_text.setText(text)

    def _set_busy(self, busy: bool, text: Optional[str] = None) -> None:
        self.busy = busy
        if busy and text:
            self._set_status_indicator(self.connected, text)
        elif not busy:
            self._set_status_indicator(
                self.connected, "Connected" if self.connected else "Disconnected"
            )
        self._refresh_enabled_state()

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    # -- list utilities ----------------------------------------------------

    def _on_image_color_only_toggled(self, checked: bool) -> None:
        """Check only color (non-depth) sources; uncheck when toggled off."""
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            source_name = item.data(Qt.UserRole)
            is_color = "depth" not in source_name
            if checked:
                item.setCheckState(Qt.Checked if is_color else Qt.Unchecked)
            else:
                if is_color:
                    item.setCheckState(Qt.Unchecked)

    @staticmethod
    def _add_checkable(
        list_widget: QListWidget, value: str, label: str, checked: bool
    ) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, value)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        list_widget.addItem(item)

    @staticmethod
    def _checked_items(list_widget: QListWidget) -> List[str]:
        values = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.Checked:
                values.append(item.data(Qt.UserRole))
        return values

    @staticmethod
    def _set_all_checked(list_widget: QListWidget, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(state)
