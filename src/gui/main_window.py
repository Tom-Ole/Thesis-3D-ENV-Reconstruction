from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.app_context import AppContext
from src.config import Config
from src.gui.tabs.capture_tab import CaptureTab

_DARK_STYLE = """
QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: #cdd6f4;
    background-color: #1e1e2e;
}
QMainWindow {
    background-color: #1e1e2e;
}
QGroupBox {
    font-weight: 600;
    font-size: 12px;
    border: 1px solid #313244;
    border-radius: 7px;
    margin-top: 10px;
    padding-top: 10px;
    color: #89b4fa;
    background-color: #1e1e2e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 7px 18px;
    min-height: 30px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #6c7086;
}
QPushButton:pressed {
    background-color: #1e1e2e;
}
QPushButton:disabled {
    background-color: #232333;
    color: #45475a;
    border-color: #313244;
}
QPushButton#startBtn {
    background-color: #1c3a28;
    color: #a6e3a1;
    border-color: #40a02b;
    font-weight: 600;
}
QPushButton#startBtn:hover {
    background-color: #23492f;
    border-color: #50c03b;
}
QPushButton#startBtn:disabled {
    background-color: #141e18;
    color: #2d4a32;
    border-color: #1e3020;
}
QPushButton#stopBtn {
    background-color: #3a1c1c;
    color: #f38ba8;
    border-color: #e64553;
    font-weight: 600;
}
QPushButton#stopBtn:hover {
    background-color: #4a2323;
    border-color: #f0506a;
}
QPushButton#stopBtn:disabled {
    background-color: #1e1418;
    color: #4a2535;
    border-color: #2e1a20;
}
QLineEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 6px 10px;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
QLineEdit:focus {
    border-color: #89b4fa;
}
QLineEdit:disabled {
    background-color: #252535;
    color: #45475a;
}
QListWidget {
    background-color: #252535;
    border: 1px solid #313244;
    border-radius: 5px;
    color: #cdd6f4;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 5px 8px;
    border-radius: 4px;
    min-height: 22px;
}
QListWidget::item:hover {
    background-color: #313244;
}
QListWidget:disabled {
    color: #45475a;
    background-color: #1e1e2e;
}
QPlainTextEdit {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 5px;
    color: #a6adc8;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 4px;
}
QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 5px 8px;
    color: #cdd6f4;
    min-height: 26px;
}
QDoubleSpinBox:focus {
    border-color: #89b4fa;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #45475a;
    border: none;
    border-radius: 3px;
    width: 18px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #6c7086;
}
QCheckBox {
    color: #cdd6f4;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 17px;
    height: 17px;
    border: 1.5px solid #6c7086;
    border-radius: 4px;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}
QCheckBox::indicator:disabled {
    border-color: #45475a;
    background-color: #252535;
}
QTabWidget::pane {
    border: none;
    background-color: #1e1e2e;
    border-top: 1px solid #313244;
}
QTabBar::tab {
    background-color: #181825;
    color: #6c7086;
    padding: 9px 24px;
    border: none;
    font-size: 13px;
}
QTabBar::tab:selected {
    color: #cdd6f4;
    background-color: #1e1e2e;
    border-bottom: 2px solid #89b4fa;
}
QTabBar::tab:hover:!selected {
    background-color: #252535;
    color: #a6adc8;
}
QSplitter::handle {
    background-color: #313244;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:hover {
    background-color: #89b4fa;
}
QStatusBar {
    background-color: #181825;
    color: #6c7086;
    font-size: 11px;
    border-top: 1px solid #313244;
}
QLabel {
    color: #cdd6f4;
    background-color: transparent;
}
QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 10px;
    border-radius: 5px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #6c7086;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background-color: #1e1e2e;
    height: 10px;
    border-radius: 5px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background-color: #45475a;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #6c7086;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""


class MainWindow(QMainWindow):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # Shared, GUI-agnostic state consumed by all tabs.
        self.context = AppContext.create(config)
        self.setup_ui()

    def setup_ui(self) -> None:
        self.setWindowTitle("SPOT 3D Capture & Reconstruction")
        self.setGeometry(100, 100, 1280, 800)
        self.setMinimumSize(900, 620)
        self.setStyleSheet(_DARK_STYLE)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Capture tab (implemented). Reconstruction tabs are future work and are
        # added as placeholders so the pipeline structure stays visible.
        self.capture_tab = CaptureTab(self.context)
        self.tabs.addTab(self.capture_tab, "Capture")
        self.tabs.addTab(
            self._placeholder("COLMAP reconstruction — coming soon"),
            "COLMAP Reconstruct",
        )
        self.tabs.addTab(
            self._placeholder("LiDAR reconstruction — coming soon"),
            "LiDAR Reconstruct",
        )
        self.tabs.addTab(
            self._placeholder("AI reconstruction — coming soon"),
            "AI Reconstruct",
        )

        main_layout.addWidget(self.tabs)
        self.setCentralWidget(main_widget)
        self.setStatusBar(QStatusBar())

    @staticmethod
    def _placeholder(text: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color:#6c7086; font-size:16px; background:transparent;")
        layout.addWidget(label)
        return widget

    def closeEvent(self, event) -> None:
        self.context.spot.disconnect()
        super().closeEvent(event)
