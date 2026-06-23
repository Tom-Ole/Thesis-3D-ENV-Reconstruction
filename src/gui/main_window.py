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


class MainWindow(QMainWindow):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # Shared, GUI-agnostic state consumed by all tabs.
        self.context = AppContext.create(config)

        self.setup_ui()

    def setup_ui(self) -> None:
        self.setWindowTitle("SPOT 3D Capture & Reconstruction")
        self.setGeometry(100, 100, 1200, 700)

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
            self._placeholder("COLMAP reconstruction - coming soon"),
            "COLMAP Reconstruct",
        )
        self.tabs.addTab(
            self._placeholder("LiDAR reconstruction - coming soon"),
            "LiDAR Reconstruct",
        )
        self.tabs.addTab(
            self._placeholder("AI reconstruction - coming soon"),
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
        label.setStyleSheet("color:#888; font-size:16px;")
        layout.addWidget(label)
        return widget

    def closeEvent(self, event) -> None:
        # Tear down the robot connection cleanly on window close.
        self.context.spot.disconnect()
        super().closeEvent(event)
