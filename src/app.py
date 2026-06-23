import logging
import sys

from PySide6.QtWidgets import QApplication

from src.config import load_config
from src.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("Starting 3D Environment Reconstruction from Robot Sensor Data")

    try:
        config = load_config()
        logger.info(
            f"Configuration loaded: hostname={config.robot_hostname}, "
            f"output={config.output_dir}"
        )

        app = QApplication(sys.argv)
        app.setStyle("Fusion")

        window = MainWindow(config)
        window.show()

        sys.exit(app.exec())

    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)
