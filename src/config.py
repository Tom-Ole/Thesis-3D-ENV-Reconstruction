
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


_AVAILABLE_CAMERA_SOURCES = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
]


@dataclass
class Config:

    # Robot auth
    robot_hostname: str
    robot_username: str
    robot_password: str

    # Output
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "captures")

    # Capture rates for continuous recording (editable in the GUI). Set 0 to
    # disable a stream. These are *target* rates -- effective rate is limited by
    # robot RPC latency and disk write speed.
    lidar_sample_rate: float = 10.0  # hz (point-cloud scans per second)
    image_sample_rate: float = 5.0  # hz (image captures per second, all cameras)

    # IMU / odometry recording rate (polling fallback). The high-rate
    # robot-state-streaming service is used instead when available.
    state_sample_rate: float = 50.0  # hz
    prefer_state_streaming: bool = True  # use 333Hz IMU stream when present

    # Rotate front fisheye images upright on save. Note: this changes pixel
    # coordinates so saved images no longer match the raw intrinsics stored in
    # metadata. Disable if you need unrotated images for calibration tools that
    # consume the intrinsics directly.
    apply_camera_rotation: bool = True
    available_cameras: list[str] = field(
        default_factory=lambda: list(_AVAILABLE_CAMERA_SOURCES)
    )


def load_config() -> Config:

    load_dotenv()

    hostname = os.getenv("BOSDYN_HOSTNAME", "192.168.10.3")
    username = os.getenv("BOSDYN_USERNAME", "student")
    password = os.getenv("BOSDYN_PASSWORD", "")

    output_dir = os.getenv("CAPTURE_OUTPUT_DIR")

    config = Config(
        robot_hostname=hostname,
        robot_username=username,
        robot_password=password,
    )
    if output_dir:
        config.output_dir = Path(output_dir)

    return config
