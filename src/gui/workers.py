"""Background-task helpers for the Qt GUI.

Network operations (connect, capture) must not run on the UI thread. These wrap
work in ``QRunnable`` objects and report progress/result/errors back to the GUI
via thread-safe signals. Callables are given a ``log`` keyword whose value is a
signal emitter, so services can stream status lines to the UI.
"""
from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from src.data_models import CAPTURE_TYPE_IMAGE

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    finished = Signal(object)  # the callable's return value
    error = Signal(str)
    log = Signal(str)


class FunctionWorker(QRunnable):
    """Runs ``fn(*args, log=..., **kwargs)`` on a thread-pool thread."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, log=self.signals.log.emit, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - report to UI, never crash thread
            logger.exception("Background worker failed")
            self.signals.error.emit(str(exc))


class CaptureWorkerSignals(QObject):
    started = Signal(str)  # active session name
    tick = Signal(object)  # CaptureBatchResult for one capture tick
    progress = Signal(int, int, float)  # images, scans, elapsed seconds
    finished = Signal(int, int, float)  # images, scans, elapsed seconds
    error = Signal(str)
    log = Signal(str)


class ContinuousCaptureWorker(QRunnable):
    """Records images and point clouds continuously at independent rates.

    Drives :meth:`CaptureController.capture` on a timer for each modality until
    :meth:`stop` is called. Image and LiDAR have separate target rates; when both
    fall due on the same tick they are captured together under one event id.

    Next-capture times are scheduled from the *completion* of each tick so a slow
    RPC or disk write never triggers a catch-up burst -- this records as close to
    the target rate as the hardware allows and is honest about it.
    """

    def __init__(
        self,
        controller,
        image_sources,
        pc_sources,
        image_hz: float,
        lidar_hz: float,
    ):
        super().__init__()
        self.controller = controller
        self.image_sources = list(image_sources)
        self.pc_sources = list(pc_sources)
        self.image_hz = image_hz
        self.lidar_hz = lidar_hz
        self.signals = CaptureWorkerSignals()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @Slot()
    def run(self) -> None:
        try:
            session = self.controller.prepare_recording(log=self.signals.log.emit)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
            self.signals.finished.emit(0, 0, 0.0)
            return
        self.signals.started.emit(session.name)

        img_interval = (
            1.0 / self.image_hz
            if self.image_hz > 0 and self.image_sources
            else None
        )
        lidar_interval = (
            1.0 / self.lidar_hz
            if self.lidar_hz > 0 and self.pc_sources
            else None
        )
        if img_interval is None and lidar_interval is None:
            self.signals.error.emit(
                "Nothing to capture: select at least one source with a rate > 0."
            )
            self.signals.finished.emit(0, 0, 0.0)
            return

        start = time.monotonic()
        next_img = start if img_interval is not None else None
        next_lidar = start if lidar_interval is not None else None
        images = scans = 0

        while not self._stop.is_set():
            now = time.monotonic()
            do_img = next_img is not None and now >= next_img
            do_pc = next_lidar is not None and now >= next_lidar

            if not (do_img or do_pc):
                upcoming = [t for t in (next_img, next_lidar) if t is not None]
                wait = (min(upcoming) - now) if upcoming else 0.05
                time.sleep(max(0.002, min(wait, 0.05)))
                continue

            batch = self.controller.capture(
                self.image_sources if do_img else [],
                self.pc_sources if do_pc else [],
                log=self.signals.log.emit,
            )
            for res in batch.results:
                if res.capture_type == CAPTURE_TYPE_IMAGE:
                    images += 1
                else:
                    scans += 1
            for err in batch.errors:
                self.signals.log.emit(f"ERROR: {err}")

            self.signals.tick.emit(batch)
            self.signals.progress.emit(images, scans, now - start)

            done = time.monotonic()
            if do_img and img_interval is not None:
                next_img = done + img_interval
            if do_pc and lidar_interval is not None:
                next_lidar = done + lidar_interval

        self.signals.finished.emit(images, scans, time.monotonic() - start)


class StateRecorderSignals(QObject):
    finished = Signal(int)  # number of samples written
    error = Signal(str)
    log = Signal(str)


class StateRecorderWorker(QRunnable):
    """Records the IMU / odometry stream to a JSONL file alongside captures.

    Runs concurrently with :class:`ContinuousCaptureWorker` on its own thread so
    the high-rate state stream is not throttled by image/point-cloud RPCs. All
    robot I/O lives in :meth:`SpotService.record_state`; this worker just owns
    the thread and the stop flag.
    """

    def __init__(self, spot_service, jsonl_path, target_hz: float):
        super().__init__()
        self.spot = spot_service
        self.jsonl_path = jsonl_path
        self.target_hz = target_hz
        self.signals = StateRecorderSignals()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @Slot()
    def run(self) -> None:
        try:
            count = self.spot.record_state(
                self.jsonl_path,
                self._stop,
                self.target_hz,
                log=self.signals.log.emit,
            )
            self.signals.finished.emit(count)
        except Exception as exc:  # noqa: BLE001
            logger.exception("State recorder failed")
            self.signals.error.emit(str(exc))
            self.signals.finished.emit(0)
