
"""Save raw ADC frames received by RadarStream."""

import json
import time
import csv
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread

import numpy as np


class RawRecorder:

    def __init__(self, output_dir="raw_recordings"):

        # Create the folder where recordings will be saved
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Queue holds ADC frames until they are written
        self.queue = Queue(maxsize=512)

        # Thread synchronization
        self.lock = Lock()
        self.stop_event = Event()
        self.thread = None

        # Recording status
        self.recording = False
        self.frames_queued = 0
        self.frames_written = 0
        self.frames_dropped = 0
        self.invalid_frames = 0
        self.error = None

        # File information
        self.output_path = None
        self.metadata_path = None
        self.frame_values = 0
        self.metadata = {}
        self.deadline = 0.0

    @property
    def busy(self):
        """Check whether the writer thread is still running."""
        return self.thread is not None and self.thread.is_alive()

    def start(self, radar_config, seconds=40, metadata_extra=None, diagnostics_provider=None):
        """Start a new raw ADC recording."""

        with self.lock:

            if self.recording or self.busy:
                raise RuntimeError("Recording is already active.")

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            # Create unique filenames
            self.output_path = (
                self.output_dir / f"radar_{timestamp}.bin"
            )

            self.metadata_path = (
                self.output_dir / f"radar_{timestamp}.json"
            )

            # Calculate expected raw samples per frame
            self.frame_values = radar_config.raw_values_per_frame

            # Reset counters
            self.frames_queued = 0
            self.frames_written = 0
            self.frames_dropped = 0
            self.invalid_frames = 0
            self.error = None

            self.queue = Queue(maxsize=512)
            self.stop_event.clear()

            # Set recording duration
            self.start_monotonic_ns=time.monotonic_ns()
            self.diagnostics_provider=diagnostics_provider
            self.diagnostics_start=diagnostics_provider() if diagnostics_provider else {}
            self.deadline = time.monotonic() + seconds
            self.recording = True

            # Save radar configuration
            self.metadata = {
                "start_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "requested_duration_seconds": seconds,
                "adc_samples": radar_config.adc_samples,
                "chirps_per_frame": radar_config.chirps_per_frame,
                "tx_antennas": radar_config.tx_antennas,
                "rx_antennas": radar_config.rx_antennas,
                "int16_values_per_frame": self.frame_values,
                "bytes_per_frame": self.frame_values * 2,
                "dtype": "<i2",
                "source": "RadarStream CaptureBuffer",
            }

            self.metadata.update(metadata_extra or {})
            self.metadata["frame_timestamps_file"]=self.output_path.with_suffix(".frames.csv").name
            # Start writing data in a separate thread
            self.thread = Thread(
                target=self._writer,
                name="RawADCWriter",
                daemon=False,
            )

            self.thread.start()

        return self.output_path

    def submit(self, raw_frame, observed_sequence=None, observed_monotonic_ns=None, source_sequence=None):
        """Receive one raw ADC frame from data_pipeline.py."""

        if not self.recording:
            return

        # Stop accepting frames after the time limit
        if time.monotonic() >= self.deadline:
            self.stop()
            return

        samples = np.asarray(raw_frame)

        with self.lock:

            if not self.recording:
                return

            # Verify the frame has the expected dimensions
            if (
                samples.dtype != np.int16
                or samples.size != self.frame_values
            ):
                self.invalid_frames += 1
                return

            # Convert the raw samples to binary bytes
            payload = samples.astype(
                "<i2", copy=False
            ).tobytes()

            try:
                timestamp=observed_monotonic_ns if observed_monotonic_ns is not None else time.monotonic_ns()
                self.queue.put_nowait((payload,dict(observed_sequence=observed_sequence,
                    source_sequence=source_sequence,observed_monotonic_ns=timestamp,
                    relative_observation_time_s=(timestamp-self.start_monotonic_ns)/1e9)))
                self.frames_queued += 1

            except Full:
                self.frames_dropped += 1

    def stop(self):
        """Stop recording and finish saving queued frames."""

        with self.lock:
            self.recording = False
            self.stop_event.set()

    def wait(self, timeout=5):
        """Wait for the writer thread to finish."""

        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def _writer(self):
        """Write incoming ADC frames to a .bin file."""

        try:
            with self.output_path.open("wb") as output_file, self.output_path.with_suffix('.frames.csv').open('w',newline='',encoding='utf-8') as times_file:
                fields=['written_frame','observed_sequence','source_sequence','observed_monotonic_ns','relative_observation_time_s']
                times_writer=csv.DictWriter(times_file,fieldnames=fields);times_writer.writeheader()

                while (
                    not self.stop_event.is_set()
                    or not self.queue.empty()
                ):

                    try:
                        payload,frame_info = self.queue.get(timeout=0.1)

                    except Empty:
                        continue

                    output_file.write(payload)
                    times_writer.writerow(dict(written_frame=self.frames_written,**frame_info))
                    self.frames_written += 1

        except Exception as exc:
            self.error = str(exc)

        finally:

            with self.lock:
                self.recording = False
                self.stop_event.set()

            # Save information about the recording
            result = dict(self.metadata)
            if self.diagnostics_provider:
                diagnostics=self.diagnostics_provider()
                delta={key:diagnostics[key]-self.diagnostics_start.get(key,0) for key in
                    ('observed_frames','processed_frames','dsp_queue_dropped_frames','unstable_copies','incomplete_frames','processing_errors','source_sequence_gaps','source_duplicate_frames','source_reordered_frames')}
                result['capture_diagnostics']=dict(final_snapshot=diagnostics,interval_counter_deltas=delta,
                    note='Observation counters, not hardware frame counts. DSP may still finish queued frames after recording stops.')
            result['timestamp_definition']='Host observation time, not radar acquisition timestamp'
            result['udp_packet_gaps']=None
            result['native_overwritten_frames']=None

            result.update({
                "end_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "frames_queued": self.frames_queued,
                "frames_written": self.frames_written,
                "queue_dropped_frames": self.frames_dropped,
                "invalid_frames": self.invalid_frames,
                "file_bytes": (
                    self.output_path.stat().st_size
                    if self.output_path.exists()
                    else 0
                ),
                "error": self.error,
            })

            try:
                with self.metadata_path.open(
                    "w", encoding="utf-8"
                ) as metadata_file:

                    json.dump(
                        result,
                        metadata_file,
                        indent=2,
                    )

            except Exception as exc:
                if self.error is None:
                    self.error = f"Metadata error: {exc}"
