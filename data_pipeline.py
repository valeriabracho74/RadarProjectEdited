"""Threads and buffer adapters for real-time radar capture."""

import time
from ctypes import POINTER, c_int, c_short, cast, cdll
from dataclasses import dataclass, field
from multiprocessing import Process, RawArray
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread, Lock

import numpy as np

from app_config import DEFAULT_CONFIG, RadarFrameConfig
from radar_dsp.utils import Window
from runtime_state import FeatureMode, ProcessingMode


@dataclass(frozen=True)
class FeatureFrame:
    """All feature views derived from the same ADC frame."""

    sequence: int
    rti: np.ndarray
    dti: np.ndarray
    rdi: np.ndarray
    rai: np.ndarray
    rei: np.ndarray
    points: tuple = ()
    diagnostics: dict = field(default_factory=dict)
    physical_maps: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MicroDopplerFrame:
    """Long single-RX Doppler-time history for the micro-Doppler view."""

    sequence: int
    micro_doppler: np.ndarray


class CaptureBuffer:
    """Own the double buffer shared with the native UDP capture library."""

    def __init__(self, frame_length, library_path):
        self.frame_length = frame_length
        self.library_path = str(Path(library_path).resolve())
        if not Path(self.library_path).is_file():
            raise FileNotFoundError(
                "Native capture library not found: {}".format(self.library_path)
            )
        # RawArray keeps the two native buffers visible to the parent while
        # captureudp runs in a child process. A process can be terminated and
        # restarted when a selected radar cfg changes the frame length.
        self._active_half = RawArray(c_int, 1)
        self._samples = RawArray(c_short, frame_length * 2)

    @property
    def active_half(self):
        return self._active_half[0]

    def capture(self):
        library = cdll.LoadLibrary(self.library_path)
        library.captureudp(
            cast(self._active_half, POINTER(c_int)),
            cast(self._samples, POINTER(c_short)),
            self.frame_length,
        )

    def copy_latest_frame(self, active_half=None):
        if active_half is None:
            active_half = self.active_half
        start = self.frame_length * (1 - active_half)
        end = start + self.frame_length
        samples = np.frombuffer(self._samples, dtype=np.int16)
        return samples[start:end].copy()


class UdpListener(Process):
    """Run blocking native UDP capture in a restartable child process."""

    def __init__(self, name, capture_buffer):
        super().__init__(name=name, daemon=True)
        self.capture_buffer = capture_buffer

    def run(self):
        self.capture_buffer.capture()

    def stop(self):
        if self.is_alive():
            self.terminate()


class DataProcessor(Thread):
    """Decode raw frames and publish synchronized DSP feature frames."""

    def __init__(
        self,
        name,
        capture_buffer,
        signal_processor,
        output_queue,
        radar_config=DEFAULT_CONFIG.radar,
        raw_recorder=None,
        
    ):
        super().__init__(name=name, daemon=True)
        if not isinstance(radar_config, RadarFrameConfig):
            raise TypeError("radar_config must be a RadarFrameConfig")
        self.capture_buffer = capture_buffer
        self.signal_processor = signal_processor
        self.output_queue = output_queue
        self.radar_config = radar_config
        self._stop_event = Event()
        self.raw_recorder = raw_recorder
        settings=getattr(signal_processor,'settings',None)
        self.dsp_queue=Queue(maxsize=getattr(settings,'dsp_queue_frames',8))
        self._stats_lock=Lock()
        self._stats=dict(observed_frames=0,processed_frames=0,processing_attempts=0,dsp_queue_dropped_frames=0,
            unstable_copies=0,incomplete_frames=0,processing_errors=0,
            source_sequence_gaps=0,source_duplicate_frames=0,source_reordered_frames=0,source_sequence_available=False,last_source_sequence=None,processing_total_ms=0.,processing_max_ms=0.,
            capture_notification='alternating_half_flag',upstream_lost_frames=None,udp_packet_gaps=None,
            native_overwritten_frames=None,native_incomplete_frames=None)
        self._dsp_worker=None

    def snapshot_diagnostics(self):
        with self._stats_lock:
            result=dict(self._stats)
        result['processing_mean_ms']=result['processing_total_ms']/max(result['processing_attempts'],1)
        result['dsp_queue_depth']=self.dsp_queue.qsize()
        return result

    def _count(self,**values):
        with self._stats_lock:
            for key,value in values.items(): self._stats[key]+=value

    def stop(self):
        self._stop_event.set()

    def _accept_frame(self,raw_frame,timestamp_ns,source_sequence=None):
        raw_frame=np.asarray(raw_frame)
        if raw_frame.dtype!=np.int16 or raw_frame.size!=self.radar_config.raw_values_per_frame:
            self._count(incomplete_frames=1);return
        raw_frame=raw_frame.copy()
        self._count(observed_frames=1)
        observed=self.snapshot_diagnostics()['observed_frames']
        if source_sequence is not None:
            with self._stats_lock:
                self._stats['source_sequence_available']=True
                last=self._stats['last_source_sequence']
                if last is not None and source_sequence==last:self._stats['source_duplicate_frames']+=1
                if last is not None and source_sequence<last:self._stats['source_reordered_frames']+=1
                if last is not None and source_sequence>last+1:
                    self._stats['source_sequence_gaps']+=source_sequence-last-1
                self._stats['last_source_sequence']=source_sequence
        # Recording never waits for DSP. Payload is copied/queued by recorder.
        if self.raw_recorder is not None:
            self.raw_recorder.submit(raw_frame,observed_sequence=observed,
                observed_monotonic_ns=timestamp_ns,source_sequence=source_sequence)
        packet=(observed,timestamp_ns,raw_frame)
        try: self.dsp_queue.put_nowait(packet)
        except Full:
            try: self.dsp_queue.get_nowait();self._count(dsp_queue_dropped_frames=1)
            except Empty: pass
            try: self.dsp_queue.put_nowait(packet)
            except Full: self._count(dsp_queue_dropped_frames=1)

    def _consume_dsp(self):
        while not self._stop_event.is_set() or not self.dsp_queue.empty():
            try: sequence,timestamp,raw=self.dsp_queue.get(timeout=.05)
            except Empty: continue
            started=time.perf_counter()
            try:
                frame=self._process_frame(self._decode_frame(raw),sequence)
                self._count(processed_frames=1)
                if frame is not None:
                    if hasattr(frame,'diagnostics'):
                        frame.diagnostics.update(observed_monotonic_ns=timestamp,
                            queue_latency_ms=(time.monotonic_ns()-timestamp)/1e6)
                    if hasattr(frame,'points'):
                        for point in frame.points:
                            point.update(sequence=sequence,timestamp_s=timestamp/1e9,timestamp_source='host_observation_monotonic')
                    self._publish_latest(frame)
            except Exception as error:
                self._count(processing_errors=1)
                with self._stats_lock:self._stats['last_processing_error']=repr(error)
            finally:
                elapsed=(time.perf_counter()-started)*1000
                self._count(processing_attempts=1,processing_total_ms=elapsed)
                with self._stats_lock:self._stats['processing_max_ms']=max(self._stats['processing_max_ms'],elapsed)

    def run(self):
        self._dsp_worker=Thread(target=self._consume_dsp,name=self.name+'-DSP',daemon=True)
        self._dsp_worker.start()
        try:
            if hasattr(self.capture_buffer,'get_next_frame'):
                # Optional sequence-aware transport adapter (e.g. CPSL).
                # The bundled DLL does NOT expose this interface.
                with self._stats_lock:self._stats['capture_notification']='sequence_aware_adapter'
                while not self._stop_event.is_set():
                    packet=self.capture_buffer.get_next_frame(timeout=.05)
                    if packet is not None:
                        raw,source_sequence,timestamp_ns=packet
                        self._accept_frame(raw,timestamp_ns,source_sequence)
            else:
                last_half=self.capture_buffer.active_half
                while not self._stop_event.is_set():
                    half=self.capture_buffer.active_half
                    if half==last_half:
                        self._stop_event.wait(.0005);continue
                    raw=self.capture_buffer.copy_latest_frame(half)
                    # Detect only an observed switch during copy. Two or more
                    # unobserved transitions cannot be inferred from one bit.
                    last_half=half
                    if self.capture_buffer.active_half!=half:
                        self._count(unstable_copies=1);continue
                    self._accept_frame(raw,time.monotonic_ns())
        finally:
            self._stop_event.set()
            self._dsp_worker.join()

    def _process_frame(self, adc_frame, sequence):
        """Run only the DSP path selected by the display-mode menu."""

        policy=self.signal_processor.runtime_state.processing_mode
        if getattr(self,'_last_policy',None) is not policy:
            for name in ('rti_history','rdi_history','rai_history','rei_history'):
                if hasattr(self.signal_processor,name):getattr(self.signal_processor,name).clear()
            if hasattr(self.signal_processor,'reset_micro_doppler_history'):self.signal_processor.reset_micro_doppler_history()
            self._last_policy=policy
        suppress=policy is ProcessingMode.MOTION
        mode = self.signal_processor.runtime_state.feature_mode
        if mode is FeatureMode.MICRO_DOPPLER:
            micro_doppler = self.signal_processor.process_micro_doppler(
                adc_frame, window_type_1d=Window.HANNING
            )
            if micro_doppler.size == 0:
                return None
            if self.signal_processor.runtime_state.feature_mode is mode and self.signal_processor.runtime_state.processing_mode is policy:
                return MicroDopplerFrame(sequence, micro_doppler)
            return None

        rti, rdi, dti = self.signal_processor.process_time_features(
            adc_frame, window_type_1d=Window.HANNING,clutter_removal_enabled=suppress
        )
        if self.signal_processor.runtime_state.feature_mode is not mode:
            return None
        rai, rei = self.signal_processor.process_angle_features(adc_frame,clutter_removal_enabled=suppress)
        if self.signal_processor.runtime_state.feature_mode is mode and self.signal_processor.runtime_state.processing_mode is policy:
            mapping=getattr(self.signal_processor,'latest_mapping',{})
            return FeatureFrame(sequence,rti,dti,rdi,rai,rei,
                tuple(mapping.get('points',[])),dict(mapping.get('diagnostics',{})),
                {key:mapping[key] for key in ('ranges','fd','rd','ra','re','az','el') if key in mapping})
        return None

    def _decode_frame(self, raw_frame):
        expected = self.radar_config.raw_values_per_frame
        if raw_frame.size != expected:
            raise ValueError(
                "Expected {} raw values, received {}".format(expected, raw_frame.size)
            )

        # DCA1000 complex mode packs [I0, I1, Q0, Q1] repeatedly.
        packed = raw_frame.reshape(-1, 4)
        complex_samples = packed[:, 0:2] + 1j * packed[:, 2:4]
        adc_data = complex_samples.reshape(
            self.radar_config.chirps_per_frame,
            self.radar_config.rx_antennas,
            self.radar_config.adc_samples,
        )
        adc_data = adc_data.transpose((0, 2, 1))

        profile=getattr(getattr(self.signal_processor,'config',None),'physical_profile',None)
        order=profile.tx_order if profile is not None else tuple(range(self.radar_config.tx_antennas))
        tx_channels = [
            adc_data[order.index(tx_index)::self.radar_config.tx_antennas]
            for tx_index in range(self.radar_config.tx_antennas)
        ]
        return np.concatenate(tx_channels, axis=2)

    def _publish_latest(self, feature_frame):
        try:
            self.output_queue.put_nowait(feature_frame)
            return
        except Full:
            pass

        try:
            self.output_queue.get_nowait()
        except Empty:
            pass
        self.output_queue.put_nowait(feature_frame)
