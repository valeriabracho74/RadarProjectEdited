import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from queue import Queue
import json
import time
import numpy as np
from app_config import AppConfig, DspConfig, RadarFrameConfig
from runtime_state import RuntimeState, ProcessingMode
from signal_processor import RadarSignalProcessor
from data_pipeline import DataProcessor
from raw_recorder import RawRecorder
from radar_dsp.mapping import FrameMapper, TemporalAssociator, analysis_to_ros, ros_to_analysis, adapt_virtual_cube, adapt_tdm_chirps
import multi_antenna_analysis as dsp

ROOT=Path(__file__).resolve().parents[1]


class MappingFixTests(unittest.TestCase):
    def setUp(self):
        self.p=dsp.load_profile(ROOT/'radar_configs/iwr6843_generated.cfg')
        self.settings=DspConfig()

    def target(self,az=20,el=10,range_bin=20,doppler_bin=0):
        p=self.p
        x,z=dsp.isk_positions().T
        aa,ee=np.deg2rad([az,el])
        steering=np.exp(2j*np.pi/p.wavelength_m*(x*np.sin(aa)*np.cos(ee)+z*np.sin(ee)))
        fd=doppler_bin/(p.loops*p.same_tx_s)
        offsets=np.repeat(np.arange(3)*p.chirp_s,4)
        return 100*steering[None,None,:]*np.exp(-2j*np.pi*range_bin*np.arange(p.samples)[None,:,None]/p.samples)*np.exp(2j*np.pi*fd*(np.arange(p.loops)[:,None,None]*p.same_tx_s+offsets[None,None,:]))

    def test_stationary_and_moving_model_directions(self):
        mapper=FrameMapper(self.p,self.settings)
        for az,el,doppler in [(0,0,0),(-20,10,0),(20,-10,5)]:
            result=mapper.process(self.target(az,el,doppler_bin=doppler))
            matches=[r for r in result['points'] if abs(r['range_m']-20*self.p.range_step_m)<1e-6 and abs(r['azimuth_deg']-az)<=2 and abs(r['elevation_deg']-el)<=2]
            self.assertTrue(matches,(az,el,doppler))
            self.assertFalse(any(r['validated'] for r in result['points']))

    def test_mapping_preserves_static_motion_suppresses_it(self):
        mapper=FrameMapper(self.p,self.settings);adc=self.target()
        mapping=mapper.process(adc,'mapping');motion=mapper.process(adc,'motion')
        self.assertGreater(mapping['rd'].max(),1)
        self.assertLess(motion['rd'].max(),mapping['rd'].max()*1e-20)
        self.assertEqual(len(motion['points']),0)
        moving=mapper.process(self.target(doppler_bin=5),'motion')
        self.assertGreater(moving['rd'].max(),1)

    def test_stable_clutter_removal_on_each_axis(self):
        from radar_dsp.compensation import clutter_removal
        values=np.arange(120).reshape(4,5,6)+1j
        for axis in (0,1,2,-1):
            np.testing.assert_allclose(clutter_removal(values,axis),values-values.mean(axis=axis,keepdims=True),atol=1e-12)
        repeated=np.repeat(np.array([[.1+.3j,.2+.7j]]),64,axis=0)
        np.testing.assert_equal(clutter_removal(repeated),np.zeros_like(repeated))

    def test_covariance_bartlett_equals_direct_doppler_power(self):
        mapper=FrameMapper(self.p,self.settings);adc=self.target()
        result=mapper.process(adc)
        r=dsp.range_fft(adc);indices=np.arange(63,31,-1)
        d=dsp.doppler_fft(r[:,indices]);ri=19
        fd=np.fft.fftshift(np.fft.fftfreq(self.p.loops,self.p.same_tx_s))
        compensated=np.array([dsp.tdm_compensate(row,frequency,self.p) for row,frequency in zip(d[:,ri],fd)])
        direct=np.mean(abs(compensated@mapper.steer.conj().T)**2,axis=0)/144
        self.assertTrue(np.allclose(result['ra'][ri],direct.reshape(len(mapper.el),len(mapper.az)).max(axis=0)))

    def test_ros_axes_and_mount_rotation(self):
        points=np.array([[0,1,0],[-1,1,0],[1,1,0],[0,1,1],[0,1,-1]])
        expected=np.array([[1,0,0],[1,1,0],[1,-1,0],[1,0,1],[1,0,-1]])
        np.testing.assert_equal(analysis_to_ros(points),expected)
        np.testing.assert_equal(ros_to_analysis(expected),points)
        np.testing.assert_allclose(dsp.rotate_mount((1,1,0),90),(0,1,-1),atol=1e-12)

    def test_explicit_transport_adapter_axes_and_channel_order(self):
        canonical=self.target()
        pairs=[(tx,rx) for tx in range(3) for rx in range(4)][::-1]
        transport=canonical[:,:,::-1].transpose(2,0,1)
        result=adapt_virtual_cube(transport,('channel','loop','sample'),pairs)
        np.testing.assert_equal(result,canonical)
        slots=(2,0,1);rx=(3,1,0,2)
        chirps=np.empty((192,4,64),complex)
        for slot,tx in enumerate(slots):chirps[slot::3]=canonical[:,:,tx*4:(tx+1)*4][:,:,[*rx]].transpose(0,2,1)
        np.testing.assert_equal(adapt_tdm_chirps(chirps,slots,rx),canonical)
        with self.assertRaises(ValueError):adapt_virtual_cube(transport,('channel','loop','sample'),pairs[:-1])

    def test_far_target_remains_eligible_no_range_wrap(self):
        power=np.ones((64,32));power[32,29]=1e6
        cells,_=dsp.ca_cfar(power)
        self.assertTrue(np.any((cells==[32,29]).all(1)))
        _,first=dsp.ca_cfar(np.ones((64,32)))
        power=np.ones((64,32));power[32,-1]=1e9
        _,second=dsp.ca_cfar(power)
        np.testing.assert_equal(first[:,0],second[:,0])

    def test_full_sample_fft_and_no_legacy_decimation_option(self):
        signal=RadarSignalProcessor()
        z=self.target(range_bin=30)
        rti,_,_=signal.process_time_features(z)
        # The unflipped channel-0 fast-time cube is stored in RTI history.
        self.assertEqual(np.argmax(abs(signal.rti_history[-1][0])),34)
        with self.assertRaises(ValueError):DspConfig(range_preprocessing='legacy_features')

    def test_nonfinite_frame_rejected_without_fabricated_coordinates(self):
        z=self.target();z[0,0,0]=np.nan
        with self.assertRaises(ValueError):FrameMapper(self.p,self.settings).process(z)

    def test_modes_and_timestamp_metadata(self):
        state=RuntimeState();self.assertIs(state.processing_mode,ProcessingMode.MAPPING)
        state.set_processing_mode(ProcessingMode.MOTION)
        with TemporaryDirectory() as temp:
            recorder=RawRecorder(temp)
            binary=recorder.start(RadarFrameConfig(),seconds=1,metadata_extra={'dsp_processing_mode':'mapping','active_cfg_sha256':'test'})
            recorder.submit(np.zeros(98304,dtype=np.int16),observed_sequence=7,observed_monotonic_ns=time.monotonic_ns(),source_sequence=45)
            recorder.stop();recorder.wait(None)
            meta=json.loads(binary.with_suffix('.json').read_text())
            self.assertEqual(meta['frames_written'],1)
            self.assertEqual(meta['dsp_processing_mode'],'mapping')
            self.assertIsNone(meta['udp_packet_gaps'])
            self.assertIn(',7,45,',binary.with_suffix('.frames.csv').read_text())
            times,source,digest=dsp.observation_time_axis(binary,meta,1,.04)
            self.assertEqual(len(times),1)
            self.assertEqual(source,'host_observation_monotonic_not_radar_time')
            self.assertTrue(digest)

    def test_dsp_queue_drop_does_not_drop_recording_submission(self):
        class Recorder:
            def __init__(self):self.n=0
            def submit(self,raw,**kwargs):self.n+=1
        recorder=Recorder();processor=DataProcessor('test',object(),object(),Queue(),raw_recorder=recorder)
        for i in range(20):processor._accept_frame(np.zeros(98304,dtype=np.int16),time.monotonic_ns(),i)
        self.assertEqual(recorder.n,20)
        self.assertEqual(processor.snapshot_diagnostics()['dsp_queue_dropped_frames'],12)
        self.assertIsNone(processor.snapshot_diagnostics()['upstream_lost_frames'])

    def test_temporal_links_do_not_duplicate_points_or_validate_identity(self):
        tracker=TemporalAssociator()
        row=dict(x_m=0.,y_m=1.,z_m=0.,passes_angle_quality=True,validated=False)
        first=tracker.update([dict(row)],0.)
        second=tracker.update([dict(row)],.04)
        self.assertEqual(len(second),1)
        self.assertEqual(first[0]['temporal_track_id'],second[0]['temporal_track_id'])
        self.assertTrue(second[0]['temporally_associated'])
        self.assertFalse(second[0]['validated'])
        with self.assertRaises(ValueError):tracker.update([], .04)

    def test_source_sequence_gap_is_distinct_from_unknown_native_loss(self):
        processor=DataProcessor('test',object(),object(),Queue())
        for sequence in (1,2,5,5,4):processor._accept_frame(np.zeros(98304,dtype=np.int16),time.monotonic_ns(),sequence)
        stats=processor.snapshot_diagnostics()
        self.assertEqual(stats['source_sequence_gaps'],2)
        self.assertEqual(stats['source_duplicate_frames'],1)
        self.assertEqual(stats['source_reordered_frames'],1)
        self.assertIsNone(stats['udp_packet_gaps'])

    def test_tx_slot_permutation_is_canonicalized(self):
        p=replace(self.p,tx_order=(2,0,1))
        config=replace(AppConfig(),physical_profile=p)
        signal=RadarSignalProcessor(config)
        processor=DataProcessor('test',object(),signal,Queue())
        raw=np.arange(98304,dtype=np.int16)
        np.testing.assert_equal(processor._decode_frame(raw),dsp.decode_frame(raw,p))

    def test_measured_mount_transform_has_exact_inverse(self):
        xyz=(.2,1.1,-.3)
        mounted=dsp.rotate_mount(xyz,25,-10,40)
        np.testing.assert_allclose(dsp.rotate_mount(mounted,25,-10,40,inverse=True),xyz,atol=1e-12)

    def test_optional_channel_calibration_restores_synthetic_direction(self):
        with TemporaryDirectory() as temp:
            phase=np.linspace(-1,1,12)
            calibration=Path(temp)/'synthetic_calibration.json'
            calibration.write_text(json.dumps(dict(geometry='IWR6843ISK',cfg_sha256='unit-test',phase_sign=1,
                correction_real=np.cos(phase).tolist(),correction_imag=(-np.sin(phase)).tolist(),range_bias_m=0)))
            settings=replace(self.settings,calibration_path=str(calibration))
            result=FrameMapper(self.p,settings,'unit-test').process(self.target()*np.exp(1j*phase)[None,None,:])
            self.assertTrue(any(abs(row['azimuth_deg']-20)<=2 and abs(row['elevation_deg']-10)<=2 and row['passes_angle_quality'] for row in result['points']))

    def test_geometry_aware_capon_and_music_display_models(self):
        for method in ('capon','music'):
            mapper=FrameMapper(self.p,replace(self.settings,angle_method=method))
            result=mapper.process(self.target())
            ri=19
            self.assertLessEqual(abs(result['az'][result['ra'][ri].argmax()]-20),4)
            self.assertLessEqual(abs(result['el'][result['re'][ri].argmax()]-10),4)
            self.assertEqual(result['diagnostics']['heatmap_angle_method'],method)

    def test_gui_mode_and_active_profile_context_without_hardware(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        import main
        from unittest.mock import patch
        with patch.object(main.list_ports,'comports',return_value=[]):
            app=main.RadarStreamApplication();app._build_ui()
        try:
            app.processing_group.actions()[1].trigger()
            self.assertIs(app.runtime_state.processing_mode,ProcessingMode.MOTION)
            self.assertIn('motion',app.processing_status.text())
            with patch.object(app,'open_radar') as hardware:
                app._send_radar_config_file(str(ROOT/'radar_configs/iwr6843_generated.cfg'),micro_doppler_only=False)
            hardware.assert_called_once()
            self.assertEqual(app.config.profile_cfg_sha256,dsp.sha256(ROOT/'radar_configs/iwr6843_generated.cfg'))
            self.assertIn('profileCfg',app.config.profile_cfg_text)
            self.assertEqual(app.config.physical_profile,self.p)
        finally:
            app.shutdown();app.main_window.close()


if __name__=='__main__':unittest.main()
