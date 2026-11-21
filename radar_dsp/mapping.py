"""Hardware-independent frame adapter around the existing audited DSP functions.

Input: complex [same-TX loop, ADC sample, TX-major/RX-minor channel].
No acquisition, GUI, ROS, interpolation, scene registration or surface fitting.
"""
import time
import numpy as np
import multi_antenna_analysis as dsp


def adapt_virtual_cube(data, axes, channel_pairs):
    """Explicit adapter for an already-complex 3D transport cube.

    axes names the three dimensions; channel_pairs names every (physical TX,RX)
    in transport order. These must come from the transport's descriptor, not
    from an assumption that CPSL uses RadarStream binary packing.
    """
    a=np.asarray(data)
    if a.ndim!=3 or not np.iscomplexobj(a) or len(axes)!=3 or set(axes)!={'loop','sample','channel'}:
        raise ValueError('Explicit complex loop/sample/channel axes required')
    desired=[(tx,rx) for tx in range(3) for rx in range(4)]
    pairs=[tuple(pair) for pair in channel_pairs]
    if len(pairs)!=12 or set(pairs)!=set(desired):raise ValueError('Complete unique physical channel map required')
    a=a.transpose([axes.index(name) for name in ('loop','sample','channel')])
    if a.shape[2]!=12 or not np.isfinite(a).all():raise ValueError('Invalid complex cube')
    return a[:,:,[pairs.index(pair) for pair in desired]]


def adapt_tdm_chirps(data,tx_slot_order,rx_order):
    """Explicit complex [chirp,RX,sample] -> canonical ISK cube."""
    a=np.asarray(data)
    if a.ndim!=3 or not np.iscomplexobj(a) or a.shape[0]%3 or a.shape[1]!=4:
        raise ValueError('Expected complete complex TDM chirp/RX/sample frame')
    if sorted(tx_slot_order)!=[0,1,2] or sorted(rx_order)!=[0,1,2,3]:
        raise ValueError('Physical TX/RX permutations required')
    a=a[:,[rx_order.index(rx) for rx in range(4)],:].transpose(0,2,1)
    if not np.isfinite(a).all():raise ValueError('Nonfinite transport cube')
    return np.concatenate([a[tx_slot_order.index(tx)::3] for tx in range(3)],axis=2)


def analysis_to_ros(xyz):
    """X right/Y forward/Z up -> ROS X forward/Y left/Z up."""
    a=np.asarray(xyz,float)
    if a.shape[-1]!=3 or not np.isfinite(a).all(): raise ValueError('Finite XYZ required')
    return np.stack((a[...,1],-a[...,0],a[...,2]),axis=-1)


def ros_to_analysis(xyz):
    a=np.asarray(xyz,float)
    if a.shape[-1]!=3 or not np.isfinite(a).all(): raise ValueError('Finite XYZ required')
    return np.stack((-a[...,1],a[...,0],a[...,2]),axis=-1)


class TemporalAssociator:
    """Optional one-to-one proximity links; no new points or object identity.

    Only angle-quality-passing candidates enter tracks. Hardware/host/nominal
    timestamp provenance must travel separately with the observations.
    """
    def __init__(self,max_distance_m=.15,max_gap_s=.2):
        if min(max_distance_m,max_gap_s)<=0: raise ValueError('Positive gates required')
        self.distance=max_distance_m;self.gap=max_gap_s;self.tracks={};self.next_id=0;self.last_time=None

    def update(self,points,timestamp_s):
        if not np.isfinite(timestamp_s) or (self.last_time is not None and timestamp_s<=self.last_time):
            raise ValueError('Strictly increasing finite frame timestamps required')
        self.last_time=timestamp_s
        self.tracks={k:v for k,v in self.tracks.items() if timestamp_s-v[1]<=self.gap}
        unused=set(self.tracks);pairs=[]
        for i,row in enumerate(points):
            row.update(temporal_track_id=None,temporally_associated=False,track_observations=0)
            if not row['passes_angle_quality']: continue
            xyz=np.array([row['x_m'],row['y_m'],row['z_m']])
            for k in unused:
                distance=np.linalg.norm(xyz-self.tracks[k][0])
                if distance<=self.distance:pairs.append((distance,i,k))
        used_points=set();assignments={}
        for _,i,k in sorted(pairs):
            if i not in used_points and k in unused:
                assignments[i]=k;used_points.add(i);unused.remove(k)
        for i,row in enumerate(points):
            if not row['passes_angle_quality']:continue
            k=assignments.get(i)
            count=1
            if k is None:k=self.next_id;self.next_id+=1
            else:count=self.tracks[k][2]+1
            xyz=np.array([row['x_m'],row['y_m'],row['z_m']])
            self.tracks[k]=(xyz,timestamp_s,count)
            row.update(temporal_track_id=k,temporally_associated=count>1,track_observations=count)
        return points


class FrameMapper:
    """Reuse range/Doppler, joint geometry, compensation and corrected CFAR."""
    def __init__(self,profile,settings,cfg_hash=''):
        self.profile=profile;self.settings=settings
        self.az,self.el,self.steer=dsp.steering_grid(profile,settings.phase_sign)
        self.projectors=(self.steer.conj()[:,:,None]*self.steer[:,None,:]).reshape(len(self.steer),144)
        self.gains,self.bias,self.calibration=dsp.load_calibration(
            settings.calibration_path or None,cfg_hash,settings.phase_sign)

    def process(self,adc,mode='mapping',timestamp_s=None,sequence=None,angle_method=None):
        start=time.perf_counter();p=self.profile;s=self.settings
        adc=np.asarray(adc)
        if adc.shape!=(p.loops,p.samples,12) or not np.iscomplexobj(adc):
            raise ValueError('Expected complex [loops,samples,12] canonical ISK cube')
        if not np.isfinite(adc).all(): raise ValueError('Nonfinite ADC frame')
        if mode not in ('mapping','motion'): raise ValueError('Unknown DSP processing mode')
        r=dsp.range_fft(adc);signed=np.fft.fftfreq(p.samples)*p.samples*p.range_step_m
        indices=np.flatnonzero(signed<0 if s.beat_branch=='negative' else signed>0)
        indices=indices[np.argsort(abs(signed[indices]))]
        ranges=abs(signed[indices])-self.bias
        r=r[:,indices];d=dsp.doppler_fft(r,mode=='motion')
        fd=np.fft.fftshift(np.fft.fftfreq(p.loops,p.same_tx_s))
        power=np.mean(abs(d)**2,axis=2)
        cells,noise,audit=dsp.ca_cfar(power,pfa=s.mapping_pfa,minimum_snr_db=s.mapping_min_snr_db,return_audit=True)
        gate=(ranges>=s.mapping_min_range_m)&(ranges<=s.mapping_max_range_m)
        # Compensate every Doppler slot before angular accumulation. This
        # preserves static and moving power rather than assuming fd==0.
        offsets=np.array([p.tx_order.index(tx)*p.chirp_s for tx in range(3) for rx in range(4)])
        corrected=d*np.exp(-2j*np.pi*fd[:,None,None]*offsets[None,None,:])*self.gains
        # All-Doppler average angular power, equivalent to compensated chirp
        # covariance Bartlett power. Process range batches to bound memory.
        method=angle_method or s.angle_method
        if method not in ('geometry','capon','music'): raise ValueError('Unknown angle spectrum method')
        covariance=np.einsum('drc,drk->rck',corrected,corrected.conj(),optimize=True)/p.loops
        if method=='geometry':
            scores=(covariance.reshape(len(ranges),144)@self.projectors.T).real/12**2
        else:
            scores=np.zeros((len(ranges),len(self.steer)))
            for ri,cov in enumerate(covariance):
                energy=float(np.trace(cov).real)
                if energy<=1e-25:continue
                if method=='capon':
                    # Actual-position steering; no ULA forward/backward assumption.
                    inverse=np.linalg.pinv(cov+np.eye(12)*energy/12*1e-3)
                    denominator=np.sum((self.steer.conj()@inverse)*self.steer,axis=1).real
                else:
                    # One-source MUSIC is an explicit model for the display.
                    _,vectors=np.linalg.eigh(cov);noise_basis=vectors[:,:-1]
                    denominator=np.sum(abs(self.steer.conj()@noise_basis)**2,axis=1)
                scores[ri]=1/np.maximum(denominator,1e-30)
        joint=np.maximum(scores,0).reshape(len(ranges),len(self.el),len(self.az))
        points=[];in_range=0
        for di,ri in cells:
            if not gate[ri]: continue
            if mode=='motion' and abs(fd[di]*p.wavelength_m/2)<s.motion_min_abs_velocity_m_s: continue
            in_range+=1
            peaks=dsp.estimate_angles(corrected[di,ri],self.steer,self.az,self.el,max_peaks=s.mapping_max_angle_peaks)
            for aa,ee,coherence in peaks:
                flags=[]
                if coherence<s.mapping_min_coherence: flags.append('low_coherence')
                if abs(aa)==60: flags.append('azimuth_boundary')
                if abs(ee)==20: flags.append('elevation_boundary')
                xyz=dsp.rotate_mount(dsp.xyz_from_angles(ranges[ri],aa,ee),s.mount_roll_deg,s.mount_pitch_deg,s.mount_yaw_deg)
                ros=analysis_to_ros(xyz)
                points.append(dict(sequence=sequence,timestamp_s=timestamp_s,
                    range_m=float(ranges[ri]),azimuth_deg=aa,elevation_deg=ee,
                    x_m=xyz[0],y_m=xyz[1],z_m=xyz[2],
                    ros_x_m=float(ros[0]),ros_y_m=float(ros[1]),ros_z_m=float(ros[2]),
                    radial_velocity_m_s=float(fd[di]*p.wavelength_m/2),
                    signal_power_adc2=float(power[di,ri]),
                    snr_db=float(10*np.log10(max(power[di,ri],1e-30)/max(noise[di,ri],1e-30))),
                    angular_coherence=coherence,quality_flags=';'.join(flags),
                    passes_angle_quality=not flags,temporally_associated=False,validated=False,
                    position_status='angle_quality_pass' if not flags else 'uncertain_candidate'))
        elapsed=(time.perf_counter()-start)*1000
        diagnostics=dict(sequence=sequence,timestamp_s=timestamp_s,mode=mode,
            adc_samples=p.samples,tx=3,rx=4,range_sample_rate_hz=p.fs_hz,
            rd_cells=int(power.size),above_cfar=int(audit['threshold_pass_cells']),
            rd_detections=len(cells),in_range_detections=in_range,xyz_candidates=len(points),
            angle_quality_pass=sum(row['passes_angle_quality'] for row in points),
            processing_ms=elapsed,heatmap_angle_method=method,detection_angle_method='joint_bartlett_snapshot',calibration=self.calibration,mount_roll_deg=s.mount_roll_deg,mount_pitch_deg=s.mount_pitch_deg,mount_yaw_deg=s.mount_yaw_deg,
            coordinate_frame='X right/Y forward/Z up; ROS transform explicit',
            physical_orientation_validated=False)
        return dict(points=points,diagnostics=diagnostics,ranges=ranges,fd=fd,
                    range_power=np.mean(abs(r)**2,axis=(0,2)),rd=power,
                    ra=joint.max(axis=1),re=joint.max(axis=2),az=self.az,el=self.el)
