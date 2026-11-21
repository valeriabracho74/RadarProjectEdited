"""Offline IWR6843ISK / DCA1000 analysis. Requires numpy and matplotlib.

Run inside RadarStream-main:
  python multi_antenna_analysis.py --bin raw_recordings/NAME.bin
For candidate angles/XYZ explicitly select the beat branch, e.g.:
  python multi_antenna_analysis.py --bin raw_recordings/NAME.bin --range-side negative
CSV/data save automatically under multi_antenna_results/<recording>/<run>.
Uncertain angle candidates are exported with quality flags by default.
Graphs still open by default; --output DIR saves graphs instead.
No acquisition-app imports or hardware communication. Estimates are unvalidated.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np

C = 299792458.0
GEOMETRY_SOURCE = 'https://dev.ti.com/tirex/explore/content/radar_academy_2_10_00_2/_build_radar_academy_2_10_00_2/source/integration/antenna_design.html'
RAW_SOURCE = 'https://www.ti.com/lit/SWRA581'


@dataclass(frozen=True)
class Profile:
    samples: int
    loops: int
    tx_order: tuple
    receivers: tuple
    fs_hz: float
    slope_hz_s: float
    chirp_s: float
    frame_s: float
    center_hz: float

    @property
    def chirps(self): return self.loops * len(self.tx_order)
    @property
    def words(self): return 2 * self.samples * self.chirps * len(self.receivers)
    @property
    def same_tx_s(self): return len(self.tx_order) * self.chirp_s
    @property
    def wavelength_m(self): return C / self.center_hz
    @property
    def range_step_m(self): return C * self.fs_hz / (2 * self.slope_hz_s * self.samples)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load_profile(path):
    """Strictly support three distinct TX chirps, one shared profile, four RX."""
    commands = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        t = line.split('%')[0].split('#')[0].split()
        if t: commands.setdefault(t[0], []).append(t)
    def one(key):
        if len(commands.get(key, [])) != 1: raise ValueError('Requires exactly one ' + key)
        return commands[key][0]
    if commands.get('advFrameCfg'): raise ValueError('Advanced frames unsupported')
    p, f, ch, adc, buf = [one(k) for k in ('profileCfg','frameCfg','channelCfg','adcCfg','adcbufCfg')]
    if list(map(int, adc[1:3])) != [2, 1] or list(map(int, buf[2:5])) != [0, 1, 1]:
        raise ValueError('Requires project 16-bit complex, sample-swap=1, non-interleaved ADC settings')
    if int(ch[1]) != 15 or int(ch[2]) != 7: raise ValueError('Requires all four RX and all three TX enabled')
    start, end, loops = map(int, f[1:4])
    if end - start != 2: raise ValueError('Requires three chirp indexes per loop')
    tx = []
    for index in range(start, end+1):
        entries = [t for t in commands.get('chirpCfg', []) if int(t[1]) <= index <= int(t[2])]
        if len(entries) != 1: raise ValueError('Chirp indexes missing or overlapping')
        t = entries[0]; mask = int(t[8])
        if int(t[3]) != int(p[1]) or any(float(x) != 0 for x in t[4:8]) or mask not in (1,2,4):
            raise ValueError('Requires single-TX TDM with no chirp profile variations')
        tx.append({1:0,2:1,4:2}[mask])
    if set(tx) != {0,1,2}: raise ValueError('Each TX must appear once per loop')
    ns = int(p[10]); fs = float(p[11])*1e3; slope = float(p[8])*1e12
    adc_start = float(p[4])*1e-6; ramp = float(p[5])*1e-6
    chirp = (float(p[3])+float(p[5]))*1e-6; frame = float(f[5])*1e-3
    center = float(p[2])*1e9 + slope*(adc_start+ns/fs/2)
    values = (fs,slope,chirp,frame,center)
    if ns < 4 or ns%2 or loops < 16 or not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError('Invalid FFT dimensions or RF settings')
    if adc_start+ns/fs > ramp+1e-12 or loops*3*chirp > frame+1e-12:
        raise ValueError('ADC window or burst exceeds configured timing')
    return Profile(ns,loops,tuple(tx),(0,1,2,3),fs,slope,chirp,frame,center)


def validate_recording(binary, metadata, p):
    m = json.loads(Path(metadata).read_text(encoding='utf-8-sig'))
    expected = dict(adc_samples=p.samples,chirps_per_frame=p.chirps,tx_antennas=3,rx_antennas=4,
                    int16_values_per_frame=p.words,bytes_per_frame=2*p.words)
    for key, value in expected.items():
        if type(m.get(key)) is not int or m[key] != value: raise ValueError('Metadata mismatch: '+key)
    if m.get('dtype') != '<i2': raise ValueError('Requires little-endian signed int16')
    size = Path(binary).stat().st_size
    if not size or size%(2*p.words): raise ValueError('Empty capture or partial frame; no silent truncation')
    nf = size//(2*p.words)
    if type(m.get('frames_written')) is not int or m['frames_written'] != nf or m.get('file_bytes') != size:
        raise ValueError('Metadata file/frame counts disagree with binary')
    if m.get('error'): raise ValueError('Recorder error: '+str(m['error']))
    return m, np.memmap(binary,dtype='<i2',mode='r',shape=(nf,p.words))


def decode_frame(raw, p):
    """Output [loop, sample, TX-major/RX-minor virtual channel].

    Flattened groups are I0 I1 Q0 Q1. RX sample blocks are inside
    chirps; sequential chirps follow p.tx_order. Exactly the project's
    decoder for TX0,TX1,TX2; other permutations are explicitly canonicalized.
    """
    if np.asarray(raw).size != p.words: raise ValueError('Wrong frame length')
    g = np.asarray(raw).reshape(-1,4).astype(np.float64)
    z = (g[:,:2]+1j*g[:,2:]).reshape(p.chirps,4,p.samples)
    by_tx = {tx:z[slot::3].transpose(0,2,1) for slot,tx in enumerate(p.tx_order)}
    return np.concatenate([by_tx[tx] for tx in range(3)],axis=2)


def decode_capture(raw, p):
    """Explicit materialization helper; analysis itself streams frames."""
    return np.stack([decode_frame(row,p) for row in raw])


def range_fft(adc, remove_dc=True):
    data = adc-adc.mean(axis=1,keepdims=True) if remove_dc else adc
    w = np.hanning(adc.shape[1])
    return np.fft.fft(data*w[None,:,None],axis=1)/w.sum()


def doppler_fft(ranges, suppress_static=False):
    if suppress_static:
        centered=ranges-ranges[:1]
        data=centered-centered.mean(axis=0,keepdims=True)
    else:
        data=ranges
    w = np.hanning(data.shape[0])
    return np.fft.fftshift(np.fft.fft(data*w[:,None,None],axis=0),axes=0)/w.sum()


def isk_positions():
    """Physical virtual coordinates (x,z), metres, TI ISK layout at design 60 GHz.
    TX0: x=0..3,z=0; TX1: x=2..5,z=1; TX2: x=4..7,z=0;
    all integer positions in design-lambda/2 units. Not ODS/AOP geometry.
    """
    half = C/60e9/2
    return np.array([(rx+dx,dz) for dx,dz in ((0,0),(2,1),(4,0)) for rx in range(4)])*half


def tdm_compensate(snapshot, fd_hz, p):
    offsets = np.array([p.tx_order.index(tx)*p.chirp_s for tx in range(3) for rx in range(4)])
    return snapshot*np.exp(-2j*np.pi*fd_hz*offsets)


def steering_grid(p, phase_sign=1, step=2, azimuth_limit=60, elevation_limit=20):
    az = np.arange(-azimuth_limit,azimuth_limit+step/2,step,dtype=float)
    el = np.arange(-elevation_limit,elevation_limit+step/2,step,dtype=float)
    aa,ee = np.meshgrid(np.deg2rad(az),np.deg2rad(el))
    u = np.stack([np.sin(aa)*np.cos(ee),np.sin(ee)],axis=-1).reshape(-1,2)
    steering = np.exp(phase_sign*2j*np.pi/p.wavelength_m*(u@isk_positions().T))
    return az,el,steering


def estimate_angle(snapshot, steering, az, el):
    """Conventional 2D Bartlett scan; one angular maximum per RD cell."""
    energy = float(np.vdot(snapshot,snapshot).real)
    score = abs(steering.conj()@snapshot)**2/12**2
    im = score.reshape(len(el),len(az)); j,i = np.unravel_index(im.argmax(),im.shape)
    coherence = float(score.max()/max(energy/12,1e-30))
    return float(az[i]),float(el[j]),coherence,im


def estimate_angles(snapshot, steering, az, el, max_peaks=3, separation_deg=20, relative_db_limit=3, return_audit=False):
    """Separated local Bartlett maxima; these remain unresolved-target candidates.

    Separation avoids filling a main lobe with duplicate points. It does not
    establish the physical angular resolution or make sidelobes real objects.
    """
    energy=float(np.vdot(snapshot,snapshot).real)
    image=(abs(steering.conj()@snapshot)**2/12**2).reshape(len(el),len(az))
    padded=np.pad(image,1,constant_values=-np.inf)
    local=np.ones(image.shape,bool)
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr or dc:
                local &= image>=padded[1+dr:1+dr+len(el),1+dc:1+dc+len(az)]
    relative=image>=image.max()*10**(-relative_db_limit/10)
    candidates=np.argwhere(local & relative)
    candidates=sorted(candidates,key=lambda cell:image[tuple(cell)],reverse=True)
    peaks=[];directions=[]
    for j,i in candidates:
        aa,ee=np.deg2rad([az[i],el[j]])
        direction=np.array([np.cos(ee)*np.sin(aa),np.cos(ee)*np.cos(aa),np.sin(ee)])
        if any(np.rad2deg(np.arccos(np.clip(direction@other,-1,1)))<separation_deg for other in directions): continue
        peaks.append((float(az[i]),float(el[j]),float(image[j,i]/max(energy/12,1e-30))))
        directions.append(direction)
    audit=dict(angle_local_maxima=int(local.sum()),angle_within_relative_db=len(candidates),
               angle_after_separation=len(peaks),angle_after_peak_cap=min(len(peaks),max_peaks))
    peaks=peaks[:max_peaks]
    return (peaks,audit) if return_audit else peaks


def xyz_from_angles(r, az_deg, el_deg):
    az,el = np.deg2rad([az_deg,el_deg])
    return (float(r*np.cos(el)*np.sin(az)),float(r*np.cos(el)*np.cos(az)),float(r*np.sin(el)))


def rotate_mount(xyz, roll_deg=0, pitch_deg=0, yaw_deg=0, inverse=False):
    """Measured sensor-to-mount rotation: Rz(yaw) Rx(pitch) Ry(roll).

    Analysis axes: X right/Y forward/Z up. Roll is about forward Y,
    pitch about lateral X, yaw about up Z. No angle is inferred from data.
    inverse=True applies the transpose of this exact rotation.
    """
    roll,pitch,yaw=np.deg2rad([roll_deg,pitch_deg,yaw_deg])
    cr,sr=np.cos(roll),np.sin(roll);cp,sp=np.cos(pitch),np.sin(pitch);cy,sy=np.cos(yaw),np.sin(yaw)
    ry=np.array([[cr,0,sr],[0,1,0],[-sr,0,cr]])
    rx=np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    rotation=rz@rx@ry
    result=(rotation.T if inverse else rotation)@np.asarray(xyz)
    return tuple(float(v) for v in result)


def compare_angle_models(snapshots,p,min_coherence):
    """Compare global maxima on identical CFAR cells; never choose a model automatically."""
    if len(snapshots)==0: return []
    snapshots=np.asarray(snapshots);energy=np.sum(abs(snapshots)**2,axis=1)
    reports=[]
    for sign in (1,-1):
        for az_limit,el_limit in ((60,20),(60,40),(90,20),(90,60)):
            az,el,steer=steering_grid(p,sign,azimuth_limit=az_limit,elevation_limit=el_limit)
            scores=abs(snapshots@steer.conj().T)**2/(12*np.maximum(energy[:,None],1e-30))
            j,i=np.divmod(scores.argmax(axis=1),len(az));coherence=scores.max(axis=1)
            valid=coherence>=min_coherence;ab=abs(az[i])>=az_limit;eb=abs(el[j])>=el_limit
            reports.append(dict(phase_sign=sign,azimuth_limit_deg=az_limit,elevation_limit_deg=el_limit,
                cfar_cells=len(snapshots),coherent_global_maxima=int(valid.sum()),azimuth_boundary=int(sum(valid&ab)),
                elevation_boundary=int(sum(valid&eb)),interior_coherent_global_maxima=int(sum(valid&~ab&~eb)),
                outside_original_azimuth_fov=int(sum(valid&(abs(az[i])>60))),median_abs_azimuth_deg=float(np.median(abs(az[i])))))
    return reports


def _range_shift(values, offset, fill):
    """Shift range without joining the far and near ends; Doppler alone wraps."""
    out=np.full_like(values,fill,dtype=float)
    if offset==0: return values.astype(float,copy=True)
    if abs(offset)>=values.shape[1]: return out
    if offset>0: out[:,offset:]=values[:,:-offset]
    else: out[:,:offset]=values[:,-offset:]
    return out


def ca_cfar(power, pfa=1e-3, train=(4,3), guard=(1,1), minimum_snr_db=8,
            edge_policy='truncate', return_audit=False):
    """Local CA-CFAR with counted, truncated range-edge training windows.

    The former policy made the 1.405 m cell ineligible regardless of signal.
    Truncation retains the configured threshold/SNR and adapts the CFAR scale
    to the actual training-cell count. This is not calibrated false-alarm PFA.
    --cfar-edge-policy legacy provides an identical before/after comparison.
    """
    if not 0<pfa<1 or edge_policy not in ('truncate','legacy'): raise ValueError('Invalid CFAR settings')
    power=np.asarray(power,float)
    if power.ndim!=2 or not np.isfinite(power).all() or np.any(power<0):
        raise ValueError('CFAR requires finite nonnegative 2D linear power')
    td,tr=train;gd,gr=guard;hd,hr=td+gd,tr+gr
    total=np.zeros_like(power);count=np.zeros_like(power)
    for di in range(-hd,hd+1):
        for ri in range(-hr,hr+1):
            if abs(di)<=gd and abs(ri)<=gr: continue
            row=np.roll(power,di,axis=0)
            if edge_policy=='legacy':
                total+=np.roll(row,ri,axis=1);count+=1
            else:
                total+=_range_shift(row,ri,0)
                count+=_range_shift(np.ones_like(power),ri,0)
    safe=np.maximum(count,1);noise=total/safe
    alpha=safe*(pfa**(-1/safe)-1)
    threshold=noise*np.maximum(alpha,10**(minimum_snr_db/10))
    above=(power>threshold)&(count>0)
    eligible=above.copy()
    if edge_policy=='legacy': eligible[:,:hr]=False;eligible[:,-hr:]=False
    local=np.ones_like(eligible)
    for di in (-1,0,1):
        for ri in (-1,0,1):
            if not (di or ri): continue
            row=np.roll(power,di,axis=0)
            neighbor=np.roll(row,ri,axis=1) if edge_policy=='legacy' else _range_shift(row,ri,-np.inf)
            local &= power>=neighbor
    selected=eligible&local
    result=(np.argwhere(selected),noise)
    if not return_audit: return result
    audit=dict(cells_total=power.size,threshold_pass_cells=int(above.sum()),
               edge_policy_pass_cells=int(eligible.sum()),nms_cells=int(selected.sum()),
               threshold_pass_by_range=above.sum(axis=0),nms_by_range=selected.sum(axis=0),
               training_cells_by_range=count[0].astype(int))
    return (*result,audit)


def load_calibration(path, cfg_hash, phase_sign):
    if path is None: return np.ones(12,dtype=complex),0.0,'none'
    m=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if m.get('geometry')!='IWR6843ISK' or m.get('cfg_sha256')!=cfg_hash or m.get('phase_sign')!=phase_sign:
        raise ValueError('Calibration geometry/CFG/phase-sign mismatch')
    gains=np.asarray(m['correction_real'],float)+1j*np.asarray(m['correction_imag'],float)
    bias=float(m.get('range_bias_m',0))
    if gains.shape!=(12,) or not np.isfinite(gains).all() or np.any(abs(gains)<1e-12) or not np.isfinite(bias):
        raise ValueError('Invalid calibration coefficients')
    return gains,bias,str(Path(path).resolve())


def build_phase_calibration(raw,p,distance,side,cfg_hash,phase_sign):
    """Phase-only calibration for an isolated stationary boresight reflector.
    No automatic range bias or semantic target identification.
    """
    f=np.fft.fftfreq(p.samples,1/p.fs_hz)
    r=C*f/(2*p.slope_hz_s)
    eligible=np.where(r<0 if side=='negative' else r>0)[0]
    k=eligible[np.argmin(abs(abs(r[eligible])-distance))]
    cross=np.zeros(12,complex);energy=np.zeros(12)
    for row in raw:
        snap=range_fft(decode_frame(row,p))[:,k,:]
        cross+=np.mean(snap*np.conj(snap[:,0,None]),axis=0)
        energy+=np.mean(abs(snap)**2,axis=0)
    corr=np.exp(-1j*np.angle(cross))
    coherence=abs(cross)/np.sqrt(np.maximum(energy*energy[0],1e-30))
    if np.any(coherence<.8): raise ValueError('Calibration channel coherence <0.8; isolate a stationary boresight target')
    return dict(geometry='IWR6843ISK',cfg_sha256=cfg_hash,phase_sign=phase_sign,
                correction_real=corr.real.tolist(),correction_imag=corr.imag.tolist(),range_bias_m=0.0,
                target_distance_m=distance,selected_bin_m=float(abs(r[k])),channel_coherence=coherence.tolist(),
                note='Phase-only candidate. Verify with independent left/right/high/low captures. No range bias fitted.')


def relative_db(power,reference=None):
    ref=float(np.max(power) if reference is None else reference)
    return 10*np.log10(np.maximum(power,1e-30)/max(ref,1e-30))


def analyze(raw,p,args,gains,bias):
    nf=len(raw);ns=p.samples;nl=p.loops
    signed=C*np.fft.fftfreq(ns,1/p.fs_hz)/(2*p.slope_hz_s)
    if args.range_side=='signed': indexes=np.argsort(signed);ranges=signed[indexes]
    else:
        indexes=np.where(signed<0 if args.range_side=='negative' else signed>0)[0]
        indexes=indexes[np.argsort(abs(signed[indexes]))];ranges=abs(signed[indexes])-bias
    fd=np.fft.fftshift(np.fft.fftfreq(nl,p.same_tx_s));velocity=args.velocity_sign*fd*p.wavelength_m/2
    rt=np.zeros((nf,len(indexes)));dt=np.zeros((nf,nl));dtc=np.zeros_like(dt)
    rd=np.zeros((nl,len(indexes)));rdc=np.zeros_like(rd)
    signed_original=np.zeros((ns,12));signed_clean=np.zeros_like(signed_original)
    az,el,steer=steering_grid(p,args.phase_sign,azimuth_limit=args.azimuth_limit,elevation_limit=args.elevation_limit)
    ra=np.zeros((len(indexes),len(az)));re=np.zeros((len(indexes),len(el)))
    intensity_sum = np.zeros((len(indexes), len(el), len(az)))
    conjugate_steering = steer.conj().T
    records=[];rms=np.zeros(12);means=np.zeros(12,complex);duplicates=0;rails=0;zeros=0
    phase_cross=np.zeros(12,complex);phase_energy=np.zeros(12)
    nonfinite_samples = 0
    sample_power=np.zeros(ns);sample_peak_counts=np.zeros(ns,dtype=int);boundary_rejections=0
    frame_counts=[];rejected=[];audit_snapshots=[]
    cfar_range_audit=[]
    gate=(abs(ranges)>=args.min_range)&(abs(ranges)<=args.max_range)
    if not gate.any(): raise ValueError('Requested range interval contains no bins')
    example=decode_frame(raw[0],p)[0]
    import time
    frame_processing_ms=[]
    for fi,row in enumerate(raw):
        frame_started=time.perf_counter()
        if fi and np.array_equal(row,raw[fi-1]): duplicates+=1
        rails+=int(np.count_nonzero((row==-32768)|(row==32767)));zeros+=int(np.count_nonzero(row==0))
        z=decode_frame(row,p);rms+=np.mean(abs(z)**2,axis=(0,1))/nf;means+=z.mean(axis=(0,1))/nf
        nonfinite_samples += int(np.count_nonzero(~np.isfinite(z)))
        per_sample=np.mean(abs(z)**2,axis=(0,2));sample_power+=per_sample/nf;sample_peak_counts[int(per_sample.argmax())]+=1
        repaired_indices = getattr(args, 'interpolate_samples', [])
        if repaired_indices:
            original_adc = z.copy()
            good = np.ones(ns, dtype=bool)
            good[repaired_indices] = False
            sample_axis = np.arange(ns)
            for loop in range(nl):
                for channel in range(12):
                    z[loop, repaired_indices, channel] = (
                        np.interp(sample_axis[repaired_indices], sample_axis[good], original_adc[loop, good, channel].real)
                        + 1j*np.interp(sample_axis[repaired_indices], sample_axis[good], original_adc[loop, good, channel].imag))
        original=range_fft(z,False);r=range_fft(z,not args.keep_adc_dc)
        signed_original+=np.mean(abs(original)**2,axis=0)/nf;signed_clean+=np.mean(abs(r)**2,axis=0)/nf
        r=r[:,indexes,:];d=doppler_fft(r);dc=doppler_fft(r,True)
        power=np.mean(abs(d)**2,axis=2);pc=np.mean(abs(dc)**2,axis=2)
        rt[fi]=np.mean(abs(r)**2,axis=(0,2));rd+=power/nf;rdc+=pc/nf
        dt[fi]=power[:,gate].mean(axis=1);dtc[fi]=pc[:,gate].mean(axis=1)
        # Channel phase diagnostic at strongest mean-power range cell (not a target identity).
        k=int(rt[fi].argmax());snap=d[nl//2,k,:]
        phase_cross+=snap*np.conj(snap[0]);phase_energy+=abs(snap)**2
        counts=dict(frame=fi,time_s=fi*p.frame_s,cfar_cells=0,in_range_cells=0,angle_peaks=0,low_coherence=0,boundary_rejected=0,valid_points=0,cap_rejected=0,exported_points=0,uncertain_exported_points=0,cells_total=0,threshold_pass_cells=0,edge_policy_pass_cells=0,nms_cells=0,angle_local_maxima=0,angle_within_relative_db=0,angle_after_separation=0,angle_after_peak_cap=0)
        # Zero Doppler retains stationary reflectors. Average linear power
        # across frames before conversion to dB, with no detection threshold.
        static = d[nl // 2] * gains[None, :]
        scores = np.abs(static @ conjugate_steering) ** 2 / steer.shape[1] ** 2
        joint = scores.reshape(len(indexes), len(el), len(az))
        intensity_sum += joint
        if args.range_side != 'signed':
            ra+=joint.max(axis=1)/nf;re+=joint.max(axis=2)/nf
            detection_power=pc if args.suppress_static else power
            candidates,noise,cfar_audit=ca_cfar(detection_power,args.pfa,minimum_snr_db=args.min_snr,edge_policy=args.cfar_edge_policy,return_audit=True)
            for key in ('cells_total','threshold_pass_cells','edge_policy_pass_cells','nms_cells'): counts[key]=cfar_audit[key]
            for ri in range(len(ranges)):
                cfar_range_audit.append(dict(frame=fi,range_m=float(ranges[ri]),in_requested_range=bool(gate[ri]),
                    threshold_pass_cells=int(cfar_audit['threshold_pass_by_range'][ri]),nms_cells=int(cfar_audit['nms_by_range'][ri]),
                    training_cells=int(cfar_audit['training_cells_by_range'][ri]),legacy_edge_excluded=bool(ri<4 or ri>=len(ranges)-4)))
            counts['cfar_cells']=len(candidates)
            candidates=[(int(di),int(ri)) for di,ri in candidates if gate[ri]]
            counts['in_range_cells']=len(candidates)
            cube=dc if args.suppress_static else d
            frame_records=[]
            for di,ri in candidates:
                snap=tdm_compensate(cube[di,ri,:],fd[di],p)*gains
                if args.audit_angle_model: audit_snapshots.append(snap.copy())
                peaks,angle_audit=estimate_angles(snap,steer,az,el,args.max_angle_peaks,args.angle_separation,args.angle_peak_db,return_audit=True)
                for key,value in angle_audit.items(): counts[key]+=value
                counts['angle_peaks']+=len(peaks)
                for peak_rank,(azimuth,elevation,coherence) in enumerate(peaks,1):
                    ab=bool(np.isclose(abs(azimuth),args.azimuth_limit));eb=bool(np.isclose(abs(elevation),args.elevation_limit))
                    boundary=ab or eb
                    distance=float(ranges[ri]);sensor_xyz=xyz_from_angles(distance,azimuth,elevation)
                    x,y,zpos=rotate_mount(sensor_xyz,args.mount_roll_deg,getattr(args,'mount_pitch_deg',0),getattr(args,'mount_yaw_deg',0))
                    record=dict(frame=fi,time_s=fi*p.frame_s,range_m=distance,doppler_hz=float(fd[di]),
                        radial_velocity_m_s=float(velocity[di]),azimuth_deg=azimuth,elevation_deg=elevation,
                        x_m=x,y_m=y,z_m=zpos,sensor_x_m=sensor_xyz[0],sensor_y_m=sensor_xyz[1],sensor_z_m=sensor_xyz[2],
                        mount_roll_deg=args.mount_roll_deg,mount_pitch_deg=getattr(args,'mount_pitch_deg',0),mount_yaw_deg=getattr(args,'mount_yaw_deg',0),snr_db=float(10*np.log10(detection_power[di,ri]/max(noise[di,ri],1e-30))),
                        signal_power_adc2=float(detection_power[di,ri]),
                        angular_coherence=coherence,angle_at_search_boundary=boundary,azimuth_at_boundary=ab,elevation_at_boundary=eb,
                        outside_nominal_search=bool(abs(azimuth)>60 or abs(elevation)>20),
                        range_bin=int(indexes[ri]),doppler_bin=di,angle_peak_rank=peak_rank,validated=False)
                    # Keep uncertain directions by default; never confuse them with
                    # independently validated target positions.
                    flags=[]
                    low=coherence<args.min_coherence
                    if low:
                        counts['low_coherence']+=1;flags.append('low_coherence')
                    if ab: flags.append('azimuth_boundary')
                    if eb: flags.append('elevation_boundary')
                    uncertain=low or boundary
                    record.update(quality_flags=';'.join(flags),passes_angle_quality=not uncertain,
                                  retained_uncertain=uncertain)
                    reject_reasons=[]
                    if low: reject_reasons.append('low_coherence')
                    if boundary and not args.include_boundary:
                        reject_reasons.extend(flag for flag in flags if flag.endswith('_boundary'))
                    if getattr(args,'reject_uncertain',False) and reject_reasons:
                        if boundary and not low:
                            boundary_rejections+=1;counts['boundary_rejected']+=1
                        rejected.append(dict(record,rejection_reason=';'.join(reject_reasons)));continue
                    frame_records.append(record)
            counts['valid_points']=sum(row['passes_angle_quality'] for row in frame_records)
            uncapped_count=len(frame_records)
            frame_records.sort(key=lambda row:row['snr_db'],reverse=True)
            if args.max_detections>0:
                rejected.extend(dict(record,rejection_reason='accepted_point_cap') for record in frame_records[args.max_detections:])
                frame_records=frame_records[:args.max_detections]
            counts['cap_rejected']=uncapped_count-len(frame_records)
            counts['uncertain_exported_points']=sum(row['retained_uncertain'] for row in frame_records)
            counts['exported_points']=len(frame_records)
            records.extend(frame_records)
        counts['processing_ms']=(time.perf_counter()-frame_started)*1000
        frame_processing_ms.append(counts['processing_ms'])
        frame_counts.append(counts)
        if (fi+1)%50==0: print(f'Processed {fi+1}/{nf} frames',flush=True)
    return dict(ranges=ranges,signed_ranges=signed,fd=fd,velocity=velocity,times=np.arange(nf)*p.frame_s,
                rt=rt,rd=rd,rdc=rdc,dt=dt,dtc=dtc,signed_original=signed_original,signed_clean=signed_clean,
                channel_rms=np.sqrt(rms),channel_mean=means,phase_relative_deg=np.rad2deg(np.angle(phase_cross)),
                phase_coherence=abs(phase_cross)/np.sqrt(np.maximum(phase_energy*phase_energy[0],1e-30)),
                example=example,az=az,el=el,ra=ra,re=re,intensity=intensity_sum/nf,records=records,duplicate_adjacent_frames=duplicates,
                sample_power=sample_power,sample_peak_counts=sample_peak_counts,boundary_rejections=boundary_rejections,frame_counts=frame_counts,rejected=rejected,
                frame_processing_ms=frame_processing_ms,cfar_range_audit=cfar_range_audit,angle_model_audit=compare_angle_models(audit_snapshots,p,args.min_coherence) if args.audit_angle_model else [],
                nonfinite_samples=nonfinite_samples,rail_fraction=rails/raw.size,zero_fraction=zeros/raw.size)


def intensity_slice_indices(data, args):
    """Choose evenly spaced native range bins inside the plot interval."""
    lower, upper = args.intensity_range or (args.min_range, args.max_range)
    ranges = data['ranges']
    distance = np.abs(ranges) if args.range_side == 'signed' else ranges
    eligible = np.flatnonzero((distance >= lower) & (distance <= upper))
    if eligible.size == 0:
        raise ValueError('Intensity slice interval contains no range bins')
    count = min(args.intensity_slices, eligible.size)
    offsets = np.rint(np.linspace(0, eligible.size - 1, count)).astype(int)
    return eligible[offsets]


def plot_intensity_slices(data, args, reference_power=None):
    """Draw azimuth/elevation intensity planes along the range axis."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Polygon
    from matplotlib.transforms import Affine2D

    selected = intensity_slice_indices(data, args)
    ranges = data['ranges'][selected]
    planes = data['intensity'][selected]
    reference = reference_power if reference_power is not None else planes.max()
    norm = Normalize(-32, 0)
    fig = plt.figure(figsize=(max(8.2, 1.42 * len(selected)), 5.5), facecolor='white')
    ax = fig.add_axes([0.09, 0.22, 0.86, 0.59])
    ax.set_axis_off()
    ax.set_aspect('equal')
    plane_width, plane_height = 1.0, 0.92
    step_x, step_y = 0.91, 0.20
    shear = -0.14

    for position in reversed(range(len(selected))):
        x, y = position * step_x, position * step_y
        transform = Affine2D.from_values(
            plane_width, shear, 0, plane_height, x, y
        ) + ax.transData
        ax.imshow(
            relative_db(planes[position], reference), extent=(0, 1, 0, 1),
            origin='lower', interpolation='nearest', cmap='magma', norm=norm,
            transform=transform, aspect='equal', zorder=2 + len(selected) - position,
        )
        ax.add_patch(Polygon(
            [(x, y), (x + plane_width, y + shear),
             (x + plane_width, y + shear + plane_height), (x, y + plane_height)],
            closed=True, fill=False, edgecolor='#526071', linewidth=0.8,
            zorder=2.1 + len(selected) - position,
        ))
        ax.text(x + 0.49, y - 0.22, f'{ranges[position]:.3f} m',
                ha='center', va='top', fontsize=9, color='#253249', zorder=30)

    final_x = (len(selected) - 1) * step_x
    final_y = (len(selected) - 1) * step_y
    ax.set_xlim(-0.52, final_x + 1.20)
    ax.set_ylim(-0.62, final_y + 1.52)
    ax.annotate('', xy=(-0.13, 0.88), xytext=(-0.13, 0.04),
                arrowprops=dict(arrowstyle='->', color='#253249', lw=1.2))
    ax.text(-0.30, 0.46, 'Elevation', rotation=90, va='center', ha='center', fontsize=10)
    ax.annotate('', xy=(0.97, -0.46), xytext=(0.01, -0.34),
                arrowprops=dict(arrowstyle='->', color='#253249', lw=1.2))
    ax.text(0.48, -0.55, 'Azimuth', ha='center', fontsize=10)
    if len(selected) > 1:
        ax.annotate('', xy=(final_x + 0.52, final_y + 1.25), xytext=(0.36, 1.25),
                    arrowprops=dict(arrowstyle='->', color='#253249', lw=1.2))
        ax.text(final_x * 0.5 + 0.45, final_y * 0.5 + 1.42,
                'Signed range bins' if args.range_side == 'signed' else 'Increasing range',
                ha='center', fontsize=10, color='#253249')

    fig.text(0.09, 0.94, 'Radar intensity slices', fontsize=21, weight='bold', color='#182438')
    fig.text(0.09, 0.88, args.bin.stem + f"  |  {len(data['times'])} frames  |  {args.range_side} beat-frequency view",
             fontsize=10, color='#526071')
    bar = fig.add_axes([0.24, 0.135, 0.52, 0.027])
    fig.colorbar(ScalarMappable(norm=norm, cmap='magma'), cax=bar,
                 orientation='horizontal', ticks=[-32, -24, -16, -8, 0])
    reference_label = 'shared reference' if reference_power is not None else 'peak of displayed slices'
    bar.set_xlabel(f'Zero-Doppler power (dB relative to {reference_label})', fontsize=9)
    fig.text(0.09, 0.033,
             f"Sensor angles span {data['az'][0]:g} to {data['az'][-1]:g} degrees azimuth and "
             f"{data['el'][0]:g} to {data['el'][-1]:g} degrees elevation. Range uses the supplied CFG.",
             fontsize=8.5, color='#526071')
    return fig


def make_plots(data,p,args):
    import matplotlib
    matplotlib.use('TkAgg' if args.show_plots else 'Agg')
    import matplotlib.pyplot as plt
    figures=[]
    def finish(fig,name):
        fig.tight_layout();fig.canvas.manager.set_window_title('Multi-antenna: '+name.replace('_',' '))
        if not args.show_plots: fig.savefig(args.output/(name+'.png'),dpi=150);plt.close(fig)
        else: figures.append(fig)
    def edges(v,step): return np.r_[v-step/2,v[-1]+step/2]
    def heat(z,x,y,title,xlabel,ylabel,name,ref=None):
        fig,ax=plt.subplots(figsize=(10,5));im=ax.pcolormesh(x,y,relative_db(z,ref),shading='flat',vmin=-60,vmax=0)
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
        scale=args.velocity_sign*p.wavelength_m/2
        if name.startswith('range_doppler'):
            ax.secondary_xaxis('top',functions=(lambda f:f*scale,lambda v:v/scale)).set_xlabel('Radial velocity (m/s; sign unvalidated)')
        elif name.startswith('doppler_time'):
            ax.secondary_yaxis('right',functions=(lambda f:f*scale,lambda v:v/scale)).set_ylabel('Radial velocity (m/s; sign unvalidated)')
        fig.colorbar(im,ax=ax,label='Relative power (dB)',pad=.15 if name.startswith('doppler_time') else .02);finish(fig,name)
    labels=[f'T{t}R{r}' for t in range(3) for r in range(4)]
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    axes[0,0].bar(labels,data['channel_rms']);axes[0,0].set(title='All-channel RMS',ylabel='ADC counts')
    for k in range(12): axes[0,1].plot(abs(data['example'][:,k]),label=labels[k])
    axes[0,1].set(title='Frame 0 / loop 0 magnitudes',xlabel='ADC sample',ylabel='ADC counts');axes[0,1].legend(ncol=3,fontsize=6)
    axes[1,0].plot(labels,data['phase_relative_deg'],'o-');axes[1,0].set(title='Relative phase at per-frame strongest range cell',ylabel='Degrees; not calibration')
    axes[1,1].bar(labels,data['phase_coherence']);axes[1,1].set(title='Phase consistency versus T0R0',ylim=(0,1.05),ylabel='Coherence')
    for ax in axes.flat: ax.tick_params(axis='x',labelrotation=45)
    fig.suptitle(f'Decoding diagnostics | {args.bin.name}');finish(fig,'channel_diagnostics')
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for k in range(12): axes[0].plot(data['example'][:,k].real,alpha=.6);axes[1].plot(data['example'][:,k].imag,alpha=.6)
    axes[0].set(title='Frame 0 / loop 0: I, all channels',ylabel='ADC counts');axes[1].set(title='Q, all channels',xlabel='ADC sample',ylabel='ADC counts');finish(fig,'adc_iq')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(data['sample_power'],'o-');axes[0].set(title='Mean sample power across loops, frames, channels',xlabel='ADC sample',ylabel='ADC counts squared')
    axes[1].bar(np.arange(p.samples),data['sample_peak_counts']);axes[1].set(title='Frame count with strongest power at each sample',xlabel='ADC sample',ylabel='Frames')
    finish(fig,'sample_integrity')
    fig,ax=plt.subplots(figsize=(10,4))
    order=np.argsort(data['signed_ranges']);ref=data['signed_original'].mean(axis=1).max()
    ax.plot(data['signed_ranges'][order],relative_db(data['signed_original'].mean(axis=1)[order],ref),label='Original')
    ax.plot(data['signed_ranges'][order],relative_db(data['signed_clean'].mean(axis=1)[order],ref),label='Analysis input')
    ax.set(title='Both beat-frequency sides retained for diagnosis',xlabel='Signed beat-equivalent range (m)',ylabel='Power (dB, original peak reference)');ax.legend();ax.grid(alpha=.3);finish(fig,'signed_range_diagnostic')
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(data['ranges'],relative_db(data['rt'].mean(axis=0)),'o-')
    rlabel='Signed beat-equivalent range (m)' if args.range_side=='signed' else 'Candidate range (m; selected '+args.range_side+' branch)'
    ax.set(title='Range FFT: all samples, mean linear power across frames/channels',xlabel=rlabel,ylabel='Relative power (dB)');ax.grid(alpha=.3);finish(fig,'range_fft')
    re=edges(data['ranges'],p.range_step_m);de=edges(data['fd'],1/(p.loops*p.same_tx_s));te=np.arange(len(data['times'])+1)*p.frame_s
    heat(data['rt'].T,te,re,'Range-Time; nominal frame timing','Nominal time (s)',rlabel,'range_time')
    for suffix,rd,dt in [('',data['rd'],data['dt']),('_static_suppressed',data['rdc'],data['dtc'])]:
        heat(rd.T,de,re,'Range-Doppler'+suffix,'Measured Doppler frequency (Hz)',rlabel,'range_doppler'+suffix,data['rd'].max())
        heat(dt.T,te,de,f'Doppler-Time, |range| {args.min_range:g}-{args.max_range:g} m'+suffix,'Nominal time (s)','Measured Doppler frequency (Hz)','doppler_time'+suffix,data['dt'].max())
    if args.range_side!='signed':
        heat(data['ra'],edges(data['az'],2),re,'Static Range-Azimuth; candidate ISK geometry, sign/calibration unvalidated','Azimuth (degrees)',rlabel,'range_azimuth')
        heat(data['re'],edges(data['el'],2),re,'Static Range-Elevation; broad vertical beam, unvalidated','Elevation (degrees)',rlabel,'range_elevation')
        rows=data['records'];fig,axes=plt.subplots(1,2,figsize=(12,5),subplot_kw={})
        if rows:
            x=[r['x_m'] for r in rows];y=[r['y_m'] for r in rows];z=[r['z_m'] for r in rows];t=[r['time_s'] for r in rows]
            sc=axes[0].scatter(x,y,c=t,s=5,cmap='viridis');fig.colorbar(sc,ax=axes[0],label='Nominal time (s)')
        else: axes[0].text(.5,.5,'No detections passed thresholds',ha='center',transform=axes[0].transAxes)
        axes[0].set(xlabel='X right (m)',ylabel='Y forward (m)',title=f'Candidate mounting-frame XY; roll {args.mount_roll_deg:g} deg');axes[0].set_aspect('equal',adjustable='datalim')
        axes[1].remove();ax=fig.add_subplot(122,projection='3d')
        if rows: ax.scatter(x,y,z,c=t,s=4,cmap='viridis')
        ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z up (m)',title=f'Unvalidated mounting-frame XYZ; roll {args.mount_roll_deg:g} deg');finish(fig,'candidate_point_cloud')
    intensity_fig = plot_intensity_slices(data, args, args.intensity_reference_power)
    intensity_fig.canvas.manager.set_window_title('Multi-antenna: radar intensity slices')
    # Also save this figure for side-by-side comparison after an interactive run.
    intensity_fig.savefig(args.output / 'radar_intensity_slices.png', dpi=170)
    if not args.show_plots:
        plt.close(intensity_fig)
    else:
        figures.append(intensity_fig)
    if args.show_plots:
        print(f'Opening {len(figures)} windows; CSV/data saved in {args.output}. Close windows to finish.',flush=True)
        plt.show()


def observation_time_axis(binary,metadata,nframes,period_s):
    name=metadata.get('frame_timestamps_file')
    if not name:return np.arange(nframes)*period_s,'nominal_frame_period_unverified',None
    if Path(name).name!=name:raise ValueError('Timestamp sidecar must be beside the binary')
    path=Path(binary).parent/name
    with path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=nframes or [int(row['written_frame']) for row in rows]!=list(range(nframes)):
        raise ValueError('Timestamp sidecar frame count/order mismatch')
    times=np.array([float(row['relative_observation_time_s']) for row in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Observation timestamps must increase')
    return times,'host_observation_monotonic_not_radar_time',sha256(path)


def write_results(data,p,args,metadata,calibration_status,cfg_hash,bias):
    times,time_source,time_hash=observation_time_axis(args.bin,metadata,len(data['times']),p.frame_s)
    data['times']=times;data['timestamp_source']=time_source
    for rows in (data['records'],data['rejected'],data['frame_counts']):
        for row in rows:row['time_s']=float(times[row['frame']])
    warnings=[
        'Audited copy: edge CFAR uses actual available training cells; nominal PFA remains approximate.',
        'RF CFG supplied separately; recording JSON does not bind actual slope/sample-rate to this file.',
        'Software decoding matches RadarStream and TI nominal layout; native capture implementation and hardware ordering are not independently certified.',
        'No packet sequence counters or per-frame timestamps. Queue-drop count does not certify loss-free UDP capture.',
        'No target identity. CFAR/angle estimates are candidate detections, not accurate mapping or object recognition.',
        'ISK geometry only, not ODS/AOP. Spatial phase sign, board orientation and calibration require controlled target validation.',
        'Velocity sign is a selected convention. TDM compensation assumes unaliased measured Doppler; fast-target ambiguity is unresolved.',
        'Selected branch uses only signed FFT half-band; out-of-band targets may alias or appear on the other side. No automatic unwrapping.',
        'Separated local angle peaks per range-Doppler cell are candidates; sidelobes, broad elevation response and unresolved reflectors can still create ambiguous points.',
        'ADC mean removal changes near-zero response. Static suppression removes stationary targets when enabled.',
        'Static angle heatmaps use zero-Doppler snapshots; they do not represent every moving target.',
        'Nominal CA-CFAR PFA is not a calibrated false-alarm rate for channel-averaged/windowed power.',
        'XYZ accumulation is not a registered map: no tracking, IMU pose compensation or ROS integration is applied.']
    spike_ratio=float(data['sample_power'].max()/max(np.median(data['sample_power']),1e-30))
    if spike_ratio>10:
        warnings.append(f'Concentrated fast-time spike: sample {int(data["sample_power"].argmax())}, mean power {spike_ratio:.1f}x sample median. No automatic repair; inspect capture/hardware before trusting XYZ.')
    for key in ('queue_dropped_frames','invalid_frames'):
        if metadata.get(key,0): warnings.append(f'Recorder reported {key}={metadata[key]}')
    report=dict(binary=str(args.bin.resolve()),binary_sha256=sha256(args.bin),cfg=str(args.cfg.resolve()),cfg_sha256=cfg_hash,
                profile=asdict(p),metadata=metadata,decoded_shape=[len(data['times']),p.loops,p.samples,12],
                virtual_channels=[dict(index=4*t+r,tx=t,rx=r,x_m=float(isk_positions()[4*t+r,0]),z_m=float(isk_positions()[4*t+r,1])) for t in range(3) for r in range(4)],
                range_bin_spacing_m=p.range_step_m,doppler_bin_hz=1/(p.loops*p.same_tx_s),same_tx_interval_s=p.same_tx_s,
                velocity_bin_m_s=p.wavelength_m/(2*p.loops*p.same_tx_s),sampled_bandwidth_hz=p.slope_hz_s*p.samples/p.fs_hz,
                timestamp_source=time_source,timestamp_sidecar_sha256=time_hash,range_side=args.range_side,spatial_phase_sign=args.phase_sign,velocity_sign=args.velocity_sign,calibration=calibration_status,
                azimuth_limit_deg=args.azimuth_limit,elevation_limit_deg=args.elevation_limit,mount_roll_deg=args.mount_roll_deg,
                intensity_definition='Mean per-frame Bartlett power at zero Doppler, indexed by range, elevation, azimuth',
                intensity_slice_ranges_m=data['ranges'][intensity_slice_indices(data,args)].tolist(),
                intensity_reference_power=float(args.intensity_reference_power if args.intensity_reference_power is not None
                    else data['intensity'][intensity_slice_indices(data,args)].max()),
                angle_model_audit=data['angle_model_audit'],rejected_angle_peaks=len(data['rejected']),
                applied_range_bias_m=bias,channel_rms=data['channel_rms'].tolist(),channel_phase_coherence=data['phase_coherence'].tolist(),
                rail_fraction=data['rail_fraction'],zero_fraction=data['zero_fraction'],duplicate_adjacent_frames=data['duplicate_adjacent_frames'],
                candidate_detections=len(data['records']),boundary_rejections=data['boundary_rejections'],include_boundary=args.include_boundary,reject_uncertain=getattr(args,'reject_uncertain',False),
                sample_power=data['sample_power'].tolist(),sample_peak_counts=data['sample_peak_counts'].tolist(),sample_spike_ratio=spike_ratio,pfa=args.pfa,min_snr_db=args.min_snr,min_coherence=args.min_coherence,
                detection_range_m=[args.min_range,args.max_range],max_detections_per_frame=args.max_detections,
                max_angle_peaks_per_cell=args.max_angle_peaks,angle_separation_deg=args.angle_separation,angle_peak_relative_limit_db=args.angle_peak_db,
                cfar_edge_policy=args.cfar_edge_policy,processing_mean_ms=float(np.mean(data['frame_processing_ms'])),processing_max_ms=float(np.max(data['frame_processing_ms'])),
                selection_totals={key:sum(row[key] for row in data['frame_counts']) for key in ('cells_total','threshold_pass_cells','edge_policy_pass_cells','nms_cells','angle_local_maxima','angle_within_relative_db','angle_after_separation','angle_after_peak_cap','cfar_cells','in_range_cells','angle_peaks','low_coherence','boundary_rejected','valid_points','cap_rejected','exported_points','uncertain_exported_points')},
                interpolated_sample_indices=getattr(args,'interpolate_samples',[]),nonfinite_samples=data['nonfinite_samples'],
                static_suppression=args.suppress_static,adc_mean_removed=not args.keep_adc_dc,validated=False,warnings=warnings,
                sources=[GEOMETRY_SOURCE,RAW_SOURCE])
    if args.output:
        (args.output/'diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (args.output/'input_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        (args.output/'input_configuration.cfg').write_text(args.cfg.read_text(encoding='utf-8-sig'),encoding='utf-8')
        columns=['frame','time_s','range_m','doppler_hz','radial_velocity_m_s','azimuth_deg','elevation_deg','x_m','y_m','z_m','sensor_x_m','sensor_y_m','sensor_z_m','mount_roll_deg','mount_pitch_deg','mount_yaw_deg','snr_db','signal_power_adc2','angular_coherence','angle_at_search_boundary','azimuth_at_boundary','elevation_at_boundary','outside_nominal_search','range_bin','doppler_bin','angle_peak_rank','validated','quality_flags','passes_angle_quality','retained_uncertain']
        with (args.output/'candidate_detections.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(data['records'])
        with (args.output/'rejected_angle_candidates.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns+['rejection_reason']);writer.writeheader();writer.writerows(data['rejected'])
        if args.audit_angle_model:
            (args.output/'angle_model_comparison.json').write_text(json.dumps(data['angle_model_audit'],indent=2),encoding='utf-8')
        with (args.output/'frame_point_counts.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data['frame_counts'][0]));writer.writeheader();writer.writerows(data['frame_counts'])
        np.savez_compressed(args.output/'analysis_arrays.npz',**{key:data[key] for key in ('ranges','signed_ranges','fd','velocity','times','rt','rd','rdc','dt','dtc','signed_original','signed_clean','ra','re','az','el','intensity')})
        np.savetxt(args.output/'channel_diagnostics.csv',np.c_[np.arange(12),data['channel_rms'],data['channel_mean'].real,data['channel_mean'].imag,data['phase_relative_deg'],data['phase_coherence']],delimiter=',',header='channel,rms_adc,mean_i,mean_q,relative_phase_deg,phase_coherence',comments='')
        if args.save_decoded: np.save(args.output/'decoded_adc.npy',decode_capture(np.memmap(args.bin,dtype='<i2',mode='r').reshape(-1,p.words),p).astype(np.complex64))
    if args.output:
        with (args.output/'cfar_range_audit.csv').open('w',newline='',encoding='utf-8') as stream:
            columns=['frame','range_m','in_requested_range','threshold_pass_cells','nms_cells','training_cells','legacy_edge_excluded']
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(data['cfar_range_audit'])
        _write_stage_audit(data,args)
    print(json.dumps({k:report[k] for k in ('decoded_shape','range_bin_spacing_m','same_tx_interval_s','doppler_bin_hz','velocity_bin_m_s','candidate_detections','boundary_rejections','sample_spike_ratio','calibration','validated')},indent=2),flush=True)
    print('CSV saved:',(args.output/'candidate_detections.csv').resolve(),flush=True)
    print('Frame counts:',(args.output/'frame_point_counts.csv').resolve(),flush=True)
    print('RF profile, beat/spatial signs and XYZ require physical validation.',flush=True)
    for warning in warnings:
        if warning.startswith('Concentrated'): print('WARNING:',warning,flush=True)
    return report


def _write_stage_audit(data,args):
    """Separate range-Doppler cells from angular peaks; a peak can branch."""
    import matplotlib.pyplot as plt
    stages=[('Range-Doppler cells','cells_total'),('Above CFAR/SNR','threshold_pass_cells'),
            ('After edge policy','edge_policy_pass_cells'),('After local RD maxima','nms_cells'),
            ('Inside requested range','in_range_cells'),('Local angular maxima','angle_local_maxima'),
            ('Within angular dB window','angle_within_relative_db'),('After angle separation','angle_after_separation'),
            ('After angular peak cap','angle_after_peak_cap'),('Exported XYZ','exported_points'),
            ('Uncertain exported XYZ','uncertain_exported_points')]
    rows=[dict(stage=label,total=sum(row[key] for row in data['frame_counts']),unit='RD cells' if i<5 else 'angle peaks / XYZ') for i,(label,key) in enumerate(stages)]
    with (args.output/'stage_totals.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['stage','total','unit']);writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,subset,title in [(axes[0],rows[:5],'Range-Doppler cells'),(axes[1],rows[5:],'Angular candidates (can branch per cell)')]:
        labels=[row['stage'] for row in subset];values=[row['total'] for row in subset]
        ax.barh(labels,values);ax.invert_yaxis();ax.set_title(title)
        ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Count, logarithmic scale above 1')
        for i,v in enumerate(values): ax.text(v,i,f' {v:,}',va='center',fontsize=8)
    fig.suptitle(args.bin.stem+' | '+args.cfar_edge_policy+' edge policy')
    fig.tight_layout();fig.savefig(args.output/'processing_stage_counts.png',dpi=160);plt.close(fig)


def build_parser():
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin',type=Path,required=True)
    ap.add_argument('--json',type=Path)
    ap.add_argument('--cfg',type=Path,default=(root/'radar_configs/iwr6843_generated.cfg' if (root/'radar_configs/iwr6843_generated.cfg').is_file() else root/'iwr6843_generated.cfg'))
    ap.add_argument('--range-side',choices=['signed','negative','positive'],default='signed',help='Signed is diagnostic only; explicitly select a branch for candidate XYZ')
    ap.add_argument('--azimuth-limit',type=float,default=60,help='Sensor azimuth search half-width; wider than 60 is diagnostic only')
    ap.add_argument('--elevation-limit',type=float,default=20,help='Sensor elevation search half-width; wider than 20 is diagnostic only')
    ap.add_argument('--mount-roll-deg',type=float,default=0,help='Explicit sensor-to-mount rotation about +Y; +90 maps sensor +X to -Z, +Z to +X')
    ap.add_argument('--audit-angle-model',action='store_true',help='Compare phase signs and wider searches on identical CFAR snapshots without changing accepted points')
    ap.add_argument('--mount-pitch-deg',type=float,default=0,help='Measured rotation about analysis X, after roll')
    ap.add_argument('--mount-yaw-deg',type=float,default=0,help='Measured rotation about analysis Z, after pitch')
    ap.add_argument('--phase-sign',type=int,choices=[-1,1],default=1,help='Spatial steering sign; validate left/right/up/down')
    ap.add_argument('--velocity-sign',type=int,choices=[-1,1],default=1,help='Physical radial velocity convention, unvalidated by default')
    ap.add_argument('--include-boundary',action='store_true',help='With --reject-uncertain, still retain boundary estimates')
    ap.add_argument('--reject-uncertain',action='store_true',help='Optional strict angle filter; default exports low-coherence and boundary candidates with flags')
    ap.add_argument('--calibration',type=Path,help='Explicit 12-channel complex correction JSON')
    ap.add_argument('--calibrate-at',type=float,help='Write phase calibration for an isolated stationary boresight target at this range')
    ap.add_argument('--keep-adc-dc',action='store_true')
    ap.add_argument('--suppress-static',action='store_true',help='Use clutter-suppressed detections; stationary targets disappear')
    ap.add_argument('--min-range',type=float,default=.20)
    ap.add_argument('--max-range',type=float,default=1.45)
    ap.add_argument('--cfar-edge-policy',choices=['truncate','legacy'],default='truncate',help='Count valid edge training cells; legacy reproduces excluded near/far bins')
    ap.add_argument('--pfa',type=float,default=.001)
    ap.add_argument('--min-snr',type=float,default=8)
    ap.add_argument('--min-coherence',type=float,default=.5)
    ap.add_argument('--max-detections',type=int,default=0,help='Maximum accepted XYZ points per frame; 0 keeps all qualifying points')
    ap.add_argument('--max-angle-peaks',type=int,default=3,help='Maximum separated local angle peaks per CFAR cell')
    ap.add_argument('--angle-separation',type=float,default=20,help='Minimum angle between exported peaks in one cell, degrees')
    ap.add_argument('--angle-peak-db',type=float,default=3,help='Keep local angle peaks within this many dB of the strongest')
    ap.add_argument('--output',type=Path,help='Custom results folder; also saves plots instead of opening them')
    ap.add_argument('--no-show',action='store_true',help='Save plots to the automatic results folder without opening windows')
    ap.add_argument('--intensity-range', nargs=2, type=float, metavar=('MIN_M', 'MAX_M'),
                    help='Distance interval for intensity slices (default: detection interval)')
    ap.add_argument('--intensity-slices', type=int, default=9, help='Number of native range slices to display')
    ap.add_argument('--intensity-reference-power', type=float,
                    help='Fixed linear-power reference to compare intensity figures')
    ap.add_argument('--interpolate-samples', nargs='+', type=int, default=[],
                    help='Explicit diagnostic sample indices to interpolate; records and original files are preserved')
    ap.add_argument('--save-decoded',action='store_true',help='With --output, also save full complex64 cube')
    return ap


def main():
    root = Path(__file__).resolve().parent
    ap = build_parser()
    args=ap.parse_args()
    if not np.isfinite([args.min_range,args.max_range,args.pfa,args.min_snr,args.min_coherence,args.angle_separation,args.angle_peak_db,args.azimuth_limit,args.elevation_limit,args.mount_roll_deg,args.mount_pitch_deg,args.mount_yaw_deg]).all() or not 0<=args.min_range<args.max_range or not 0<args.pfa<1 or not 0<=args.min_coherence<=1 or args.max_detections<0 or args.max_angle_peaks<1 or not 0<args.angle_separation<=180 or args.angle_peak_db<0:
        ap.error('Invalid range, threshold or detection-limit arguments')
    if not 2<=args.azimuth_limit<=90 or not 2<=args.elevation_limit<=90 or args.azimuth_limit%2 or args.elevation_limit%2:
        ap.error('Search limits must be even degree values between 2 and 90')
    if not 1 <= args.intensity_slices <= 30:
        ap.error('--intensity-slices must be between 1 and 30')
    if args.intensity_range is not None:
        low, high = args.intensity_range
        if not np.isfinite([low, high]).all() or not 0 <= low < high:
            ap.error('--intensity-range requires finite 0 <= MIN_M < MAX_M')
    if args.intensity_reference_power is not None and (
        not np.isfinite(args.intensity_reference_power) or args.intensity_reference_power <= 0
    ):
        ap.error('--intensity-reference-power must be finite and positive')
    if args.interpolate_samples:
        profile_for_indices = load_profile(args.cfg)
        if any(index <= 0 or index >= profile_for_indices.samples-1 for index in args.interpolate_samples):
            ap.error('Interpolated sample indices must exclude the first and last ADC samples')
    args.show_plots=args.output is None and not args.no_show
    if args.output is None:
        args.output=root/'multi_antenna_results'/args.bin.stem/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    p=load_profile(args.cfg);metadata,raw=validate_recording(args.bin,args.json or args.bin.with_suffix('.json'),p)
    cfg_hash=sha256(args.cfg);gains,bias,calibration_status=load_calibration(args.calibration,cfg_hash,args.phase_sign)
    if args.output: args.output.mkdir(parents=True,exist_ok=True)
    if args.calibrate_at is not None:
        if not args.output or args.range_side=='signed' or not np.isfinite(args.calibrate_at) or not 0<args.calibrate_at<p.fs_hz*C/(4*p.slope_hz_s):
            ap.error('--calibrate-at needs --output, a selected branch and a target inside the signed half-band')
        result=build_phase_calibration(raw,p,args.calibrate_at,args.range_side,cfg_hash,args.phase_sign)
        target=args.output/'phase_calibration.json';target.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Candidate boresight calibration:',target);return
    data=analyze(raw,p,args,gains,bias)
    write_results(data,p,args,metadata,calibration_status,cfg_hash,bias)
    if args.range_side=='signed': print('Diagnostic mode: XYZ CSV is empty. Use --range-side negative or positive only after checking known targets.',flush=True)
    make_plots(data,p,args)


if __name__ == '__main__':
    import sys
    batch_flags = {'--manifest', '--empty', '--batch', '--show', '--associate'}
    use_batch = any(flag in sys.argv for flag in batch_flags)
    if '--bin' in sys.argv:
        position = sys.argv.index('--bin') + 1
        paths = []
        while position < len(sys.argv) and not sys.argv[position].startswith('--'):
            paths.append(sys.argv[position])
            position += 1
        use_batch = use_batch or len(paths) > 1
    if use_batch:
        from analyze_radar_3d import main as batch_main
        batch_main()
    else:
        main()

