"""Source-matched offline RadarStream analysis (numpy, matplotlib).
python ffts.py --bin recording.bin --cfg radar.cfg
Opens interactive graph windows by default; no files are saved.
Use --output FOLDER to save plots and numerical results instead.
Single TX/RX complex captures only. CFG is user supplied, not authenticated
against historical recording. Signed frequency is retained unless --side is set.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

C = 299792458.0


def decode(raw, nc, ns):
    """Exactly matches DataProcessor._decode_frame for one TX and RX."""
    p = np.asarray(raw).reshape(-1, 4).astype(np.float64)
    return (p[:, :2] + 1j*p[:, 2:]).reshape(nc, ns)


def profile(path, meta):
    commands = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        t = line.split('%')[0].split()
        if t:
            commands.setdefault(t[0], []).append(t)
    def one(k):
        if len(commands.get(k, [])) != 1:
            raise ValueError('Requires exactly one '+k)
        return commands[k][0]
    p, f, ch, adc, buf = [one(k) for k in ('profileCfg','frameCfg','channelCfg','adcCfg','adcbufCfg')]
    if commands.get('advFrameCfg'):
        raise ValueError('Advanced frames are unsupported')
    if int(ch[1]).bit_count()!=1 or int(ch[2]).bit_count()!=1:
        raise ValueError('Requires one enabled TX and RX')
    if int(adc[1])!=2 or int(adc[2])!=1 or list(map(int,buf[2:5])) != [0,1,1]:
        raise ValueError('Requires project complex 16-bit adcCfg/adcbufCfg convention')
    start,end,loops=map(int,f[1:4])
    if start!=end:
        raise ValueError('Only one chirp index per loop is supported')
    active=[t for t in commands.get('chirpCfg',[]) if int(t[1])<=start<=int(t[2])]
    if len(active)!=1 or int(active[0][3])!=int(p[1]) or int(active[0][8])!=int(ch[2]) or any(float(v)!=0 for v in active[0][4:8]):
        raise ValueError('Requires one matching chirpCfg without profile variations')
    ns=int(p[10]); fs=float(p[11])*1e3; slope=float(p[8])*1e12
    period=(float(p[3])+float(p[5]))*1e-6; fp=float(f[5])*1e-3
    if (ns,loops)!=(meta['adc_samples'],meta['chirps_per_frame']):
        raise ValueError('CFG sample/chirp dimensions disagree with JSON')
    if not all(np.isfinite(v) and v>0 for v in (fs,slope,period,fp)) or ns<4 or loops<4:
        raise ValueError('Invalid profile dimensions/timing')
    if loops*period>fp+1e-12 or float(p[4])*1e-6+ns/fs>float(p[5])*1e-6+1e-12:
        raise ValueError('Chirp burst or ADC window exceeds configured timing')
    return ns,loops,fs,slope,period,fp


def spectrum(z):
    w=np.hanning(z.shape[1])
    r=np.fft.fft(z*w,axis=1)/w.sum()
    clean=np.fft.fft((z-z.mean(axis=1,keepdims=True))*w,axis=1)/w.sum()
    return r,clean


def db(p,ref=None):
    return 10*np.log10(np.maximum(p,1e-30)/max(float(np.max(p) if ref is None else ref),1e-30))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bin',type=Path,required=True)
    ap.add_argument('--json',type=Path)
    ap.add_argument('--cfg',type=Path,required=True)
    ap.add_argument('--output',type=Path,help='Save results to this folder instead of opening graph windows')
    ap.add_argument('--side',choices=['signed','positive','negative'],default='signed',help='Never inferred from signal strength')
    ap.add_argument('--gate',nargs=2,type=float,default=[0.25,0.85],help='Absolute beat-equivalent range bounds, metres')
    a=ap.parse_args()
    if not np.isfinite(a.gate).all() or not 0<=a.gate[0]<a.gate[1]:
        raise ValueError('Gate must be finite, nonnegative and increasing')
    m=json.loads((a.json or a.bin.with_suffix('.json')).read_text(encoding='utf-8-sig'))
    for k in ('adc_samples','chirps_per_frame','tx_antennas','rx_antennas','int16_values_per_frame','bytes_per_frame','frames_written'):
        if type(m[k]) is not int or m[k]<=0: raise ValueError('Invalid metadata '+k)
    if m['tx_antennas']!=1 or m['rx_antennas']!=1 or m.get('dtype')!='<i2':
        raise ValueError('Requires single TX/RX little-endian int16 recording')
    ns,nc,fs,slope,tp,fp=profile(a.cfg,m)
    size=a.bin.stat().st_size; words=2*ns*nc
    if size==0 or size%(words*2) or m['int16_values_per_frame']!=words or m['bytes_per_frame']!=words*2 or m['frames_written']!=size//(words*2) or m.get('file_bytes')!=size:
        raise ValueError('Binary length / frame dimensions / JSON mismatch; no truncation allowed')
    if m.get('error'): raise ValueError('Recorder reported error: '+str(m['error']))
    nf=size//(words*2)
    raw=np.memmap(a.bin,dtype='<i2',mode='r',shape=(nf,words))
    freq=np.fft.fftshift(np.fft.fftfreq(ns,1/fs)); signed=C*freq/(2*slope)
    keep=np.ones(ns,dtype=bool) if a.side=='signed' else (signed>=0 if a.side=='positive' else signed<=0)
    idx=np.where(keep)[0]
    if a.side=='negative': idx=idx[::-1]
    ranges=signed[idx] if a.side=='signed' else np.abs(signed[idx])
    gate=(np.abs(ranges)>=a.gate[0])&(np.abs(ranges)<=a.gate[1])
    if not gate.any(): raise ValueError('Gate contains no FFT bins')
    rt=np.empty((nf,len(idx))); dt=np.empty((nf,nc)); dtc=np.empty_like(dt)
    rd=np.zeros((nc,len(idx))); rdc=np.zeros_like(rd)
    original=np.zeros(ns); corrected=np.zeros(ns); wd=np.hanning(nc)[:,None]
    for i,row in enumerate(raw):
        z=decode(row,nc,ns); r,clean=spectrum(z)
        original+=np.mean(abs(r)**2,axis=0)/nf
        corrected+=np.mean(abs(clean)**2,axis=0)/nf
        r=np.fft.fftshift(clean,axes=1)[:,idx]
        d=np.fft.fftshift(np.fft.fft(r*wd,axis=0),axes=0)/wd.sum()
        dc=np.fft.fftshift(np.fft.fft((r-r.mean(axis=0))*wd,axis=0),axes=0)/wd.sum()
        power,pc=abs(d)**2,abs(dc)**2
        rt[i]=np.mean(abs(r)**2,axis=0);dt[i]=power[:,gate].mean(axis=1);dtc[i]=pc[:,gate].mean(axis=1)
        rd+=power/nf;rdc+=pc/nf
    fd=np.fft.fftshift(np.fft.fftfreq(nc,tp));time=np.arange(nf)*fp
    out=a.output
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg' if out is not None else 'TkAgg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout()
        fig.canvas.manager.set_window_title(name.replace('_',' ').title())
        if out is not None:
            fig.savefig(out/(name+'.png'),dpi=160)
            plt.close(fig)
    note='Source-matched I0 I1 Q0 Q1; supplied CFG; beat sign uncalibrated'
    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.plot(signed,db(np.fft.fftshift(original)),label='Original')
    ax.plot(signed,db(np.fft.fftshift(corrected),original.max()),label='Per-chirp ADC mean removed')
    ax.set(xlabel='Signed beat-equivalent range (m)',ylabel='Power relative to original peak (dB)',title=note);ax.legend();ax.grid(alpha=.3);save(fig,'signed_range_diagnostic')
    fig,ax=plt.subplots(figsize=(11,4.5));ax.plot(ranges,db(rt.mean(axis=0)))
    ax.set(xlabel='Signed beat-equivalent range (m)' if a.side=='signed' else 'Beat-equivalent range (m)',ylabel='Relative power (dB)',title='Range FFT; ADC mean removed; stationary targets retained');ax.grid(alpha=.3);save(fig,'range_fft')
    def edges(v,step): return np.r_[v-step/2,v[-1]+step/2]
    re=edges(ranges,C*fs/(2*slope*ns));de=edges(fd,1/(nc*tp));te=np.arange(nf+1)*fp
    def heat(p,x,y,title,xlabel,ylabel,name,ref=None):
        fig,ax=plt.subplots(figsize=(11,5));im=ax.pcolormesh(x,y,db(p,ref),vmin=-70,vmax=0,shading='flat')
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);fig.colorbar(im,ax=ax,label='Relative power (dB)');save(fig,name)
    rlabel='Signed beat-equivalent range (m)' if a.side=='signed' else 'Beat-equivalent range (m)'
    heat(rt.T,te,re,'Range-Time; ADC mean removed','Nominal frame time (s)',rlabel,'range_time')
    for suffix,p,d in [('',rd,dt),('_clutter_suppressed',rdc,dtc)]:
        heat(p.T,de,re,'Range-Doppler; mean frame power'+suffix,'Signed Doppler frequency (Hz)',rlabel,'range_doppler'+suffix,rd.max())
        heat(d.T,te,de,f'Doppler-Time; absolute range gate {a.gate} m'+suffix,'Nominal frame time (s)','Signed Doppler frequency (Hz)','doppler_time'+suffix,dt.max())
    z=decode(raw[0],nc,ns);fig,ax=plt.subplots(figsize=(11,4));ax.plot(z[0].real,label='I');ax.plot(z[0].imag,label='Q');ax.legend();ax.set(xlabel='ADC sample',ylabel='Signed ADC counts',title='Frame 0 / chirp 0');save(fig,'adc_iq')
    warnings=['CFG supplied separately; matching dimensions do not authenticate historical RF settings.', 'Beat-frequency sign and physical motion direction require a controlled target experiment.', 'No packet sequence numbers or frame timestamps: upstream loss, buffer overwrites and jitter cannot be certified.', 'ADC mean removal attenuates targets close to DC; original spectrum retained for comparison.', 'Nominal frame timing includes inter-frame gap; Doppler FFT never crosses frame boundaries.']
    for k in ('queue_dropped_frames','invalid_frames'):
        if m.get(k,0): warnings.append(f'Recorder {k}={m[k]}; time continuity is uncertain.')
    shifted=np.fft.fftshift(corrected);rank=np.argsort(shifted)[::-1][:10]
    report=dict(binary=str(a.bin.resolve()),metadata=m,cfg=str(a.cfg.resolve()),cfg_sha256=hashlib.sha256(a.cfg.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(a.bin.read_bytes()).hexdigest(),frames=nf,range_bin_m=C*fs/(2*slope*ns),doppler_bin_hz=1/(nc*tp),chirp_period_s=tp,frame_period_s=fp,side=a.side,gate_bins_m=ranges[gate].tolist(),raw_min=int(raw.min()),raw_max=int(raw.max()),zero_fraction=float(np.mean(raw==0)),int16_rail_fraction=float(np.mean((raw==-32768)|(raw==32767))),top_bins=[dict(signed_beat_range_m=float(signed[j]),power=float(shifted[j])) for j in rank],warnings=warnings)
    print(f'Analyzed {nf} frames.', flush=True)
    if out is not None:
        (out/'diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        np.savez_compressed(out/'fft_results.npz',range_m=ranges,doppler_hz=fd,nominal_time_s=time,range_time=rt,range_doppler=rd,range_doppler_suppressed=rdc,doppler_time=dt,doppler_time_suppressed=dtc,signed_range_m=signed,original_power=np.fft.fftshift(original),adc_mean_removed_power=shifted)
        np.savetxt(out/'signed_spectrum.csv',np.c_[freq,signed,np.fft.fftshift(original),shifted],delimiter=',',header='beat_hz,signed_beat_range_m,original_power_adc_counts_squared,adc_mean_removed_power_adc_counts_squared',comments='')
        print('Saved:',out.resolve())
    else:
        print('Opening 8 graph windows. Close them to finish. No files saved.', flush=True)
        plt.show()



if __name__=='__main__':
    main()
