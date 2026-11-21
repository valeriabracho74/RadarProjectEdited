"""Batch reports around RadarStream's verified multi-antenna processing.

See RADAR_3D_GUIDE.md. Original recordings are opened read-only. All XYZ
positions are candidates until compared with independently measured targets.
"""
from __future__ import annotations
import copy
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multi_antenna_analysis as core


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, Path): return str(value.resolve())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False), encoding='utf-8')


def validate_targets(table, experiment, args):
    """Associate a compact reflector or measured box; never label all points as it."""
    target = experiment.get('target')
    table['validated'] = False
    table['validation_status'] = 'unvalidated'
    table['target_associated'] = False
    columns = ['frame','detection_id','xyz_or_region_error_m','within_tolerance',
               'range_error_m','azimuth_error_deg','elevation_error_deg']
    if not target:
        return pd.DataFrame(columns=columns), {'status':'no_measured_ground_truth'}
    xyz = table[['x_m','y_m','z_m']].to_numpy(float)
    radius = float(target.get('association_radius_m', args.association_radius))
    tolerance = float(target.get('tolerance_m', args.gt_tolerance))
    if not np.isfinite([radius,tolerance]).all() or min(radius,tolerance)<=0:
        raise ValueError('Target association radius and tolerance must be positive')
    kind = target['kind']
    if kind == 'point':
        truth = np.asarray(target['xyz_m'],float)
        if truth.shape != (3,) or not np.isfinite(truth).all(): raise ValueError('Invalid measured XYZ')
        distance = cdist(xyz, truth[None,:]).ravel()
        chosen = []
        for _, group in table.groupby('frame'):
            index = min(group.index, key=lambda i: distance[i])
            if distance[index]<=radius: chosen.append(index)
    elif kind == 'region':
        bounds = np.asarray(target['bounds_m'],float)
        if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[0]>=bounds[1]):
            raise ValueError('bounds_m must be [minimum XYZ, maximum XYZ] in meters')
        delta = np.maximum(np.maximum(bounds[0]-xyz, xyz-bounds[1]),0)
        distance = np.linalg.norm(delta,axis=1)
        chosen = np.flatnonzero(distance<=radius).tolist()
    else: raise ValueError('Target kind must be point or region')
    rows = []
    for i in chosen:
        row = table.loc[i]; good = bool(distance[i]<=tolerance)
        table.loc[i,'target_associated'] = True
        table.loc[i,'validated'] = good
        table.loc[i,'validation_status'] = 'measured_target_within_tolerance' if good else 'measured_target_outside_tolerance'
        errors = [np.nan]*3
        if kind=='point':
            sensor = np.asarray(core.rotate_mount(truth,args.mount_roll_deg,getattr(args,'mount_pitch_deg',0),getattr(args,'mount_yaw_deg',0),inverse=True))
            r = np.linalg.norm(sensor)
            if r>0:
                az = np.degrees(np.arctan2(sensor[0],sensor[1]))
                el = np.degrees(np.arcsin(sensor[2]/r))
                errors = [row.range_m-r,(row.azimuth_deg-az+180)%360-180,row.elevation_deg-el]
        rows.append([int(row.frame),int(row.detection_id),distance[i],good,*errors])
    matches = pd.DataFrame(rows,columns=columns)
    summary = {'status':'compared_with_measured_'+kind,'associated_candidates':len(rows),
               'frames_with_association':int(table.loc[chosen,'frame'].nunique()),
               'association_is_a_hypothesis':True,'association_radius_m':radius,'tolerance_m':tolerance}
    if rows:
        summary.update(mean_error_m=matches.xyz_or_region_error_m.mean(),
                       median_error_m=matches.xyz_or_region_error_m.median(),
                       within_tolerance=int(matches.within_tolerance.sum()))
        summary['position_spread_std_m'] = table.loc[chosen,['x_m','y_m','z_m']].std(ddof=0).to_dict()
        for name in columns[-3:]: summary['mean_absolute_'+name] = matches[name].abs().mean()
    return matches,summary


def background_comparison(item, baseline, args):
    d,b = item['data'],baseline['data']
    for axis in ('ranges','az','el'):
        if not np.array_equal(d[axis],b[axis]): raise ValueError('Background axes differ: '+axis)
    if item['profile']!=baseline['profile']: raise ValueError('Background RF profiles differ')
    floor = max(float(b['intensity'].max()),float(d['intensity'].max()),1e-30)*1e-12
    change = 10*np.log10(np.maximum(d['intensity'],floor)/np.maximum(b['intensity'],floor))
    mask = d['intensity']>=max(float(d['intensity'].max()),1e-30)*10**(args.background_min_power_db/10)
    item['background_change'] = change
    item['background_mask'] = mask
    frame = item['table']
    for i,row in frame.iterrows():
        if abs(row.doppler_hz)>1/(item['profile'].loops*item['profile'].same_tx_s)/2:
            frame.loc[i,'background_status']='moving_cell_uncompared'; continue
        ri = int(np.argmin(abs(d['ranges']-row.range_m)))
        ai = int(np.argmin(abs(d['az']-row.azimuth_deg)))
        ei = int(np.argmin(abs(d['el']-row.elevation_deg)))
        gain = float(change[ri,ei,ai]); frame.loc[i,'background_change_db']=gain
        frame.loc[i,'background_status'] = ('increased_return_candidate' if gain>=args.background_db and mask[ri,ei,ai]
                                            else 'decreased_return' if gain<=-args.background_db else 'background_like')
    target,empty = d['rt'].mean(0),b['rt'].mean(0)
    eligible = (d['ranges']>=args.min_range)&(d['ranges']<=args.max_range)
    delta = np.where(eligible,target-empty,-np.inf)
    k = int(delta.argmax()); ratio = 10*np.log10(max(target[k],floor)/max(empty[k],floor))
    item['background'] = {'baseline_recording':baseline['name'],'range_profile_positive_change_peak_m':float(d['ranges'][k]) if delta[k]>0 else None,
                          'peak_change_db':float(ratio),'increased_candidate_count':int((frame.background_status=='increased_return_candidate').sum()),
                          'same_radar_pose_confirmed':bool(item['experiment'].get('same_radar_pose_as_baseline',False)),
                          'interpretation':'Power comparison only; occlusion, multipath and pose changes can cause differences. Not coherent subtraction or target validation.'}


def quality_report(data, metadata, profile):
    sample = data['sample_power']; ratio = float(sample.max()/max(np.median(sample),1e-30))
    elapsed = None
    # Recorder versions have used both names; preserve original metadata as well.
    for start,end in [('start_utc','end_utc'),('started_at','stopped_at'),('start_time','end_time'),('started_at_utc','stopped_at_utc')]:
        if metadata.get(start) and metadata.get(end):
            elapsed = (datetime.fromisoformat(metadata[end])-datetime.fromisoformat(metadata[start])).total_seconds(); break
    requested = metadata.get('requested_duration_seconds',metadata.get('requested_duration_s',metadata.get('duration_seconds',10)))
    duration = elapsed if elapsed is not None else float(requested)
    records = pd.DataFrame(data['records'])
    return {'nonfinite_complex_samples':data['nonfinite_samples'],'strongest_sample_index':int(sample.argmax()),
            'sample_max_to_median_power_ratio':ratio,'suspicious_sample_concentration':ratio>10,
            'adjacent_identical_frames':data['duplicate_adjacent_frames'],'recording_elapsed_s':elapsed,
            'duration_for_cadence_check_s':duration,'observed_frames':len(data['times']),
            'expected_frames_at_configured_cadence':duration/profile.frame_s,
            'frame_cadence_disagrees':abs(len(data['times'])*profile.frame_s-duration)>max(.2,.1*duration),
            'time_s_definition':data.get('timestamp_source','nominal_frame_period_unverified'),
            'most_common_azimuth_fraction':float(records.azimuth_deg.value_counts(normalize=True).max()) if len(records) else None,
            'note':'Concentration can indicate a real reflector, sidelobes, packing faults or phase calibration; this test cannot determine the cause.'}


def plot_recording(item, args, reference):
    d,t,out,p = item['data'],item['table'],item['out'],item['profile']
    def finish(fig,name):
        fig.suptitle(item['experiment'].get('label',item['name']),fontsize=12)
        fig.tight_layout(); fig.savefig(out/(name+'.png'),dpi=160); plt.close(fig)
    def range_mark(ax):
        target=item['experiment'].get('target',{})
        if target.get('kind')=='point': ax.axvline(np.linalg.norm(target['xyz_m']),color='red',ls='--',label='Measured target range'); ax.legend()
    fig,ax=plt.subplots(figsize=(9,4)); ax.plot(d['ranges'],core.relative_db(d['rt'].mean(0),reference['rt']))
    ax.set(xlabel='Range (m)',ylabel='Power relative to batch peak (dB)',xlim=(args.min_range,args.max_range),ylim=(-65,3)); range_mark(ax); ax.grid(alpha=.2); finish(fig,'range_profile')
    for key,y,ylabel,name in [('rd',d['velocity'],'Radial velocity (m/s)','range_doppler'),('rdc',d['velocity'],'Radial velocity (m/s)','range_doppler_static_suppressed'),('ra',d['az'],'Sensor azimuth (degrees)','range_azimuth'),('re',d['el'],'Sensor elevation (degrees)','range_elevation')]:
        values=d[key] if key.startswith('rd') else d[key].T
        if y[0]>y[-1]: y=y[::-1]; values=values[::-1]
        fig,ax=plt.subplots(figsize=(8,4.5))
        image=ax.pcolormesh(d['ranges'],y,core.relative_db(values,reference[key]),shading='nearest',cmap='viridis',vmin=-40,vmax=0)
        ax.set(xlabel='Range (m)',ylabel=ylabel,xlim=(args.min_range,args.max_range)); range_mark(ax)
        fig.colorbar(image,ax=ax,label='Power relative to batch peak (dB)'); finish(fig,name)
    fig,ax=plt.subplots(figsize=(6,6))
    if len(t):
        sc=ax.scatter(t.x_m,t.y_m,c=t.time_s,s=12,cmap='viridis'); fig.colorbar(sc,ax=ax,label='Host observation time (s)' if d.get('timestamp_source','').startswith('host_') else 'Nominal time (s)')
    else: ax.text(.5,.5,'No accepted XYZ candidates',ha='center',transform=ax.transAxes)
    target=item['experiment'].get('target',{})
    if target.get('kind')=='point': ax.scatter(*target['xyz_m'][:2],marker='x',color='red',label='Measured target'); ax.legend()
    if target.get('kind')=='region':
        from matplotlib.patches import Rectangle
        lo,hi=np.asarray(target['bounds_m']); ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],fill=False,edgecolor='red',label='Measured target region')); ax.legend()
    ax.set(xlabel='X right (m)',ylabel='Y forward (m)',xlim=(-args.max_range,args.max_range),ylim=(0,args.max_range)); ax.set_aspect('equal'); ax.grid(alpha=.2); finish(fig,'xy_detections')
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection='3d')
    if len(t):
        sc=ax.scatter(t.x_m,t.y_m,t.z_m,c=t.snr_db,s=10,vmin=reference['snr_min'],vmax=reference['snr_max']); fig.colorbar(sc,ax=ax,label='Local CFAR SNR (dB)',shrink=.6)
    else: ax.text2D(.25,.5,'No accepted XYZ candidates',transform=ax.transAxes)
    if target.get('kind')=='point': ax.scatter(*target['xyz_m'],marker='x',color='red')
    ax.set(xlabel='X right (m)',ylabel='Y forward (m)',zlabel='Z up (m)',xlim=(-args.max_range,args.max_range),ylim=(0,args.max_range),zlim=(-args.max_range,args.max_range)); finish(fig,'xyz_point_cloud')
    import plotly.graph_objects as go
    cloud=go.Figure()
    if len(t):
        cloud.add_trace(go.Scatter3d(x=t.x_m,y=t.y_m,z=t.z_m,mode='markers',marker=dict(size=3,color=t.snr_db,cmin=reference['snr_min'],cmax=reference['snr_max'],colorscale='Viridis',colorbar=dict(title='SNR dB')),
            text=[f'Frame {r.frame}; range {r.range_m:.3f} m; {r.validation_status}; {r.background_status}; flags: {r.quality_flags}' for r in t.itertuples()],hovertemplate='%{text}<br>X=%{x:.3f} Y=%{y:.3f} Z=%{z:.3f}<extra></extra>',name='Candidates'))
    else: cloud.add_annotation(text='No accepted XYZ candidates',showarrow=False)
    if target.get('kind')=='point':
        x,y,z=target['xyz_m']; cloud.add_trace(go.Scatter3d(x=[x],y=[y],z=[z],mode='markers',marker=dict(size=7,color='red',symbol='cross'),name='Measured target'))
    cloud.update_layout(title=item['experiment'].get('label',item['name'])+' â€” candidate XYZ, no tracking',scene=dict(xaxis=dict(title='X right (m)',range=[-args.max_range,args.max_range]),yaxis=dict(title='Y forward (m)',range=[0,args.max_range]),zaxis=dict(title='Z up (m)',range=[-args.max_range,args.max_range]),aspectmode='data'))
    cloud.write_html(out/'xyz_point_cloud.html',include_plotlyjs=True)
    fig,ax=plt.subplots(figsize=(9,4)); counts=pd.DataFrame(d['frame_counts']); ax.plot(counts.frame,counts.exported_points)
    ax.set(xlabel='Frame index',ylabel='Exported XYZ candidates (including uncertain)',ylim=(0,reference['count'])); ax.grid(alpha=.2); finish(fig,'detections_per_frame')
    fig,ax=plt.subplots(figsize=(9,4))
    if len(t): ax.scatter(t.frame,t.range_m,c=t.snr_db,s=10,vmin=reference['snr_min'],vmax=reference['snr_max'])
    ax.set(xlabel='Frame index',ylabel='Detection range (m)',ylim=(args.min_range,args.max_range)); finish(fig,'detection_range_vs_frame')
    fig,ax=plt.subplots(figsize=(8,4)); ax.hist(t.snr_db,bins=reference['snr_bins']); ax.set(xlabel='Signal / local CFAR training-cell mean (dB)',ylabel='Candidate count'); finish(fig,'snr_distribution')
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].plot(d['sample_power']); axes[0].set(xlabel='ADC sample index',ylabel='Mean ADC power (countsÂ²)'); axes[1].bar(np.arange(12),d['channel_rms']); axes[1].set(xlabel='TX-major / RX-minor channel',ylabel='RMS ADC counts'); finish(fig,'sample_channel_integrity')
    local=copy.copy(item['local']); local.output=out
    fig=core.plot_intensity_slices(d,local,reference['intensity']); fig.savefig(out/'radar_intensity_slices.png',dpi=160); plt.close(fig)
    if 'background_change' in item:
        elevation=d['intensity'].argmax(axis=1)
        change=np.take_along_axis(item['background_change'],elevation[:,None,:],axis=1)[:,0,:]
        mask=np.take_along_axis(item['background_mask'],elevation[:,None,:],axis=1)[:,0,:]
        fig,ax=plt.subplots(figsize=(9,5)); image=ax.pcolormesh(d['az'],d['ranges'],np.where(mask,change,np.nan),shading='nearest',cmap='RdBu_r',vmin=-8,vmax=8)
        ax.set(xlabel='Sensor azimuth (degrees)',ylabel='Range (m)',ylim=(args.min_range,args.max_range)); fig.colorbar(image,ax=ax,label='Object / baseline power (dB)'); finish(fig,'empty_scene_comparison')


def compare_batch(items, out, args, reference):
    out.mkdir(); rows=[]
    for item in items:
        table,d=item['table'],item['data']; counts=np.array([v['exported_points'] for v in d['frame_counts']])
        row=dict(recording=item['name'],label=item['experiment'].get('label',''),frames=len(counts),xyz_candidates=len(table),uncertain_xyz_candidates=int(table.retained_uncertain.sum()),
                 points_per_frame_min=counts.min(),points_per_frame_mean=counts.mean(),points_per_frame_max=counts.max(),
                 median_candidate_range_m=table.range_m.median(),median_candidate_x_m=table.x_m.median(),
                 target_associated_candidates=int(table.target_associated.sum()),validation_status=item['validation']['status'],
                 sample_spike_ratio=item['quality']['sample_max_to_median_power_ratio'],frame_cadence_disagrees=item['quality']['frame_cadence_disagrees'])
        row.update(item.get('background',{})); rows.append(row)
    summary=pd.DataFrame(rows); summary.to_csv(out/'recording_comparison.csv',index=False)
    trends=[]
    for item in items:
        relation=item['experiment'].get('expected_relation',{}); previous=relation.get('previous')
        if not previous: continue
        current=summary.loc[summary.recording==item['name']].iloc[0]
        earlier=summary.loc[summary.recording==previous]
        if earlier.empty: raise ValueError('Unknown previous recording '+previous)
        for axis,column in [('range','range_profile_positive_change_peak_m'),('x','median_candidate_x_m')]:
            expected=relation.get(axis)
            if not expected: continue
            change=current.get(column,np.nan)-earlier.iloc[0].get(column,np.nan)
            same_baseline=(current.get('baseline_recording')==earlier.iloc[0].get('baseline_recording'))
            resolved=np.isfinite(change) and abs(change)>item['profile'].range_step_m/2
            trends.append(dict(recording=item['name'],previous=previous,axis=axis,expected=expected,change_m=change,
                               outcome=('incomparable_baselines' if axis=='range' and not same_baseline else 'unresolved' if not resolved else 'consistent' if ((change<0)==(expected=='decrease')) else 'opposite'),
                               interpretation='Unassociated scene-level peak/median proxy; not target tracking or measured accuracy'))
    pd.DataFrame(trends,columns=['recording','previous','axis','expected','change_m','outcome','interpretation']).to_csv(out/'expected_direction_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5))
    for item in items: ax.plot(item['data']['ranges'],core.relative_db(item['data']['rt'].mean(0),reference['rt']),label=item['experiment'].get('label',item['name']))
    ax.set(xlabel='Range (m)',ylabel='Power relative to batch peak (dB)',xlim=(args.min_range,args.max_range),ylim=(-60,3)); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'range_profile_comparison.png',dpi=160); plt.close(fig)
    lines=['RadarStream multi-recording comparison','XYZ include uncertain angle candidates, marked in the CSV. They are not reconstructed object surfaces.']
    for row in rows: lines.append(f"{row['label']}: {row['frames']} frames; {row['xyz_candidates']} XYZ candidates ({row['uncertain_xyz_candidates']} uncertain);  {row['points_per_frame_mean']:.2f}/frame on average (min {row['points_per_frame_min']}, max {row['points_per_frame_max']}); {row['validation_status']}.")
    lines += ['Empty-scene differences can include occlusion and multipath. Target identity is unconfirmed.',
              'No distance accuracy can be reported without measured ground truth. time_s uses nominal 40 ms cadence.',
              'See metadata_summary.json for sample integrity, timing, calibration and configuration provenance.']
    (out/'presentation_summary.txt').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=core.build_parser()
    for action in parser._actions:
        if action.dest in ('bin','json'): action.nargs='+'
    parser.add_argument('--associate',action='store_true',help='Optional proximity links, not object identity or surface reconstruction')
    parser.add_argument('--batch',action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--empty',type=Path,help='Optional empty-scene recording; auto-included in batch')
    parser.add_argument('--show',action='store_true',help='Open the last interactive point cloud after saving')
    parser.add_argument('--association-radius',type=float,default=.2)
    parser.add_argument('--gt-tolerance',type=float,default=.15)
    parser.add_argument('--background-db',type=float,default=3)
    parser.add_argument('--background-min-power-db',type=float,default=-30)
    args=parser.parse_args()
    if args.range_side=='signed': parser.error('Batch XYZ needs --range-side negative or positive, verified with a known target')
    if args.calibrate_at is not None: parser.error('Create calibration with the single-recording multi_antenna_analysis.py CLI')
    numbers=[args.min_range,args.max_range,args.pfa,args.min_snr,args.min_coherence,args.association_radius,args.gt_tolerance,args.background_db,args.background_min_power_db,args.mount_roll_deg,args.mount_pitch_deg,args.mount_yaw_deg,args.angle_separation,args.angle_peak_db]
    if not np.isfinite(numbers).all() or not 0<=args.min_range<args.max_range or not 0<args.pfa<1 or not 0<=args.min_coherence<=1 or min(args.association_radius,args.gt_tolerance,args.background_db)<=0 or args.max_detections<0 or args.max_angle_peaks<1 or not 0<args.angle_separation<=180 or args.angle_peak_db<0:
        parser.error('Invalid range, detection or comparison settings')
    if not 1<=args.intensity_slices<=30: parser.error('intensity-slices must be 1..30')
    if not all(2<=v<=90 and v%2==0 for v in (args.azimuth_limit,args.elevation_limit)): parser.error('Angle limits must be even degrees from 2 to 90')
    if args.intensity_range and (not np.isfinite(args.intensity_range).all() or not 0<=args.intensity_range[0]<args.intensity_range[1]): parser.error('Invalid intensity range')
    if args.intensity_reference_power is not None and (not np.isfinite(args.intensity_reference_power) or args.intensity_reference_power<=0): parser.error('Intensity reference must be positive and finite')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig')) if args.manifest else {}
    experiments=manifest.get('recordings',{})
    if args.json and len(args.json)!=len(args.bin): parser.error('Supply exactly one --json per --bin, or let sidecars be found')
    paths=list(args.bin); metadata_paths=dict(zip([p.resolve() for p in paths],args.json or [p.with_suffix('.json') for p in paths]))
    if args.empty and args.empty.resolve() not in metadata_paths:
        paths.append(args.empty); metadata_paths[args.empty.resolve()]=args.empty.with_suffix('.json')
    if len({p.stem for p in paths})!=len(paths): parser.error('Input stems must be unique')
    session=(args.output or Path(__file__).resolve().parent/'radar_results')/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True); save_json(session/'processing_config.json',vars(args)); save_json(session/'experiment_manifest.json',manifest)
    items=[]
    for binary in paths:
        experiment=experiments.get(binary.stem,{})
        cfg=Path(experiment.get('cfg',args.cfg))
        if not cfg.is_absolute() and experiment.get('cfg') and args.manifest: cfg=args.manifest.parent/cfg
        p=core.load_profile(cfg)
        if any(i<=0 or i>=p.samples-1 for i in args.interpolate_samples): parser.error('Interpolated indices must be interior ADC samples')
        metadata_path=metadata_paths[binary.resolve()]; metadata,raw=core.validate_recording(binary,metadata_path,p)
        cfg_hash=core.sha256(cfg); gains,bias,status=core.load_calibration(args.calibration,cfg_hash,args.phase_sign)
        local=copy.copy(args); local.bin=binary; local.json=metadata_path; local.cfg=cfg; local.output=session/binary.stem; local.show_plots=False
        local.output.mkdir(); print('Analyzing',binary.name,flush=True)
        data=core.analyze(raw,p,local,gains,bias)
        report=core.write_results(data,p,local,metadata,status,cfg_hash,bias)
        table=pd.read_csv(local.output/'candidate_detections.csv'); table.insert(0,'recording',binary.stem)
        table['detection_id']=table.groupby('frame').cumcount(); table['background_status']='uncompared'; table['background_change_db']=np.nan
        from radar_dsp.mapping import analysis_to_ros, TemporalAssociator
        ros=analysis_to_ros(table[['x_m','y_m','z_m']].to_numpy().reshape(-1,3))
        for i,axis in enumerate(('x','y','z')):table['ros_'+axis+'_m']=ros[:,i]
        table['position_status']=np.where(table.passes_angle_quality,'angle_quality_pass','uncertain_candidate')
        table['temporally_associated']=False;table['temporal_track_id']=np.nan;table['track_observations']=0
        if args.associate:
            tracker=TemporalAssociator()
            for fi in range(len(data['times'])):
                selected=table.index[table.frame==fi];rows=table.loc[selected].to_dict('records')
                tracker.update(rows,float(data['times'][fi]))
                for index,row in zip(selected,rows):
                    for key in ('temporally_associated','temporal_track_id','track_observations'):table.loc[index,key]=row[key]
        matches,validation=validate_targets(table,experiment,local)
        table.loc[table.validated,'position_status']='ground_truth_compared_within_tolerance'
        table['timestamp_source']=data['timestamp_source'] 
        item=dict(name=binary.stem,data=data,table=table,profile=p,out=local.output,experiment=experiment,validation=validation,
                  quality=quality_report(data,metadata,p),report=report,metadata_path=metadata_path,local=local)
        matches.to_csv(local.output/'validation_matches.csv',index=False); pd.DataFrame([serial(validation)]).to_csv(local.output/'validation_summary.csv',index=False)
        items.append(item)
    lookup={item['name']:item for item in items}
    for item in items:
        baseline=item['experiment'].get('baseline',args.empty.stem if args.empty else None)
        if baseline and baseline!=item['name']:
            if baseline not in lookup: raise ValueError('Baseline must be included in --bin or --empty: '+baseline)
            background_comparison(item,lookup[baseline],args)
        item['table'].to_csv(item['out']/'candidate_detections.csv',index=False)
        save_json(item['out']/'metadata_summary.json',dict(report=item['report'],quality=item['quality'],experiment=item['experiment'],validation=item['validation'],background=item.get('background'),
                  processing=vars(item['local']),metadata_sha256=core.sha256(item['metadata_path']),
                  temporal_association_enabled=args.associate,temporal_association_definition='One-to-one distance links of angle-quality-passing candidates; repeated observations are not new surfaces',
                  software_versions=dict(numpy=np.__version__,pandas=pd.__version__),
                  source_code_sha256={name:core.sha256(Path(__file__).with_name(name)) for name in ('analyze_radar_3d.py','multi_antenna_analysis.py')},
                  snr_definition='10 log10(range-Doppler mean channel power / local CA-CFAR training-cell mean); not calibrated RF SNR',
                  coordinate_system='X right, Y forward, Z up; mount roll about Y; origin at antenna reference; physical signs unverified',
                  interpolation_applied=args.interpolate_samples))
        if 'background_change' in item: np.savez_compressed(item['out']/'background_arrays.npz',power_ratio_db=item['background_change'],display_mask=item['background_mask'])
    reference={key:max(float(item['data'][key].max()) for item in items) for key in ('rt','rd','rdc','ra','re','intensity')}
    if args.intensity_reference_power is not None: reference['intensity']=args.intensity_reference_power
    snr=np.concatenate([item['table'].snr_db.to_numpy() for item in items])
    reference['snr_min']=float(snr.min()) if snr.size else args.min_snr; reference['snr_max']=max(reference['snr_min']+1,float(snr.max()) if snr.size else args.min_snr+1)
    reference['snr_bins']=np.linspace(reference['snr_min'],reference['snr_max'],25)
    reference['count']=max(1,max(row['exported_points'] for item in items for row in item['data']['frame_counts']))*1.1
    for item in items: plot_recording(item,args,reference)
    compare_batch(items,session/'comparison',args,reference)
    print('Results saved:',session,flush=True)
    if args.show:
        import webbrowser
        webbrowser.open((items[-1]['out']/'xyz_point_cloud.html').resolve().as_uri())
    return session


if __name__=='__main__': main()
