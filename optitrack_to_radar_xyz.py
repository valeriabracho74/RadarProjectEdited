"""Convert Motive global CSV poses without guessing physical pivot offsets.

Example: python optitrack_to_radar_xyz.py --csv Take1.csv --config calibration.json
    --take Far --output optitrack_converted_xyz.csv
Only numpy, pandas and scipy are required. Run --help for all options.
"""
from pathlib import Path
import argparse,csv,json
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def validate_transform(value, name):
    """Require a proper local-to-parent rigid transform; reject reflections."""
    if value is None:
        return None
    transform=np.asarray(value,dtype=float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all():
        raise ValueError(name+' must be a finite 4x4 matrix')
    if not np.allclose(transform[3],[0,0,0,1],atol=1e-8,rtol=0):
        raise ValueError(name+' has an invalid homogeneous last row')
    r=transform[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-6,rtol=0) or not np.isclose(np.linalg.det(r),1,atol=1e-6,rtol=0):
        raise ValueError(name+' rotation must be orthogonal and right-handed (det=+1)')
    return transform


def load_motive_csv(path, radar_body='RPi11', object_body='RigidBody'):
    """Identify each rigid body and its pose columns from actual Motive headers."""
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.reader(stream))
    metadata=dict(zip(rows[0][::2],rows[0][1::2]))
    if metadata.get('Length Units','').lower()!='meters' or metadata.get('Coordinate Space','').lower()!='global':
        raise ValueError('Expected global poses in meters; convert export settings explicitly')
    if metadata.get('Rotation Type')!='XYZ':
        raise ValueError('This implementation accepts Motive XYZ Euler exports, in degrees')
    header=next(i for i,row in enumerate(rows) if row[:2]==['Frame','Time (Seconds)'])
    names,groups,axes=rows[header-3],rows[header-1],rows[header]
    values=pd.DataFrame(rows[header+1:]).apply(pd.to_numeric,errors='coerce')
    result=pd.DataFrame({'frame':values[0],'time_s':values[1]})
    for role,body in [('radar',radar_body),('object',object_body)]:
        for kind,group in [('rotation','Rotation'),('position','Position')]:
            for axis in 'XYZ':
                cols=[j for j in range(len(axes)) if names[j]==body and groups[j]==group and axes[j]==axis]
                if len(cols)!=1:
                    raise ValueError(f'Expected exactly one {body} {group} {axis} column')
                result[f'{role}_{kind}_{axis.lower()}']=values[cols[0]]
    return metadata,result


def make_pose(position, euler_xyz_deg):
    """Motive documented column-vector convention: R = Rx * Ry * Rz."""
    transform=np.eye(4)
    transform[:3,:3]=Rotation.from_euler('XYZ',euler_xyz_deg,degrees=True).as_matrix()
    transform[:3,3]=position
    return transform


def convert(path, config, take):
    metadata,table=load_motive_csv(path,config.get('radar_body','RPi11'),config.get('object_body','RigidBody'))
    radar_offset=validate_transform(config.get('T_radar_markers_sensor'),'T_radar_markers_sensor')
    box_offset=validate_transform(config.get('T_box_markers_center'),'T_box_markers_center')
    known=radar_offset is not None and box_offset is not None
    start=datetime.strptime(metadata['Capture Start Time'],'%Y-%m-%d %I.%M.%S.%f %p')
    output=[]
    for row in table.itertuples(index=False):
        rp=np.array([getattr(row,'radar_position_'+a) for a in 'xyz'])
        op=np.array([getattr(row,'object_position_'+a) for a in 'xyz'])
        re=np.array([getattr(row,'radar_rotation_'+a) for a in 'xyz'])
        oe=np.array([getattr(row,'object_rotation_'+a) for a in 'xyz'])
        finite=np.isfinite(np.r_[rp,op,re,oe]).all()
        entry=dict(take=take,source_csv=str(Path(path).resolve()),frame=row.frame,time_s=row.time_s,
                   local_wall_time=(start+timedelta(seconds=row.time_s)).isoformat(),
                   finite_exported_pose=bool(finite),explicit_tracking_validity_available=False,
                   transform_applied=bool(known and finite),validated=False,
                   coordinate_status='configured_physical_transform_unvalidated' if known else 'unavailable_missing_physical_transforms')
        delta=op-rp
        for j,a in enumerate('xyz'):
            entry['radar_world_'+a+'_m']=rp[j];entry['object_world_'+a+'_m']=op[j]
            entry['pivot_relative_world_'+a+'_m']=delta[j]
            entry['pivot_relative_radar_marker_'+a+'_m']=np.nan
            entry['box_center_sensor_'+a+'_m']=np.nan
            entry['radar_euler_'+a+'_deg']=re[j];entry['object_euler_'+a+'_deg']=oe[j]
        if finite:
            world_radar=make_pose(rp,re);world_object=make_pose(op,oe)
            pivot_in_marker=world_radar[:3,:3].T@delta
            for j,a in enumerate('xyz'):entry['pivot_relative_radar_marker_'+a+'_m']=pivot_in_marker[j]
            if known:
                # Both offsets are local-to-parent matrices, including measured rotation.
                world_sensor=world_radar@radar_offset
                world_center=world_object@box_offset
                sensor_center=np.linalg.solve(world_sensor,world_center)
                for j,a in enumerate('xyz'):entry['box_center_sensor_'+a+'_m']=sensor_center[j,3]
        output.append(entry)
    return pd.DataFrame(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--take',required=True)
    parser.add_argument('--config',type=Path,help='JSON containing measured local-to-parent 4x4 transforms; null means unknown')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text()) if args.config else {}
    result=convert(args.csv,config,args.take)
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print(f'{len(result)} exported frames; {int(result.transform_applied.sum())} physical XYZ transforms applied.')
    if not result.transform_applied.any():print('Physical center XYZ unavailable; raw and radar-marker-local pivot coordinates were preserved.')


if __name__=='__main__':main()
