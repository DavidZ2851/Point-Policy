
import pickle as pkl
import numpy as np

with open('/home/haotian/Point-Policy/data/pick_place_red_mug_2/processed_data_pkl/expert_demos/franka_env/pick_place_red_mug.pkl', 'rb') as f:
    data = pkl.load(f)

print(f'Type: {type(data)}')
print()

if isinstance(data, dict):
    for k, v in data.items():
        if isinstance(v, (list, tuple)):
            print(f'{k}: {type(v).__name__}, len={len(v)}')
            if len(v) > 0:
                item = v[0]
                if isinstance(item, dict):
                    print(f'  [0] keys: {list(item.keys())}')
                    for kk, vv in item.items():
                        if hasattr(vv, 'shape'):
                            print(f'    {kk}: shape={vv.shape}, dtype={vv.dtype}')
                        elif isinstance(vv, (list, tuple)):
                            print(f'    {kk}: {type(vv).__name__}, len={len(vv)}')
                        else:
                            print(f'    {kk}: {type(vv).__name__} = {vv}')
                elif hasattr(item, 'shape'):
                    print(f'  [0]: shape={item.shape}, dtype={item.dtype}')
                else:
                    print(f'  [0]: {type(item).__name__}')
        elif hasattr(v, 'shape'):
            print(f'{k}: shape={v.shape}, dtype={v.dtype}')
        else:
            print(f'{k}: {type(v).__name__} = {v}')
