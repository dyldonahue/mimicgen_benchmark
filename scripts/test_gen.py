import h5py
import numpy as np
import os

with h5py.File("./generated_data/screw_touch/demo/demo.hdf5", "r") as f:
    lengths = []
    for demo_id in f['data'].keys():
        demo_len = f[f'data/{demo_id}/actions'].shape[0]
        lengths.append(demo_len)
    
    print(f"Number of demos: {len(lengths)}")
    print(f"Mean demo length: {np.mean(lengths):.1f}")
    print(f"Max demo length: {np.max(lengths)}")
    print(f"Min demo length: {np.min(lengths)}")
    print(f"\nYour current max_steps: 400")
    print(f"Policy queries available: {400 / 8} = 50")