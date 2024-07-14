import numpy as np
import pandas as pd
from ..runcase import data_decorator

@data_decorator
def dream3_trajectories(d, subject):
    if d not in [10, 50, 100]:
        raise ValueError(f"dream3 data only contains 10 or 50 demensions but need {d}")    
    if subject not in range(5):
        raise ValueError(f"a group of fMRI NetSim data only contains 5 subjects(0-4), but need {subject}")

    path_data = f'data/dream3_{d}_{subject}.tsv'
    path_gc = f'data/dream3_{d}_{subject}_true.txt'

    data = []
    with open(path_data) as f:
        lines = f.readlines()
        cur = []
        for line in lines[1:]:
            values = line.strip('\n').split('\t')
            curt = []
            if len(values) == d + 1:
                for v in values[1:]:
                    curt.append(v)
                cur.append(curt)
            else:
                data.append(cur)
                cur = []
    data = np.array(data, dtype=np.float32).swapaxes(1, 2)
    m = np.mean(data, axis=2, keepdims=True)
    sd = np.std(data, axis=2, keepdims=True)
    data = (data - m) / sd

    gc = np.zeros((d, d, 1), dtype=np.int32)
    for i in range(d):
        gc[i, i, 0] = 1
    with open(path_gc) as f:
        lines = f.readlines()
        for line in lines:
            i, j, w = line.split('\t')
            i, j = int(i[1:]) - 1, int(j[1:]) - 1
            gc[j, i] = int(w)
    
    return data, data, gc
