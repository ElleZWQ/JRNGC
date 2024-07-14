import numpy as np
import scipy.io as scio
from ..runcase import data_decorator

@data_decorator
def fmri_net_sim(d, subject, t, t_eval):
    if d not in [15, 50]:
        raise ValueError(f"fMRI NetSim data only contains 15 or 50 demensions but need {d}")    
    if subject not in range(50):
        raise ValueError(f"a group of fMRI NetSim data only contains 50 subjects(0-49), but need {subject}")

    path = f'data/fMRI_{d}.mat'
    data = scio.loadmat(path)
    group_tm = data['Ntimepoints'][0][0]
    if (t + t_eval) > group_tm:
        raise ValueError(f"a group of this fMRI NetSim data only contains {group_tm} timepoints, but need {t}+{t_eval}")

    ts = data['ts'][group_tm * subject: group_tm * (subject + 1)]
    ts = np.swapaxes(ts, 0, 1).astype(np.float32)
    m = np.mean(ts, axis=1, keepdims=True)
    sd = np.std(ts, axis=1, keepdims=True)
    ts = (ts - m) / sd
    net = np.swapaxes(data['net'][subject], 0, 1)[:, :, np.newaxis]

    return ts[:, :t], ts[:, t:t+t_eval] if 0 != t_eval else ts[:, :t], (net != 0).astype(np.int32)

