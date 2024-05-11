import os
import numpy as np

def print_case_list():
    """
   
    """
    data_ret, model_ret = {}, {}
    for root, dirs, files in os.walk('result'):
        if not dirs:
            data_parameters = np.load(root + '/data_parameters.npy', allow_pickle=True).item()
            data_name = data_parameters['data_name']
            if data_name in data_ret:
                cur = data_ret[data_name]
                for k, v in data_parameters.items():
                    cur[k].add(str(v))
            else:
                cur = {}
                data_ret[data_name] = cur
                for k, v in data_parameters.items():
                    cur[k] = set((str(v),))

            model_parameters = np.load(root + '/model_parameters.npy', allow_pickle=True).item()
            model_name = model_parameters['model_name']
            if model_name in model_ret:
                cur = model_ret[model_name]
                for k, v in model_parameters.items():
                    cur[k].add(str(v))
            else:
                cur = {}
                model_ret[model_name] = cur
                for k, v in model_parameters.items():
                    cur[k] = set((str(v),))
                    
    for data_name, it in data_ret.items():
        print('data ' + data_name)
        for k, v in it.items():
            print(f'{k}:\t{list(v)}')
        print()
    for model_name, it in model_ret.items():
        print('model ' + model_name)
        for k, v in it.items():
            print(f'{k}:\t{list(v)}')
        print()     


def case_fix(data_conditions, model_conditions):
    """
  
    """
    ret = []
    for root, dirs, files in os.walk('result'):
        if not dirs:
            ok = True
            data_parameters = np.load(root + '/data_parameters.npy', allow_pickle=True).item()
            for k, v in data_conditions.items():
                if data_parameters[k] != v:
                    ok = False
                    break
            model_parameters = np.load(root + '/model_parameters.npy', allow_pickle=True).item()
            for k, v in model_conditions.items():
                if model_parameters[k] != v:
                    ok = False
                    break
            if ok:
                ret.append(root)
    return ret
