import functools
import contextlib
import hashlib
import json
import os
import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import torch
import torch.multiprocessing as mp

vis = set()
tasklist = []
is_batch_run = False
def _add_case(data_args:dict, model_args:dict):
    """
    
    """
    s_data = json.dumps(OrderedDict(data_args))
    s_model = json.dumps(OrderedDict(model_args))
    case_hash = hashlib.sha256((s_data + s_model).encode('utf-8')).hexdigest()
    dir = f'result/{data_args["data_name"]}_{model_args["model_name"]}/{case_hash}/'
    if not os.path.exists(dir) and not dir in vis:
        vis.add(dir)
        tasklist.append((dir, data_args, model_args))

@contextlib.contextmanager
def batch_trainer(processes:int):
    global is_batch_run, tasklist
    is_batch_run = True
    tasklist = []
    yield _add_case
    is_batch_run = False
    with mp.Pool(processes) as workers:
        with tqdm.tqdm(total=len(tasklist)) as pbar:
            for _ in workers.imap_unordered(_run_signle_case, tasklist):
                pbar.update()

def data_decorator(func):
    """
   
    """
    @functools.wraps(func)
    def generator(**kwargs):
       if is_batch_run:
           data_name = func.__name__
           kwargs['data_name'] = data_name
           return kwargs
       else:
           return func(**kwargs) 
    return generator

def model_decorator(cls):
    """
   
    """
    trainer = cls.from_train
    def new_trainer(**kwargs):
        if is_batch_run:
            model_name = cls.__name__
            kwargs['model_name'] = model_name
            return kwargs
        else:
            return trainer(**kwargs)
    cls.from_train = new_trainer
    return cls

# =====================================================

def _get_data(data_args):
    """
   
    """
    from .. import data
    name = data_args['data_name']
    data_args.pop('data_name')
    if name in data.__all__:
        generator = getattr(data, name)
        ret = generator(**data_args)
    else:
        raise ValueError("unkonw data generator " + name)
    data_args['data_name'] = name
    return ret

def _get_model(x, x_eval, model_args):
    """
    
    """
    from .. import model
    name = model_args['model_name']
    model_args.pop('model_name')
    if name in model.__all__:
        cls = getattr(model, name)
        ret = cls.from_train(x=x, x_eval=x_eval, **model_args)
    else:
        raise ValueError("unkonw model " + name)
    model_args['model_name'] = name
    return ret

def _run_signle_case(input):
    """
    
    """
    dir, data_args, model_args = input
    x, x_eval, gc = _get_data(data_args)
    model, it_list, tot_loss_list, exper_loss_list, eval_loss_list = _get_model(x, x_eval, model_args)
    try:
        ret = model.get_gc_metrics(x, gc)
    except Exception as e:
        print(f'"{str(e)}" on {dir} , please check')
        os.makedirs(dir)
        np.save(dir + 'data_parameters.npy', data_args)
        np.save(dir + 'model_parameters.npy', model_args)
        torch.save(model, dir+'model.pt')
        return

    os.makedirs(dir)
    np.save(dir + 'causal_result.npy', ret)
    np.save(dir + 'data_parameters.npy', data_args)
    np.save(dir + 'model_parameters.npy', model_args)
    ret = {'it_list': it_list, 'tot_loss_list': tot_loss_list, 'exper_loss_list':exper_loss_list, 'eval_loss_list':eval_loss_list}
    np.save(dir + 'loss.npy', ret)
    
    ret = {'eval MSE': eval_loss_list[-1]}
    np.save(dir + 'pred_result.npy', ret)
    torch.save(model, dir+'model.pt')
