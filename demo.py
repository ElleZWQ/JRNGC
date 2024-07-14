import torch

from tgc.model import JRNGC
from tgc.data import lorenz_96, var_stable,fmri_net_sim, dream3_trajectories,dream4_trajectories
from tgc.metrics import two_classify_metrics, remove_self_connection
from tgc.runcase import batch_trainer, case_fix, print_case_list


import numpy as np
import scipy.io as scio
import time
import yaml
import argparse
import os
import json
from model_params import count_num_param


parser = argparse.ArgumentParser()
parser.add_argument("--yaml_dir",type=str, default="./F_var.yaml", help="yaml path")
parser.add_argument("--data_type",type=str,default='var',choices=['var','lorenz','dream3','fmri'])
parser.add_argument("--gpu",type=int,default=0)

parser.add_argument("--var_t",type=int,default=500)
parser.add_argument("--var_t_eval",type=int,default=100)

parser.add_argument("--f_subject",type=int,default=0)
parser.add_argument("--f_t",type=int,default=200)
parser.add_argument("--f_t_eval",type=int,default=0)

# parser.add_argument("--f_lorenz",type=int,default=10)
parser.add_argument("--lorenz_t",type=int,default=500)
parser.add_argument("--lorenz_t_eval",type=int,default=100)

parser.add_argument("--dream3_subject",type=int,default=3)

parser.add_argument('--model_type',type=str,default='JRNGC',choices=['JRNGC'])
parser.add_argument('--current_start',type=int,default=0,help='if the running is interupt')

parser.add_argument("--jaco_param",type=float,help='for crossing the jaco param')
parser.add_argument("--jrngc_relu",action="store_true")













args = parser.parse_args()
device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
using_yaml_file = args.yaml_dir
with open(using_yaml_file, 'r') as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

data_dir = "./data/{}/".format(args.data_type)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

COUNT = 0
seed = 0


def get_model(dir):
    data_parameters = np.load(dir + 'data_parameters.npy', allow_pickle=True).item()
    model_parameters = np.load(dir + 'model_parameters.npy', allow_pickle=True).item()
    print(data_parameters)
    print(model_parameters)

    return data_parameters, torch.load(dir+'model.pt')

if '__main__' == __name__:
    tot_perf_lag={}
    tot_perf_no_lag = {}
    
    num_nodes = params['d']
    
    trials = params['trials']
    print_params=False

    for trial in range(args.current_start,trials):
        cur_seed=trial+seed
        print('current seed:{}'.format(cur_seed))
        COUNT=cur_seed
        if not COUNT:
            print_params = True
        if args.data_type == 'var':
            data_name = 'num_nodes_{}/true_lag_{}/noise_scale_{}/seed_{}/'.format(params['d'],params['true_lag'],params['noise_scale'],cur_seed)
            
        elif args.data_type == 'lorenz':
            data_name = 'num_nodes_{}/F_{}/seed_{}/'.format(params['d'],params['f_lorenz'],cur_seed)
        elif args.data_type == 'fmri':
            data_name = 'num_nodes_{}/subject_{}/seed_{}/'.format(params['d'],args.f_subject,cur_seed)
        elif args.data_type == 'dream3':
            dream3_data_list = ['EColi1','EColi2','Yeast1', 'Yeast2', 'Yeast3']
            dream3_using_data = dream3_data_list[args.dream3_subject]
            data_name = 'num_nodes_{}/{}/seed_{}/'.format(params['d'],dream3_using_data,cur_seed)

        if not os.path.exists(data_dir+data_name):
                os.makedirs(data_dir+data_name)

        try:
            x = np.load(data_dir+data_name+'_x.npy')
            x_eval = np.load(data_dir+data_name+'_x_eval.npy')
            gc = np.load(data_dir+data_name+'_gc.npy')
            print("data loaded...")
        except:
            
            if args.data_type == 'var':
                print("generating data from scratch...")
                x, x_eval, gc = var_stable(d=num_nodes, t=args.var_t, t_eval=args.var_t_eval, lag=params['true_lag'], sd=params['noise_scale'],seed=cur_seed)
                np.save(data_dir+data_name+'_x.npy',x)
                np.save(data_dir+data_name+'_x_eval.npy',x_eval)
                np.save(data_dir+data_name+'_gc.npy',gc)
            elif args.data_type == 'fmri':
                print('-----fmri----')
                x, x_eval, gc = fmri_net_sim(d=num_nodes, subject=args.f_subject, t=args.f_t, t_eval=args.f_t_eval)
            elif args.data_type == 'lorenz':
                print("generating data from scratch...")
                x, x_eval, gc = lorenz_96(d=num_nodes, t=args.lorenz_t, t_eval=args.lorenz_t_eval, f=params['f_lorenz'], seed=cur_seed)
                np.save(data_dir+data_name+'_x.npy',x)
                np.save(data_dir+data_name+'_x_eval.npy',x_eval)
                np.save(data_dir+data_name+'_gc.npy',gc)
            elif args.data_type == 'dream3':
                print('dream3-------')
                x, x_eval, gc = dream3_trajectories(d=num_nodes, subject=args.dream3_subject)
            
            


        
        

        tm = time.time()
        


        if args.jaco_param:
            model, it, a, b, c,best_loss_r, mean_pred_loss_r, eval_loss_r = JRNGC.from_train(max_iter=params['max_iter'], d=params['d'], lag=params['lag'], layers=params['layers'], hidden=params['hidden'], dropout=params['dropout'],  jacobian_lam=args.jaco_param,struct_loss_choice=params['struct_loss_choice'],JFn =params['JFn'], x=x, x_eval=x_eval, lr=params['lr'], seed=cur_seed, device=device, verbose=params['verbose'],relu = args.jrngc_relu)
        else:
            model, it, a, b, c,best_loss_r, mean_pred_loss_r, eval_loss_r = JRNGC.from_train(max_iter=params['max_iter'], d=params['d'], lag=params['lag'], layers=params['layers'], hidden=params['hidden'], dropout=params['dropout'], jacobian_lam=params['jacobian_lam'],struct_loss_choice=params['struct_loss_choice'],JFn =params['JFn'], x=x, x_eval=x_eval, lr=params['lr'], seed=cur_seed, device=device, verbose=params['verbose'],relu=args.jrngc_relu)
        key_metric = ['causal with lag use jaco', 'causal no lag use jaco']
        end_time = time.time()
        if print_params:
            num_params = count_num_param(model)
            with open('./result/{}/model_params_result_{}_{}.txt'.format(args.model_type,args.data_type,num_nodes),'a') as f:
                
                f.write('it is running {} Nodes {}\n'.format(args.data_type,num_nodes))
                f.write(f'the seed is {seed}\n')
                if args.model_type=='JRNGC':
                    f.write('the JRNGC {}  params:{}\n'.format(params['struct_loss_choice'],num_params))
                
                
                f.write('It is using the yaml file:{}\n'.format(args.yaml_dir))
                f.write('========================\n')
       
        
        ret = model.get_gc_metrics(x, gc)
        
        ret = [ret[key_metric[0]], ret[key_metric[1]]]
        key_i = 0
        result_filename = './result/{}/data_{}/seed_{}.txt'.format(args.model_type,args.data_type,cur_seed)
        os.makedirs(os.path.dirname(result_filename), exist_ok=True)
        for it in ret:
            
            print()
            print(f"f1: {it['f1']:.3f}, f1_eps: {it['f1_eps']:.3f}")
            print(f"acc: {it['acc']:.3f}, acc_eps: {it['acc_eps']:.3f}")
            print(f"auroc: {it['auroc']:.3f}")
            print(f"auprc: {it['auprc']:.3f}")
            print()
            
            

            with open(result_filename, 'a') as f:
                f.write("----Tt is running seed : {}----------------------------\n".format(cur_seed))
                f.write("----Tt is running {}----------------------------\n".format(key_metric[key_i]))

                f.write('used_time:{:.6f}\n'.format(end_time - tm))
               
                if args.model_type == 'JRNGC':
                    f.write('the struct_loss is {}\n'.format(params['struct_loss_choice']))
                f.write('it is using the yaml file:{} \n'.format(using_yaml_file))
                f.write('f1: {:.3f}, f1_eps: {:.3f}\n'.format(it['f1'], it['f1_eps']))
                f.write('acc: {:.3f}, acc_eps: {:.3f}\n'.format(it['acc'], it['acc_eps']))
                f.write('auroc: {:.3f}\n'.format(it['auroc']))
                f.write('auprc: {:.3f}\n'.format(it['auprc']))
                if args.jaco_param:
                    f.write(f'jacabian lam {args.jaco_param}\n')
                f.write('\n')
                f.write('best_loss:{:.5e}, train_loss:{:.5e}, eval_loss:{:.5e}\n'.format(best_loss_r,mean_pred_loss_r,eval_loss_r))
                f.write('----------------------------\n')
            if key_i ==0 :
                for key,value in it.items():
                    if key in ['f1','acc','auroc','auprc']:
                        value = float(value)
                        if key not in tot_perf_lag:
                            tot_perf_lag[key]={"value":[],"mean":[],"std":[]}
                        tot_perf_lag[key]["value"].append(value)
            elif key_i == 1:
                for key,value in it.items():
                    if key in ['f1','acc','auroc','auprc']:
                        value = float(value)
                        if key not in tot_perf_no_lag:
                            tot_perf_no_lag[key]={"value":[],"mean":[],"std":[]}
                        tot_perf_no_lag[key]["value"].append(value)

            key_i += 1

            
    for key,value in tot_perf_lag.items():
       
        perf=np.array(value["value"])
        tot_perf_lag[key]['mean']=float(np.mean(perf))
        tot_perf_lag[key]['std']=float(np.std(perf))
    for key,value in tot_perf_no_lag.items():
       
        perf=np.array(value["value"])
        tot_perf_no_lag[key]['mean']=float(np.mean(perf))
        tot_perf_no_lag[key]['std']=float(np.std(perf))
    f_dir = './result/{}/data_{}/'.format(args.model_type,args.data_type)
    
    json_filename = f_dir+data_name[:-3]
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    yaml_filename = os.path.splitext(os.path.basename(args.yaml_dir))[0]

    with open(json_filename+"_"+key_metric[0]+"_"+yaml_filename+"_"+'yaml'+".json",'w') as f:    
        json.dump(tot_perf_lag,f)  
       
    with open(json_filename+"_"+key_metric[1]+"_"+yaml_filename+"_"+'yaml'+".json",'w') as f:    
        json.dump(tot_perf_no_lag,f)  
        
    
        
    
