import torch
from torch import nn
import torch.nn.functional as F

from .model import TGCModel
from ..runcase import model_decorator
from ..metrics import two_classify_metrics, remove_self_connection
from torch.cuda import amp

import matplotlib.pyplot as plt
import numpy as np
import time
from ..metrics.jacobian import JacobianReg
from sklearn import metrics
from model_params import count_num_param


class ResidualBlock(nn.Module):
    def __init__(self, input, hidden, output, dropout):
        super(ResidualBlock, self).__init__()
        self.linear_1 = torch.nn.utils.weight_norm(nn.Linear(input, hidden))
        self.linear_2 = nn.Linear(hidden, output)
        self.linear_res = nn.Linear(input, output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output)

    def forward(self, x):
        """
        x: [Batch, hidden]
        """
        h = self.linear_1(x)
        h = self.relu(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        res = self.linear_res(x)
        out = h + res
        out = self.layernorm(out)
        return out
    
    def struct_loss(self):
        return torch.sum(self.linear_res.weight ** 2)

@model_decorator
class JRNGC(nn.Module):
    def __init__(self, d, lag, layers, hidden, dropout,jacobian_lam,struct_loss_choice,JFn,relu=False):
        super(JRNGC, self).__init__()

        self.d = d
        self.lag = lag
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.jacobian_lam = jacobian_lam
        self.struct_loss_choice = struct_loss_choice
        self.JFn = JFn
        self.relu = relu

        

        

        self.inputgate = nn.Linear(d * lag, hidden)
        self.outputgate = nn.Linear(hidden, d)
        self.inputgate = torch.nn.utils.weight_norm(self.inputgate)

        modules = [ResidualBlock(hidden, hidden, hidden, dropout) for _ in range(layers)]
        self.encoders = nn.ModuleList(modules)
        
    

    def forward(self, x):
        """
        x: [batch, d, T=lag]
        """ 
        x = x.flatten(start_dim=1).to(torch.float32)
        x = self.inputgate(x)
        if self.relu:
            x = F.relu(x)
           
            
        for net in self.encoders:
            x = net(x)
        x = self.outputgate(x)
        return x
    
    def jacobian_causal(self, x,flag=False):
        """
        x: [batch, d, T=lag]
        """
        with amp.autocast():
            if not flag:
                self.eval()
            x.requires_grad_(True)
            jac = torch.zeros((x.shape[0], x.shape[1], x.shape[1], x.shape[2]))
            for j in range(x.shape[1]):
                y = self(x)[:, j]
                y.backward(torch.ones_like(y))
                jac[:, j, :, :] = x.grad
                x.grad.zero_()
            jac = torch.mean(torch.abs(jac), dim=0)
        return jac

    def jacobian_causal_train(self, x):
        """
        x: [batch, d, T=lag]
        """
        with amp.autocast():
            x.requires_grad_(True)
            jac = torch.zeros((x.shape[0], x.shape[1], x.shape[1], x.shape[2]))
            # start_time = time.time()
            for j in range(x.shape[1]):
                y = self(x)[:, j]
                jac[:, j, :, :] = torch.autograd.grad(y,x,create_graph=True,grad_outputs=torch.ones_like(y))[0]
            # end_time = time.time()
            # print('time:',end_time-start_time)
            jac = torch.mean(torch.abs(jac), dim=0)
        return jac
    
    
    
    def compute_jacobian_F_loss(self,x):
        Jacobian_Reg = JacobianReg(n=self.JFn)

        if 2 == len(x.shape): x.unsqueeze_(0)
        x = x.transpose(1, 2).unfold(1, self.lag, 1)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x.requires_grad_(True)
        y = self(x)
        JAC_loss = self.jacobian_lam*Jacobian_Reg(x,y)
        return JAC_loss
   
    def jacobian_causal_L1_loss(self, x):
        # x = torch.tensor(x, device=next(self.parameters()).device)
        if 2 == len(x.shape): x.unsqueeze_(0)
        x = x.transpose(1, 2).unfold(1, self.lag, 1)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        jac = self.jacobian_causal_train(x)
        jac_loss = torch.sum(jac)*self.jacobian_lam

        return jac_loss
        
    
    def exper_loss(self, x):
        """
        x: [batch, d, T=lag+1]
        """
        return self.loss_fn(self(x[:, :, :-1]), x[:, :, -1])
    
 

    # @staticmethod
    def from_train(d, lag, layers, hidden, dropout,jacobian_lam,struct_loss_choice,JFn, x, x_eval, lr, seed, device, min_iter=1000, max_iter=10000, lookback=10, check_first=50, check_every=100, verbose=False,relu=False):
        """
        x: [d, t] or [batch, d, t]
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        num_nodes = x.shape[0]

        x = torch.tensor(x, device=device).to(torch.float16)
        x_eval = torch.tensor(x_eval, device=device).to(torch.float16)
        if 2 == len(x.shape): x.unsqueeze_(0)
        if 2 == len(x_eval.shape): x_eval.unsqueeze_(0)
        x = x.transpose(1, 2).unfold(1, lag + 1, 1)
        x_eval = x_eval.transpose(1, 2).unfold(1, lag + 1, 1)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x_eval = x_eval.reshape(x_eval.shape[0] * x_eval.shape[1], x_eval.shape[2], x_eval.shape[3])
        model = JRNGC(d, lag, layers, hidden, dropout, jacobian_lam, struct_loss_choice, JFn).to(device)
       
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = amp.GradScaler()

        best_it = None
        best_loss = torch.inf
        it_list, tot_loss_list, exper_loss_list, eval_loss_list = [], [], [], []
        
        for it in range(max_iter):
            # train
            with amp.autocast():
                model.train()
                pred_loss = model.exper_loss(x)
                if model.struct_loss_choice == 'JL1':
                    struct_loss = model.jacobian_causal_L1_loss(x)
                
                elif model.struct_loss_choice == 'JF':
                    struct_loss = model.compute_jacobian_F_loss(x)
                
                else:
                    struct_loss = 0
                loss = pred_loss + struct_loss
                

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
         
                optimizer.zero_grad()

            if (it < min_iter and 0 == (it + 1) % check_first) or (0 == (it + 1) % check_every):
                it_list.append(it)
                mean_pred_loss = pred_loss / d
                mean_loss = loss / d
                exper_loss_list.append(mean_pred_loss.detach().item())
                tot_loss_list.append(mean_loss.detach().item())
                # eval
                with torch.no_grad():
                    model.eval()
                    eval_loss = model.exper_loss(x_eval) / d
                eval_loss_list.append(eval_loss.detach().item())

                # 
                if verbose:
                    p = 100 * it // max_iter
                    pp = p // 4
                    print("\r" + "#" * pp + " " * (25 - pp) + "|" + str(p) + "%, best_loss: {:.5e}, train_loss: {:.5e}, eval_loss: {:.5e}".format(best_loss, mean_pred_loss, eval_loss), end='')

                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_it = it
                elif (it - best_it) >= lookback * check_every and it > min_iter:
                    if verbose:
                        print("\nStopping early")
                    break
        x_predict_eval = model(x_eval[:,:,:-1])
      

        return model, it_list, tot_loss_list, exper_loss_list, eval_loss_list,best_loss, mean_pred_loss, eval_loss
    
    def get_gc_metrics(self, x, gc):
        """
        x: [d, t] or [batch, d, t]
        """
        gc = np.array(gc)
        x = torch.tensor(x, device=next(self.parameters()).device)
        if 2 == len(x.shape): x.unsqueeze_(0)
        x = x.transpose(1, 2).unfold(1, self.lag, 1)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        jaco_gc = self.jacobian_causal(x).detach().cpu().numpy()
        ret = {}

        
        
        maxlag = max(jaco_gc.shape[2], gc.shape[2])
        pred_gc =  np.pad(jaco_gc, ((0, 0), (0, 0), (maxlag - jaco_gc.shape[2], 0)), 'constant', constant_values=0)
        true_gc = np.pad(gc, ((0, 0), (0, 0), (maxlag - gc.shape[2], 0)), 'constant', constant_values=0)
        # same (following)
        # ground_truth_flattened = true_gc.flatten()
        # score_matrix_flattened = pred_gc.flatten()
        # fpr, tpr, thresholds = metrics.roc_curve(ground_truth_flattened, score_matrix_flattened)
        # aucroc = metrics.auc(fpr, tpr)
        # auprc = metrics.average_precision_score(ground_truth_flattened, score_matrix_flattened)

        (f1, f1_eps), (acc, acc_eps), (auroc, roc_x, roc_y), (auprc, prc_x, prc_y) = two_classify_metrics(pred_gc, true_gc)
        ret['causal with lag use jaco'] = {'f1':f1, 'f1_eps':f1_eps, 'acc':acc, 'acc_eps':acc_eps, 'auroc':auroc, 'auprc':auprc, 'rox_x':roc_x, 'roc_y':roc_y, 'prc_x':prc_x, 'prc_y':prc_y}

        # Easy
        gc = (np.sum(gc, axis=2) > 0).astype(np.int32)
        jaco_gc = np.max(jaco_gc, axis=2)

        # 
        pred_gc = jaco_gc
        true_gc = gc
        

        (f1, f1_eps), (acc, acc_eps), (auroc, roc_x, roc_y), (auprc, prc_x, prc_y) = two_classify_metrics(pred_gc, true_gc)
        ret['causal no lag use jaco'] =  {'f1':f1, 'f1_eps':f1_eps, 'acc':acc, 'acc_eps':acc_eps, 'auroc':auroc, 'auprc':auprc, 'rox_x':roc_x, 'roc_y':roc_y, 'prc_x':prc_x, 'prc_y':prc_y}
        return ret
        
