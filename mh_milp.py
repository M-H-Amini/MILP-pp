import torch 
import torchvision
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
from torchvision.models import vgg16_bn, resnet18, vit_b_16
from torch.nn import Module, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import mh_data as mhd 
import mh_models as mhm
import mh_optimize as mho
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import os

def experiment(ds_name, alpha, model_names, lrs, h_init=0.01, s=1000, batch_size=1024, epochs=10, device='cuda'):
    ##  Data preparation...
    t0 = time()
    ds_unlabelled, ds_fine_tuning, ds_optimization = mhd.dataSplit(ds_name, h_init=h_init, s=s)
    t1 = time()
    t_data = round(t1 - t0, 4)
    print(f'Data preparation done! It took {t_data} seconds.')
    m = len(ds_unlabelled) + len(ds_fine_tuning) + len(ds_optimization)
    ##  Fine-tuning...
    t0 = time()
    models = []
    for i, model_name in enumerate(model_names):
        print(f'Fine-tuning {model_name}...')
        model, transform, train_loss, val_loss = mhm.finetune(model_name, ds_fine_tuning, batch_size=batch_size, lr=lrs[i], n_epochs=epochs, device=device)
        models.append((model, transform))
        print(f'Training loss: {train_loss}, Validation loss: {val_loss}')
    t1 = time()
    t_fine_tuning = round(t1 - t0, 4)
    print(f'Fine-tuning done! It took {t_fine_tuning} seconds.')
    ##  Prediction on the optimization set...
    print('Predicting on the optimization set...')
    
    df_pred_optimization = predict(models, ds_optimization, batch_size=batch_size, device=device)

    # print(df_pred_optimization)
    print('Predicting on the optimization set done!')
    
    ##  MILP optimization...
    print('Optimizing...')
    t0 = time()
    effort_optimization, w_list, x_list = mho.mh_optimize(df_pred_optimization, alpha=alpha)
    t1 = time()
    t_optimization = round(t1 - t0, 4)
    print(f'Optimization done! It took {t_optimization} seconds.')
    print('Effort on the optimization set:', effort_optimization / len(df_pred_optimization))
    print('w_list:', w_list)

    ##  Prediction on the unlabelled set...
    print('Predicting on the unlabelled set...')
    t0 = time()
    df_pred_unlabelled = predict(models, ds_unlabelled, batch_size=batch_size, device=device)
    t1 = time()
    t_labelling = round(t1 - t0, 4)
    print(f'Labelling done! It took {t_labelling} seconds.')
    df_pred_unlabelled['z'] = (df_pred_unlabelled[[f'p_l_{i}' for i in range(len(models))]].nunique(axis=1) == 1).astype(int)
    df_pred_unlabelled['b'] = (df_pred_unlabelled[[f'p_l_{i}' for i in range(len(models))] + ['y']].nunique(axis=1) == 1).astype(int)
    theta_list = [df_pred_unlabelled[f'p_theta_{i}'] for i in range(len(models))]
    f = np.sum([w_list[i] * theta_list[i] for i in range(len(models))], axis=0) - 1
    df_pred_unlabelled['f'] = f
    effort_unlabelled = len(df_pred_unlabelled) - np.sum(df_pred_unlabelled['z'] * (f>0))  ##  Unnormalized effort
    accuracy_unlabelled = np.sum(df_pred_unlabelled['b'] * (f>0)) + effort_unlabelled  ##  Unnormalized accuracy
    print('Effort on the unlabelled set:', effort_unlabelled)
    print('Accuracy on the unlabelled set:', accuracy_unlabelled)
    effort_total = (len(ds_fine_tuning) + len(ds_optimization) + effort_unlabelled) / m
    accuracy_total = (len(ds_fine_tuning) + len(ds_optimization) + accuracy_unlabelled) / m 
    print('Total effort:', effort_total)
    print('Total accuracy:', accuracy_total)
    ##  Output...
    output_name = os.path.join('outputs', f'results_milp_{"_".join(model_names)}.csv')
    columns = ['dataset', 'size', 'models', 'h_init', 's', 'batch_size', 'epochs', 't_data', 't_fine_tuning', 't_optimization', 't_labelling', 'effort_optimization', 'effort_unlabelled', 'accuracy_unlabelled', 'effort_total', 'accuracy_total']
    values = [ds_name, m, model_names, h_init, s, batch_size, epochs, t_data, t_fine_tuning, t_optimization, t_labelling, effort_optimization/len(ds_optimization), effort_unlabelled/len(ds_unlabelled), accuracy_unlabelled/len(ds_unlabelled), effort_total, accuracy_total]
    if os.path.exists(output_name):
        df_output = pd.read_csv(output_name)
        df_output.loc[len(df_output)] = values
    else:
        os.makedirs('outputs', exist_ok=True)
        df_output = pd.DataFrame(columns=columns)
        df_output.loc[0] = values
    df_output.to_csv(output_name, index=False)
    print('Output saved at:', output_name)

        

def predict(models, ds, batch_size=1024, device='cuda'):
    pred_cols = [f'p_l_{i}' for i in range(len(models))] + [f'p_theta_{i}' for i in range(len(models))] + ['y']
    df_pred = pd.DataFrame(columns=pred_cols)
    for i, (model, transform) in enumerate(models):
        model.eval()
        ds.transform = transform
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        pred_l = []
        pred_theta = []
        y_list = []
        for x, y in tqdm(dl):
            x = x.to(device)
            y_list.extend(y.cpu().numpy().tolist())
            with torch.no_grad():
                y_hat = model(x)
                l = torch.argmax(y_hat, dim=1).cpu().numpy().tolist()
                theta = torch.nn.functional.softmax(y_hat, dim=1).max(dim=1).values.cpu().numpy().tolist()
                pred_l.extend(l)
                pred_theta.extend(theta)
        
        df_pred[f'p_l_{i}'] = pred_l
        df_pred[f'p_theta_{i}'] = pred_theta
        df_pred[f'y'] = y_list
    
    return df_pred

    

if __name__ == '__main__':
    datasets = ['cifar10', 'fashionmnist', 'svhn', 'mnist']
    h_values = [0.01, 0.05, 0.15, 0.25, 0.35, 0.45]
    alpha = 1
    repeats = 5
    for ds_name in datasets:
        for h_init in h_values:
            for i in range(repeats):
                print(f'Experiment on {ds_name} with h_init={h_init} and repeat={i+1}...')
                experiment(ds_name, alpha, ['vgg', 'resnet', 'vit'], [1e-3, 1e-3, 1e-3], h_init=h_init, s=500, epochs=50)
    