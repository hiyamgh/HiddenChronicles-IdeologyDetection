#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
from evaluator import get_murank, get_pk
from sklearn.metrics.pairwise import cosine_similarity
import os


def data(model_name, TEST_ON): #model_name:{r,f,rf}; TEST_ON: 1-13    
    data_folder = '../data_final_proj/'
    ts = pickle.load(open(data_folder+'vectors.pkl', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.pkl', 'rb'))

    # data_folder = '../data/'
    # ts = pickle.load(open(data_folder + 'vectors.p', 'rb'))
    # test_idx = pickle.load(open(data_folder + 'test_idx.p', 'rb'))

    X = ts[test_idx, 0:TEST_ON, :]
    if model_name=='r':
        Y = ts[test_idx, 0:TEST_ON, :]
    elif model_name=='f':
         Y = ts[test_idx, TEST_ON:, :]
    elif model_name=='rf':
        Y =  ts[test_idx, :, :]
    folder = '../results_final/results_seq2seq_'+model_name+'/'
    model = pickle.load(open(folder+str(TEST_ON)+'model2.p', 'rb'))
    return X, Y, model


def data_baselines(model_name, TEST_ON): #reconstruct, future; 1-13   
    data_folder = '../data/'
    ts = pickle.load(open(data_folder+'vectors.p', 'rb'))
    test_idx = pickle.load(open(data_folder+'test_idx.p', 'rb'))

    X =  np.concatenate((ts[test_idx,0,:].reshape(-1,1,ts.shape[2]), ts[test_idx,TEST_ON,:].reshape(-1,1,ts.shape[2])), axis=1) 

    if model_name=='r':
        Y =  np.concatenate((ts[test_idx,0,:].reshape(-1,1,ts.shape[2]), ts[test_idx,TEST_ON,:].reshape(-1,1,ts.shape[2])), axis=1) 
    elif model_name=='f':
        X =  ts[test_idx,0,:].reshape(-1,1,ts.shape[2])
        Y = ts[test_idx, TEST_ON, :]
    
    folder = '../results/results_lstm_'+model_name+'/'
    model = pickle.load(open(folder+str(TEST_ON)+'model2.p', 'rb'))
    return X, Y, model


def get_results(model_type, model_name, TEST_ON): #{model,baseline}, {r,f,rf}, TEST_ON
    if model_type=='model':
        X, Y, model = data(model_name, TEST_ON)
    else:
        X, Y, model = data_baselines(model_name, TEST_ON)        
    predictions = model.predict(X)

    if model_name=='rf':
        Y_past = Y[:, 0:TEST_ON, :]    
        Y_future = Y[:, TEST_ON:, :]
        past_preds, future_preds = predictions[0], predictions[1]
        
        errors = []
        all_errors = []
        for timestep in range(past_preds.shape[1]):
            preds = past_preds[:,timestep,:]
            actuals = Y_past[:,timestep,:]
            tmpval = np.diag(cosine_similarity(preds, actuals))
            errors.append(tmpval)
            all_errors.append(tmpval)
        errors_past = np.array(errors)
        avg_errors_past = np.average(errors_past, axis=0)
        p5 = 100.0*get_pk(avg_errors_past, int(0.05*len(X)))
        p10 = 100.0*get_pk(avg_errors_past, int(0.1*len(X)))
        p50 = 100.0*get_pk(avg_errors_past, int(0.5*len(X)))
            
        errors = []
        for timestep in range(future_preds.shape[1]):
            preds = future_preds[:,timestep,:]
            actuals = Y_future[:,timestep,:]
            tmpval = np.diag(cosine_similarity(preds, actuals))
            errors.append(tmpval)
            all_errors.append(tmpval)
        errors_future = np.array(errors)
        avg_errors_future = np.average(errors_future, axis=0)
        p5 = 100.0*get_pk(avg_errors_future, int(0.05*len(X)))
        p10 = 100.0*get_pk(avg_errors_future, int(0.1*len(X)))
        p50 = 100.0*get_pk(avg_errors_future, int(0.5*len(X)))
        
        all_errors = np.array(all_errors)
        avg_errors_combined = np.average(all_errors, axis=0)
        murank_micro = 100.0*get_murank(avg_errors_combined)
        p5 = 100.0*get_pk(avg_errors_combined, int(0.05*len(X)))
        p10 = 100.0*get_pk(avg_errors_combined, int(0.1*len(X)))
        p50 = 100.0*get_pk(avg_errors_combined, int(0.5*len(X)))
        
        print('TEST_ON:', TEST_ON, 'Cosine mu-rank, prec@k:', '\t', murank_micro, '\t', p5, '\t', p10, '\t', p50)
        results = [murank_micro, p5, p10, p50]
    else:
        errors = []
        print('Shape of predictions:', predictions.shape)
        for timestep in range(predictions.shape[1]):
            preds = predictions[:,timestep,:]
            if (model_type=='baseline') & (model_name=='f'):
                actuals = Y
            else:
                actuals = Y[:,timestep,:] #just Y for baseline_future_lstm
            errors.append(np.diag(cosine_similarity(preds, actuals)))
        errors = np.array(errors)
        avg_errors = np.average(errors, axis=0)
        murank = 100.0*get_murank(avg_errors)
        p5 = 100.0*get_pk(avg_errors, int(0.05*len(X)))
        p10 = 100.0*get_pk(avg_errors, int(0.1*len(X)))
        p50 = 100.0*get_pk(avg_errors, int(0.5*len(X)))
    
        print('TEST_ON:', TEST_ON, 'Cosine, mu-rank, prec@k:', '\t', murank, '\t',  p5, '\t', p10, '\t', p50)
        results = [murank, p5, p10, p50]
    return results, all_errors


def generate_cosine_heatmap(data, name):
    from bidi import algorithm as bidialg
    import arabic_reshaper
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt
    with open('../data_final_proj/keywords_testing.txt', 'r', encoding='utf-8') as f:
        yticks = f.readlines()
    yticks = [bidialg.get_display(arabic_reshaper.reshape(y[:-1])) for y in yticks]

    if name != 'legend':
        # if name == 'sigmoid_future':
        #     data = np.concatenate((np.ones(data.shape[0]).reshape((-1, 1)), data),
        #                           axis=1)  # dummy input for pseudo-perfect prediction at timestep 0
        ax = sns.heatmap(data, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=True)
    else:
        ax = sns.heatmap(data, vmin=-0.1, vmax=1.0, yticklabels=yticks, cmap="YlGnBu", cbar=True)

    ax.tick_params(labelsize=8)
    # ax.set_yticks(yticks)

    folder = '../img/'
    if not os.path.exists(folder):
        os.mkdir(folder)

    figure = ax.get_figure()
    figure.savefig(folder + name + '2.png', dpi=400, bbox_inches='tight')
    figure.clear()
    ax.clear()
    # plt.clf()
    plt.show()


if __name__ == '__main__':
    _, all_errors = get_results(model_type='model', model_name='rf', TEST_ON=4)
    words_cosine_sims = np.array(all_errors).T
    generate_cosine_heatmap(words_cosine_sims, 'keywords_testing')
    # print(all_errors.shape)
    # all_errors_res = all_errors.T
    # # save result of all errors into a pkl file for plotting
    # with open(os.path.join('../data_proj/', 'all_errors.pkl'), 'wb') as f:
    #     pickle.dump(all_errors_res, f)
    #
    # with open(os.path.join('../data_proj/', 'imp_idx.pkl'), 'rb') as f:
    #     imp_idx = pickle.load(f)
    #
    # print(imp_idx)