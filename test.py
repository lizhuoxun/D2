import torch
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from models.myevaluate import evaluate_model, test_model, plot_roc
from models.data import load_splits,NVFAIRDataset,ContrastiveSampler,collate_fn,split_dataset
from models.mynetwork import IDreveal
from models.disentangler import SpatioTemporalConsistencyDisentangler


def convert_to_np_float(d):
    for key in d:
        if isinstance(d[key], float):  
            d[key] = np.float64(d[key])  
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traingen', default='facevid2vid', help='generator of training')
    parser.add_argument('--testgen', default='facevid2vid', help='generator of testing')
    parser.add_argument('--device', default=0, type=int, help='GPU')
    parser.add_argument('--savedir', default='./result/my')

    opt = parser.parse_args()
    print(opt)

    splits={}
    splits['train'], splits['val'], splits['test'] = load_splits("train_val_test_splits.txt")
    if opt.testgen != 'facevid2vid': splits['val'].remove('1060')
    meta_df = pd.read_csv("cleaned_complete_videos_38.csv", dtype={4:str})
    train_files, val_files, test_files = split_dataset(meta_df, splits, opt.testgen)
    test_set = NVFAIRDataset(test_files, splits['test'], 'test')
    test_loader = DataLoader(test_set)

    device=torch.device("cuda:%d"%opt.device)
    disentangler = SpatioTemporalConsistencyDisentangler(75, 106)
    disentangler.to(device)
    temp_3dcnn = IDreveal(71)
    temp_3dcnn.to(device)
    state = torch.load('%s/%s/model.pickle' % (opt.savedir, opt.traingen))
    disentangler.load_state_dict(state['state_dict_disentangler'])
    temp_3dcnn.load_state_dict(state['state_dict_3dcnn'])

    auc_score, fpr, tpr = test_model(disentangler, temp_3dcnn, test_loader, device)
    for key in auc_score: print(key, auc_score[key])
    plot_roc(auc_score["macro"], fpr["macro"], tpr["macro"], '%s/%s/roc_curve_%s.png' % (opt.savedir, opt.traingen, opt.testgen))
    
    np.savez('%s/%s/auc_score_%s.npz'% (opt.savedir, opt.traingen, opt.testgen), **auc_score)
    np.savez('%s/%s/fpr_%s.npz'% (opt.savedir, opt.traingen, opt.testgen), **fpr)
    np.savez('%s/%s/tpr_%s.npz'% (opt.savedir, opt.traingen, opt.testgen), **tpr)
