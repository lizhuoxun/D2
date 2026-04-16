import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from models.myevaluate import evaluate_model, plot_roc
from models.data import load_splits,NVFAIRDataset,ContrastiveSampler,collate_fn,split_dataset
from models.mynetwork import IDreveal
from models.disentangler import SpatioTemporalConsistencyDisentangler


parser = argparse.ArgumentParser()
parser.add_argument('--gen', default='facevid2vid', help='generator')
parser.add_argument('--device', default=0, type=int, help='GPU')
parser.add_argument('--savedir', default='./result/my')
parser.add_argument('--resume', default='')

opt = parser.parse_args()
print(opt)

batch_ids = 6
num_video = 6
num_clip = 5

def find_nested_index(lst, target, path=[]):
    for idx, item in enumerate(lst):
        current_path = path + [idx]
        if isinstance(item, list):
            result = find_nested_index(item, target, current_path)
            if result: return result
        elif item == target:
            return current_path
    return None

row_indices = []
index = 0
for i in range(batch_ids):
    row_indices.append([])
    for j in range(batch_ids):
        if i == j: 
            row_indices[i].append(list(range(index, index + num_video + 1)))
            index += (num_video +1)
        else:
            row_indices[i].append(list(range(index, index + num_video)))
            index += num_video  

def list_to_np(indices_list, indices):
    for index in range(len(indices)):
        ijk = find_nested_index(row_indices, index)
        indices[index] = np.array(indices_list[ijk[0]][ijk[1]][ijk[2]])
    return 0

indices_posself_list = []              
for i in range(batch_ids):
    indices_posself_list.append([])
    for j in range(batch_ids):
        indices_posself_list[i].append([])
        for k in range(num_video + int(i == j)):
            indices_posself_list[i][j].append([x for x in row_indices[i][i] if x != row_indices[i][j][k]][:num_video])
indices_posself = np.zeros((batch_ids * (batch_ids * num_video + 1), num_video), dtype=int)
list_to_np(indices_posself_list, indices_posself)

indices_neg_list = []              
for i in range(batch_ids):
    indices_neg_list.append([])
    for j in range(batch_ids):
        indices_neg_list[i].append([])
        for k in range(num_video + int(i==j)):
            indices_neg_list[i][j].append([])
            for row in range(batch_ids):
                if row != i: indices_neg_list[i][j][k] += row_indices[row][j][:num_video]
indices_neg = np.zeros((batch_ids * (batch_ids * num_video + 1), (batch_ids - 1) * num_video), dtype=int)
list_to_np(indices_neg_list, indices_neg)

def gen_poscross(indices_poscross):
    indices_poscross_list = []              
    for i in range(batch_ids):
        indices_poscross_list.append([])
        candidate = set(range(i * (batch_ids * num_video + 1), (i + 1) * (batch_ids * num_video + 1)))
        candidate = list(candidate - set(row_indices[i][i]))
        poscross = random.sample(candidate, num_video + 1)
        for j in range(batch_ids):
            indices_poscross_list[i].append([])
            for k in range(num_video + int(i==j)):
                indices_poscross_list[i][j].append([x for x in poscross if x != row_indices[i][j][k]][:num_video])
    list_to_np(indices_poscross_list, indices_poscross)


def max_norm(clips1, clips2):

    similar_metric = torch.exp(-torch.cdist(clips1, clips2) ** 2)
    return torch.clamp(torch.max(similar_metric, 3)[0], min=1e-8)

class ContrastiveLoss(nn.Module):
    def forward(self, embeddings, batch_size, indices_poscross, order_to_shuffle):
        combined = np.hstack((indices_posself, indices_poscross))
        indices_negshuffle = order_to_shuffle[combined]

        combined = np.hstack((combined, indices_neg, indices_negshuffle))
        ct_embeddings = embeddings[combined]
        embeddings = embeddings[:batch_size].unsqueeze(1).expand(-1,ct_embeddings.size(1),-1,-1)

        maxnorm = max_norm(embeddings, ct_embeddings)

        pos_sim = torch.sum(maxnorm[:, :2*num_video], 1)
        neg_sim = torch.sum(maxnorm[:, 2*num_video:], 1)

        loss = torch.sum(-torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))) / float(batch_size)
        return 500*loss

splits={}
splits['train'], splits['val'], splits['test'] = load_splits("train_val_test_splits.txt")
if opt.gen != 'facevid2vid': splits['val'].remove('1060')

meta_df = pd.read_csv("cleaned_complete_videos_38.csv", dtype={4:str})

os.makedirs('%s' % opt.savedir, exist_ok=True)
os.makedirs('%s/%s' % (opt.savedir, opt.gen), exist_ok=True)

train_files, val_files, test_files = split_dataset(meta_df, splits, opt.gen)

train_set = NVFAIRDataset(train_files, splits['train'], 'train')
val_set = NVFAIRDataset(val_files, splits['val'], 'train')
test_set = NVFAIRDataset(test_files, splits['test'], 'test')

sampler = ContrastiveSampler(train_set, batch_ids=batch_ids, num_video=num_video)

batch_size = batch_ids * (batch_ids * num_video + 1)

train_loader = DataLoader(
    train_set,
    sampler=sampler,
    collate_fn=collate_fn,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True
)
val_loader = DataLoader(val_set)

device=torch.device("cuda:%d"%opt.device)
disentangler = SpatioTemporalConsistencyDisentangler(75, 106)
temp_3dcnn = IDreveal(71)

disentangler.to(device)
temp_3dcnn.to(device)
optimizer1 = torch.optim.Adam(disentangler.parameters(), lr=0.0001)
optimizer2 = torch.optim.Adam(temp_3dcnn.parameters(), lr=0.0001)

if opt.resume != '':
    print("load the trained model")
    state = torch.load('%s'%opt.resume)
    disentangler.load_state_dict(state['state_dict_disentangler'])
    temp_3dcnn.load_state_dict(state['state_dict_3dcnn'])
    optimizer1.load_state_dict(state['state_dict_optimizer1'])
    optimizer2.load_state_dict(state['state_dict_optimizer2'])

ctloss = ContrastiveLoss()

auc_max = 0

for batch_idx, inputs in enumerate(train_loader):
    landmarks, driving_ids, target_ids = inputs

    size = landmarks.size()
    print('batch_idx:', batch_idx)

    indices_poscross = np.zeros((size[0], num_video), dtype=int)
    gen_poscross(indices_poscross)

    landmarks = landmarks.to(device)/256
    target_feat, driven_feat, reconstructed = disentangler(landmarks)
    feats = torch.cat([target_feat, driven_feat], dim=2)

    order_to_shuffle = np.zeros(size[0], dtype=int)
    self_set = sorted(set(indices_posself.reshape(-1).tolist()))
    cross_set = sorted(set(indices_poscross.reshape(-1).tolist()))
    for i, index in enumerate(self_set + cross_set): order_to_shuffle[index] =  i + size[0]
    feats = torch.cat([feats, feats[self_set], feats[cross_set]], dim=0)

    indices = torch.randperm(size[1])  
    feats[size[0]:] = feats[size[0]:, indices, :]

    noise = torch.randn_like(feats[:, :, target_feat.size(2):])
    feats_rand = feats.clone()
    feats_rand[:, :, target_feat.size(2):] += 0.05 * noise

    embedding = temp_3dcnn(feats)
    c_embedding = temp_3dcnn(feats_rand)
    dis_loss, rec_loss, target_loss, driving_loss, independence_loss = disentangler.get_loss(landmarks, reconstructed, target_feat, driven_feat)

    ct_loss = ctloss(embedding, size[0], indices_poscross, order_to_shuffle)
    ct_rand_loss = ctloss(c_embedding, size[0], indices_poscross, order_to_shuffle)

    loss = 1000 * dis_loss + ct_loss - 0.1 * ct_rand_loss
    print("rec loss:", rec_loss.item(), "target loss:", target_loss.item(), "driven loss:", driving_loss.item(), "indep loss:", independence_loss.item(), "ct loss:", ct_loss.item(), "ct rand loss:", ct_rand_loss.item())
    print("total loss:",loss.item())
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss.backward()
    optimizer1.step()
    optimizer2.step()
    if batch_idx % 100 == 0:
        auc_score, fpr, tpr = evaluate_model(disentangler, temp_3dcnn, val_loader, device)
        auc_macro = auc_score["macro"]
        print(f"Test AUC: {auc_macro:.4f}")
        if auc_macro > auc_max:
            auc_max = auc_macro
            plot_roc(auc_macro, fpr["macro"], tpr["macro"], '%s/%s/roc_curve_%d.png' % (opt.savedir, opt.gen, batch_idx))
            state = {'state_dict_disentangler':disentangler.state_dict(), 'state_dict_3dcnn':temp_3dcnn.state_dict(), 'state_dict_optimizer1': optimizer1.state_dict(), 'state_dict_optimizer2': optimizer2.state_dict()}
            torch.save(state, '%s/%s/model.pickle' % (opt.savedir, opt.gen))
        temp_3dcnn.train()
    if batch_idx == 100000:  
        print(f"Max AUC: {auc_max:.4f}")
        break
