import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.data import load_splits,NVFAIRDataset,ContrastiveSampler,collate_fn,split_dataset


def evaluate_model(disentangler, temp_3dcnn, val_loader, device):
    disentangler.eval()
    temp_3dcnn.eval()
    embeddings = []
    driving_ids = []
    target_ids = []
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Extracting embeddings"):
            landmarks = data["landmark"]
            size = landmarks.size()
            landmarks = landmarks.to(device)/256
            source_feat, driven_feat, reconstructed = disentangler(landmarks)
            feats = torch.cat([source_feat, driven_feat], dim=2)
            emb = torch.mean(temp_3dcnn(feats), dim=1)
            embeddings.append(emb.cpu().numpy())
            driving_ids += data["driving_id"]
            target_ids += data["target_id"]
    
    embeddings = np.concatenate(embeddings)

    id_to_embeddings = {}
    scores = {}
    labels = {}
    for i in range(len(embeddings)):
        if target_ids[i] not in id_to_embeddings:
            id_to_embeddings[target_ids[i]] = []
            scores[target_ids[i]] = []
            labels[target_ids[i]] = []
        if target_ids[i] == driving_ids[i]: id_to_embeddings[target_ids[i]].append((i, embeddings[i]))


    for idx in tqdm(range(len(embeddings)), desc="Computing scores"):
        current_emb = embeddings[idx]
        current_did = driving_ids[idx]
        current_tid = target_ids[idx]
        
        is_positive = (current_did == current_tid)
        labels[current_tid].append(1 if is_positive else 0)
        
        ref_embs = np.array([emb[1] for emb in id_to_embeddings[current_tid] if idx != emb[0]])
        distances = np.linalg.norm(ref_embs - current_emb, axis=1)
        score = 1 - np.mean(distances)
        
        scores[current_tid].append(score)

    fpr = {}
    tpr = {}
    roc_auc = {}
    for tid in id_to_embeddings:
        fpr[tid], tpr[tid], _ = roc_curve(labels[tid], scores[tid])
        roc_auc[tid] = auc(fpr[tid], tpr[tid])
    
    all_fpr = np.unique(np.concatenate([fpr[tid] for tid in id_to_embeddings]))
    mean_tpr = np.zeros_like(all_fpr)
    for tid in id_to_embeddings:
        mean_tpr += np.interp(all_fpr, fpr[tid], tpr[tid])
    mean_tpr /= len(id_to_embeddings)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc, fpr, tpr

def test_model(disentangler, temp_3dcnn, test_loader, device):
    disentangler.eval()
    temp_3dcnn.eval()
    embeddings = []
    driving_ids = []
    target_ids = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Extracting embeddings"):
            landmarks = data["landmark"]
            size = landmarks.size()
            landmarks = landmarks.to(device)/256
            source_feat, driven_feat, reconstructed = disentangler(landmarks)
            feats = torch.cat([source_feat, driven_feat], dim=2)
            emb = torch.squeeze(temp_3dcnn(feats), dim=0)
            embeddings.append(emb.cpu().numpy())
            driving_ids.extend(data["driving_id"] * len(emb))
            target_ids.extend(data["target_id"] * len(emb))
    

    embeddings = np.concatenate(embeddings)

    id_to_embeddings = {}
    scores = {}
    labels = {}
    for i in range(len(embeddings)):
        if target_ids[i] not in id_to_embeddings:
            id_to_embeddings[target_ids[i]] = []
            scores[target_ids[i]] = []
            labels[target_ids[i]] = []
        if target_ids[i] == driving_ids[i]: id_to_embeddings[target_ids[i]].append((i, embeddings[i]))
    
    for idx in tqdm(range(len(embeddings)), desc="Computing scores"):
        current_emb = embeddings[idx]
        current_did = driving_ids[idx]
        current_tid = target_ids[idx]
        
        is_positive = (current_did == current_tid)
        labels[current_tid].append(1 if is_positive else 0)
        
        ref_embs = np.array([emb[1] for emb in id_to_embeddings[current_tid] if idx != emb[0]])
        distances = np.linalg.norm(ref_embs - current_emb, axis=1)
        score = 1 - np.mean(distances)
        
        scores[current_tid].append(score)

    fpr = {}
    tpr = {}
    roc_auc = {}
    for tid in id_to_embeddings:
        fpr[tid], tpr[tid], _ = roc_curve(labels[tid], scores[tid])
        roc_auc[tid] = auc(fpr[tid], tpr[tid])
    
    all_fpr = np.unique(np.concatenate([fpr[tid] for tid in id_to_embeddings]))
    mean_tpr = np.zeros_like(all_fpr)
    for tid in id_to_embeddings:
        mean_tpr += np.interp(all_fpr, fpr[tid], tpr[tid])
    mean_tpr /= len(id_to_embeddings)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc, fpr, tpr


def plot_roc(roc_auc, fpr, tpr, path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=300, bbox_inches='tight')
