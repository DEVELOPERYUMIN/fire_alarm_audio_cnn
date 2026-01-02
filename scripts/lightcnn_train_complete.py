#!/usr/bin/env python3
# scripts/train_experiments.py
# ❶ 경로/라이브러리 ---------------------------------------------------------
import os, glob, random, itertools, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

BASE_DIR    = "/Users/minimac/Desktop/work/fire_classification"
DATA_DIR    = os.path.join(BASE_DIR, 'logmel_data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ❷ 모델 ---------------------------------------------------------------
class LightCNN(nn.Module):
    def __init__(self, dropout=0.3, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels*2, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels*2)
        self.pool  = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.gap   = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(channels*2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# ❸ 데이터셋 ------------------------------------------------------------
class LogMelDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        f, lab = self.samples[idx]
        mel = torch.tensor(np.load(f), dtype=torch.float32)
        if mel.ndim == 2: mel = mel.unsqueeze(0)
        return mel, torch.tensor(lab)

def prepare_splits(root, r=(0.7,0.15,0.15), seed=42):
    pos = sorted(glob.glob(os.path.join(root,'positive','*.npy')))
    neg = sorted(glob.glob(os.path.join(root,'negative','*.npy')))
    all_s = [(f,1) for f in pos]+[(f,0) for f in neg]
    random.Random(seed).shuffle(all_s)
    n=len(all_s); n_tr=int(r[0]*n); n_val=int(r[1]*n)
    return all_s[:n_tr], all_s[n_tr:n_tr+n_val], all_s[n_tr+n_val:]

# ❹ SpecAugment & Mixup -------------------------------------------------
def apply_freq_mask(x,F):
    f=random.randint(0,F); f0=random.randint(0,max(0,x.size(2)-f))
    x[:,:,f0:f0+f,:]=0; return x
def apply_time_mask(x,Tm):
    t=random.randint(0,Tm); t0=random.randint(0,max(0,x.size(3)-t))
    x[:,:,:,t0:t0+t]=0; return x
def mixup_data(x,y,a):
    if a<=0: return x,y,None,1.0
    lam=np.random.beta(a,a); idx=torch.randperm(x.size(0)).to(x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

# ❺ collate_fn (시간축 패딩) --------------------------------------------
def pad_collate(batch):
    xs,ys=zip(*batch); maxT=max(t.shape[-1] for t in xs)
    xs=[F.pad(t,(0,maxT-t.shape[-1])) for t in xs]
    return torch.stack(xs), torch.stack(ys)

# ❻ 학습·평가 -----------------------------------------------------------
def train_and_eval(args):
    tr,val,tes = prepare_splits(DATA_DIR,(0.7,0.15,0.15),args.seed)
    tl = lambda s,shuf: DataLoader(LogMelDataset(s),
                                   batch_size=args.batch_size,
                                   shuffle=shuf, collate_fn=pad_collate)
    train_loader,val_loader,test_loader = tl(tr,True),tl(val,False),tl(tes,False)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=LightCNN(args.dropout,args.channels).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0,args.pos_weight],device=device))
    optim_ = optim.Adam(model.parameters(),lr=args.lr)

    tr_losses,val_losses,val_accs=[],[],[]
    for ep in range(1,args.epochs+1):
        model.train(); run=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            for _ in range(args.num_masks):
                x=apply_freq_mask(x,args.freq_mask); x=apply_time_mask(x,args.time_mask)
            x,y_a,y_b,lam=mixup_data(x,y,args.mixup_alpha)
            optim_.zero_grad(); out=model(x)
            loss = lam*crit(out,y_a)+(1-lam)*crit(out,y_b) if y_b is not None else crit(out,y)
            loss.backward(); optim_.step(); run+=loss.item()*x.size(0)
        tr_losses.append(run/len(train_loader.dataset))

        model.eval(); run=correct=tot=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device); out=model(x)
                run+=crit(out,y).item()*x.size(0)
                preds=(torch.softmax(out,1)[:,1]>args.threshold)
                correct+=(preds==y).sum().item(); tot+=y.size(0)
        val_losses.append(run/len(val_loader.dataset)); val_accs.append(correct/tot)
        print(f"[{ep}/{args.epochs}] train_loss={tr_losses[-1]:.3f} "
              f"val_loss={val_losses[-1]:.3f} val_acc={val_accs[-1]:.3f}")

    # 테스트
    model.eval(); run=correct=tot=0; y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device); out=model(x)
            run+=crit(out,y).item()*x.size(0)
            prob=torch.softmax(out,1)[:,1]; pred=(prob>args.threshold).long()
            correct+=(pred==y).sum().item(); tot+=y.size(0)
            y_true+=y.cpu().tolist(); y_pred+=pred.cpu().tolist()
    test_loss=run/len(test_loader.dataset); test_acc=correct/tot

    # ─ 파일 저장 (일관적 tag) ------------------------------------------
    tag=f"{args.model_name}_exp{args.exp_idx:03d}"
    torch.save(model.state_dict(),os.path.join(MODELS_DIR,f"{tag}.pt"))
    plt.figure(); plt.plot(tr_losses,label='train'); plt.plot(val_losses,label='val')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.savefig(os.path.join(RESULTS_DIR,f"{tag}_loss.png")); plt.close()
    plt.figure(); plt.plot(val_accs); plt.xlabel('Epoch'); plt.ylabel('Val Acc')
    plt.savefig(os.path.join(RESULTS_DIR,f"{tag}_acc.png")); plt.close()
    cm=confusion_matrix(y_true,y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['neg','pos']).plot()
    plt.savefig(os.path.join(RESULTS_DIR,f"{tag}_cm.png")); plt.close()

    # 지표 리턴
    return dict(
        exp_idx=args.exp_idx, dropout=args.dropout, channels=args.channels,
        freq_mask=args.freq_mask, time_mask=args.time_mask, num_masks=args.num_masks,
        mixup_alpha=args.mixup_alpha, lr=args.lr, pos_weight=args.pos_weight,
        val_loss=val_losses[-1], val_acc=val_accs[-1],
        test_loss=test_loss, test_acc=test_acc)

# ❼ CLI & 그리드 --------------------------------------------------------
def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_name',default='lightcnn')
    ap.add_argument('--exp_idx',type=int,default=1)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--batch_size',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--threshold',type=float,default=0.5)
    ap.add_argument('--pos_weight',type=float,default=1.0)
    ap.add_argument('--dropout',type=float,default=0.3)
    ap.add_argument('--channels',type=int,default=32)
    ap.add_argument('--freq_mask',type=int,default=10)
    ap.add_argument('--time_mask',type=int,default=10)
    ap.add_argument('--num_masks',type=int,default=1)
    ap.add_argument('--mixup_alpha',type=float,default=0.2)
    return ap

if __name__=="__main__":
    parser=get_parser()
    # 단일 실험?
    if len(sys.argv)>1:
        args=parser.parse_args()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        train_and_eval(args)
    else:
        # -------- 그리드 설정 ----------
        dropouts=[0.3,0.5]; channels=[16,32,64]; freq=[10]
        time=[10]; num_masks=[1]; mix=[0.2]
        lrs=[1e-3,1e-4]; pos_w=[1.2,1.4,1.5,1.55]; seeds=[42]

        grid=itertools.product(dropouts,channels,freq,time,num_masks,mix,lrs,pos_w,seeds)
        results=[]; total=1
        for lst in [dropouts,channels,freq,time,num_masks,mix,lrs,pos_w,seeds]:
            total*=len(lst)
        print("Total experiments:",total)
        exp=1
        for d,ch,fm,tm,nm,ma,lr,pw,sd in grid:
            arg=parser.parse_args([])
            arg.exp_idx=exp; arg.dropout=d; arg.channels=ch
            arg.freq_mask=fm; arg.time_mask=tm; arg.num_masks=nm
            arg.mixup_alpha=ma; arg.lr=lr; arg.pos_weight=pw; arg.seed=sd
            print(f"\n=== EXP {exp}/{total} "
                  f"(d={d},ch={ch},fm={fm},tm={tm},nm={nm},ma={ma},lr={lr},pw={pw}) ===")
            random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
            results.append(train_and_eval(arg)); exp+=1

        df=pd.DataFrame(results)
        csv_path=os.path.join(RESULTS_DIR,"grid_results.csv")
        df.to_csv(csv_path,index=False)
        print(f"\n✅ 모든 실험 완료! CSV 저장 → {csv_path}")
