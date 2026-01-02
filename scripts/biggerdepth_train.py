#!/usr/bin/env python3
# scripts/train_experiments.py  ── Jetson-Nano용 BiggerDepthwise 모델 + EarlyStopping
# ---------------------------------------------------------------
import os, glob, random, itertools, sys, argparse
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# ─── 경로 -----------------------------------------------------
BASE_DIR = "/Users/minimac/Desktop/work/fire_classification"
DATA_DIR = os.path.join(BASE_DIR, "logmel_data")
MODELS_DIR, RESULTS_DIR = [os.path.join(BASE_DIR, p) for p in ("models", "results")]
os.makedirs(MODELS_DIR,  exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)

# ❶  BiggerDepthwiseCNN --------------------------------------
def dw_sep(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, k, padding=p, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )

class BiggerDepthwiseCNN(nn.Module):
    def __init__(self, ch_base=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            dw_sep(1, ch_base), nn.MaxPool2d(2),
            dw_sep(ch_base, ch_base*2), nn.MaxPool2d(2),
            dw_sep(ch_base*2, ch_base*4), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(ch_base*4, 2)
    def forward(self, x):
        return self.fc(self.dropout(self.net(x).flatten(1)))

# ❷  데이터셋/유틸 -------------------------------------------------------
class LogMelDataset(Dataset):
    def __init__(self, samples): self.s = samples
    def __len__(self): return len(self.s)

    def __getitem__(self, i):
        f, lab = self.s[i]
        try:
            mel = np.load(f)
            if mel.ndim == 2:
                mel = mel[np.newaxis, ...]          # (1, F, T)
            mel = torch.tensor(mel, dtype=torch.float32)
            return mel, torch.tensor(lab)
        except Exception as e:                                # ★
            print(f"[WARN] load fail → {f} : {e}")            # ★
            return None                                       # ★


def prepare_splits(root,r=(0.7,0.15,0.15),seed=42):
    pos=sorted(glob.glob(os.path.join(root,"positive","*.npy")))
    neg=sorted(glob.glob(os.path.join(root,"negative","*.npy")))
    all_s=[(f,1) for f in pos]+[(f,0) for f in neg]
    random.Random(seed).shuffle(all_s); n=len(all_s); a=int(r[0]*n); b=int(r[1]*n)
    return all_s[:a],all_s[a:a+b],all_s[a+b:]

def apply_freq_mask(x,F):
    f=random.randint(0,F); f0=random.randint(0,max(0,x.size(2)-f))
    x[:,:,f0:f0+f,:]=0; return x
def apply_time_mask(x,T):
    t=random.randint(0,T); t0=random.randint(0,max(0,x.size(3)-t))
    x[:,:,:,t0:t0+t]=0; return x
def mixup_data(x,y,a):
    if a<=0: return x,y,None,1.0
    lam=np.random.beta(a,a); idx=torch.randperm(x.size(0)).to(x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam
def pad_collate(batch):
    batch = [b for b in batch if b is not None]    # ★ None 제거
    if len(batch) == 0:                            # ★ 모두 실패하면 skip
        return None
    xs, ys = zip(*batch)
    maxT = max(x.shape[-1] for x in xs)
    xs = [F.pad(x, (0, maxT - x.shape[-1])) for x in xs]
    return torch.stack(xs), torch.stack(ys)


# ❸  학습·평가 -----------------------------------------------------------
def train_and_eval(args):
    tr,val,tes = prepare_splits(DATA_DIR,(0.7,0.15,0.15),args.seed)
    L=lambda s,sh:DataLoader(LogMelDataset(s),batch_size=args.batch_size,
                              shuffle=sh,collate_fn=pad_collate)
    tl,vl,sl = L(tr,True),L(val,False),L(tes,False)

    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=BiggerDepthwiseCNN(args.channels,args.dropout).to(dev)
    crit=nn.CrossEntropyLoss(weight=torch.tensor([1.0,args.pos_weight],device=dev))
    opt=optim.Adam(model.parameters(),lr=args.lr)

    trL, valL, valA = [], [], []
    best_loss=float("inf"); best_state=None; patience=5; wait=0  # ★
    min_delta=1e-3                                             # ★

    for ep in range(1,args.epochs+1):
        model.train(); run=0
        for x,y in tl:
            if x is None:
                continue
            x,y=x.to(dev),y.to(dev)
            for _ in range(args.num_masks):
                x=apply_freq_mask(x,args.freq_mask); x=apply_time_mask(x,args.time_mask)
            x,y_a,y_b,lam=mixup_data(x,y,args.mixup_alpha)
            opt.zero_grad(); out=model(x)
            loss=lam*crit(out,y_a)+(1-lam)*crit(out,y_b) if y_b is not None else crit(out,y)
            loss.backward(); opt.step(); run+=loss.item()*x.size(0)
        trL.append(run/len(tl.dataset))

        model.eval(); run=c=t=0
        with torch.no_grad():
            for x,y in vl:
                if x is None:
                    continue
                x,y=x.to(dev),y.to(dev); out=model(x)
                run+=crit(out,y).item()*x.size(0)
                c+=(torch.softmax(out,1)[:,1]>args.threshold).eq(y).sum().item(); t+=y.size(0)
        vloss=run/len(vl.dataset); vacc=c/t
        valL.append(vloss); valA.append(vacc)
        print(f"[{ep}/{args.epochs}] train={trL[-1]:.3f} val_loss={vloss:.3f} val_acc={vacc:.3f}")

        # ★ Early-Stopping 로직
        if best_loss - vloss > min_delta:
            best_loss=vloss; best_state=model.state_dict(); wait=0
        else:
            wait+=1
            if wait>=patience:
                print(f"Early stopping at epoch {ep} (best val_loss={best_loss:.3f})")
                break

    # best 모델 로드
    if best_state is not None:
        model.load_state_dict(best_state)

    # ─ 테스트 -------------------------------------------------
    model.eval(); run=c=t=0; y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in sl:
            if x is None:
                continue            
            x,y=x.to(dev),y.to(dev); out=model(x)
            run+=crit(out,y).item()*x.size(0)
            p=(torch.softmax(out,1)[:,1]>args.threshold).long()
            c+=(p==y).sum().item(); t+=y.size(0)
            y_true+=y.cpu().tolist(); y_pred+=p.cpu().tolist()
    test_loss=run/len(sl.dataset); test_acc=c/t

    tag=f"{args.model_name}_exp{args.exp_idx:03d}"
    torch.save(model.state_dict(), os.path.join(MODELS_DIR,f"{tag}.pt"))
    plt.figure(); plt.plot(trL,label="train"); plt.plot(valL,label="val")
    plt.legend(); plt.savefig(os.path.join(RESULTS_DIR,f"{tag}_loss.png")); plt.close()
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true,y_pred),
                           display_labels=['neg','pos']).plot()
    plt.savefig(os.path.join(RESULTS_DIR,f"{tag}_cm.png")); plt.close()

    return dict(exp_idx=args.exp_idx, dropout=args.dropout, channels=args.channels,
                pos_weight=args.pos_weight, lr=args.lr,
                val_loss=best_loss, val_acc=max(valA),
                test_loss=test_loss, test_acc=test_acc)

# ❹  CLI & 그리드 -------------------------------------------------------
def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_name',default='biggerdepth')
    ap.add_argument('--exp_idx',type=int,default=1)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--batch_size',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--threshold',type=float,default=0.5)
    ap.add_argument('--pos_weight',type=float,default=1.4)
    ap.add_argument('--dropout',type=float,default=0.3)
    ap.add_argument('--channels',type=int,default=64)
    ap.add_argument('--freq_mask',type=int,default=10)
    ap.add_argument('--time_mask',type=int,default=10)
    ap.add_argument('--num_masks',type=int,default=1)
    ap.add_argument('--mixup_alpha',type=float,default=0.2)
    return ap

if __name__ == "__main__":
    parser=get_parser()
    if len(sys.argv)>1:              # 단일 실험
        args=parser.parse_args()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        train_and_eval(args)
    else:                            # 그리드
        dropouts=[0.3,0.4,0.5]; channels=[16,32,64]
        lrs=[1e-3,5e-4]; pos_w=[1.2,1.3,1.4,1.5,1.55]
        grid=itertools.product(dropouts,channels,lrs,pos_w,[42])
        total=len(dropouts)*len(channels)*len(lrs)*len(pos_w)
        print("Total experiments:",total)
        results=[]; exp=1
        for d,ch,lr,pw,_ in grid:
            arg=parser.parse_args([])
            arg.exp_idx=exp; arg.dropout=d; arg.channels=ch
            arg.lr=lr; arg.pos_weight=pw
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            print(f"\n=== EXP {exp}/{total} (d={d},ch={ch},lr={lr},pw={pw}) ===")
            results.append(train_and_eval(arg)); exp+=1
        pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"biggerdepth_grid.csv"),index=False)
        print("✅ CSV 저장 → results/biggerdepth_grid.csv")
