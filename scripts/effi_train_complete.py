#!/usr/bin/env python3
# scripts/train_experiments.py ── EfficientNet-Lite0 + EarlyStopping + Epoch CSV + plots
# ------------------------------------------------------------------
import os, glob, random, itertools, sys, argparse
import numpy as np, pandas as pd, torch, timm
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── 경로 -----------------------------------------------------------------
BASE_DIR   = "/Users/minimac/Desktop/work/fire_classification"
DATA_DIR   = os.path.join(BASE_DIR, "logmel_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR= os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 헬퍼: 그래프 저장 --------------------------------------------------- ★
def plot_loss_curve(history, out_png):
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"],   label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png); plt.close()

def plot_confmat(y_true, y_pred, out_png):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["negative","positive"])
    disp.plot(values_format='d', cmap='Blues')
    plt.title("Test Confusion Matrix"); plt.tight_layout()
    plt.savefig(out_png); plt.close()

# ❶ EfficientNet-Lite0 ----------------------------------------------------
def get_efficientnet(drop=0.3):
    return timm.create_model(
        "efficientnet_lite0", pretrained=True,
        in_chans=1, num_classes=2, drop_rate=drop)

# ❷ Dataset / Collate ----------------------------------------------------
class LogMelDataset(Dataset):
    def __init__(self, samples): self.s=samples
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        f,lab=self.s[i]
        try:
            m=np.load(f); m=m[np.newaxis,...] if m.ndim==2 else m
            return torch.tensor(m,dtype=torch.float32), torch.tensor(lab)
        except Exception as e:
            print(f"[WARN] load fail → {f}:{e}"); return None

def pad_collate(batch):
    batch=[b for b in batch if b]
    if not batch: return None
    xs,ys=zip(*batch); maxT=max(x.shape[-1] for x in xs)
    xs=[F.pad(x,(0,maxT-x.shape[-1])) for x in xs]
    xs=[F.interpolate(x.unsqueeze(0),(224,224),
         mode="bilinear",align_corners=False).squeeze(0) for x in xs]
    return torch.stack(xs), torch.stack(ys)

# ── 데이터 분할 -----------------------------------------------------------
def prepare_splits(root, ratios=(0.7,0.15,0.15), seed=42):
    pos = sorted(glob.glob(os.path.join(root,"positive","*.npy")))
    neg = sorted(glob.glob(os.path.join(root,"negative","*.npy")))
    samples=[(f,1) for f in pos]+[(f,0) for f in neg]
    random.Random(seed).shuffle(samples)
    n=len(samples); n_tr=int(ratios[0]*n); n_val=int(ratios[1]*n)
    return samples[:n_tr], samples[n_tr:n_tr+n_val], samples[n_tr+n_val:]

# ❸ SpecAug / Mixup -------------------------------------------------------
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

# ❹ Train / Eval ---------------------------------------------------------
def train_and_eval(args):
    tr,val,tes = prepare_splits(DATA_DIR,(0.7,0.15,0.15),args.seed)
    L=lambda s,sh:DataLoader(LogMelDataset(s),batch_size=args.batch_size,
                             shuffle=sh,collate_fn=pad_collate)
    tl,vl,sl=L(tr,True),L(val,False),L(tes,False)

    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=get_efficientnet(args.dropout).to(dev)
    crit=nn.CrossEntropyLoss(weight=torch.tensor([1.0,args.pos_weight],device=dev))
    opt=optim.Adam(model.parameters(),lr=args.lr)

    tag=f"{args.model_name}_exp{args.exp_idx:03d}"
    epoch_csv=os.path.join(RESULTS_DIR,f"{tag}_epoch.csv")

    history={"train_loss":[], "val_loss":[]}                         # ★
    best_loss=float("inf"); wait=0; patience,delta=3,1e-3
    for ep in range(1,args.epochs+1):
        # ---------- Train ----------
        model.train(); run=correct=tot=0
        for batch in tl:
            if batch is None: continue
            x,y=batch; x,y=x.to(dev),y.to(dev); tot+=y.size(0)
            for _ in range(args.num_masks):
                x=apply_freq_mask(x,args.freq_mask); x=apply_time_mask(x,args.time_mask)
            x,y_a,y_b,lam=mixup_data(x,y,args.mixup_alpha)
            opt.zero_grad(); out=model(x)
            loss=lam*crit(out,y_a)+(1-lam)*crit(out,y_b) if y_b is not None else crit(out,y)
            loss.backward(); opt.step(); run+=loss.item()*x.size(0)
            preds=out.argmax(1); correct+=(preds==y).sum().item()
        train_loss=run/len(tl.dataset); train_acc=correct/tot
        history["train_loss"].append(train_loss)                    # ★

        # ---------- Val -------------
        model.eval(); run=correct=tot=0
        with torch.no_grad():
            for batch in vl:
                if batch is None: continue
                x,y=batch; x,y=x.to(dev),y.to(dev); out=model(x)
                run+=crit(out,y).item()*x.size(0)
                correct+=(out.argmax(1)==y).sum().item(); tot+=y.size(0)
        val_loss, val_acc = run/len(vl.dataset), correct/tot
        history["val_loss"].append(val_loss)                        # ★

        # ---------- 로그 & CSV Append ----------
        print(f"[{ep}/{args.epochs}] "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

        pd.DataFrame([{
            "exp_idx": args.exp_idx,
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }]).to_csv(epoch_csv,
                   mode='a',
                   header=ep==1 and not os.path.exists(epoch_csv),           # ★
                   index=False)

        # ---------- Early Stop ----------
        if best_loss - val_loss > delta:
            best_loss, best_state, wait = val_loss, model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep} (best val_loss={best_loss:.3f})")
                break

    if best_state: model.load_state_dict(best_state)

    # ---------- Test ----------
    model.eval(); run=correct=tot=0; y_true=[]; y_pred=[]
    with torch.no_grad():
        for batch in sl:
            if batch is None: continue
            x,y=batch; x,y=x.to(dev),y.to(dev); out=model(x)
            run+=crit(out,y).item()*x.size(0)
            p=out.argmax(1); correct+=(p==y).sum().item(); tot+=y.size(0)
            y_true+=y.cpu().tolist(); y_pred+=p.cpu().tolist()
    test_loss, test_acc = run/len(sl.dataset), correct/tot

    # ---------- 시각화 파일 저장 --------------------------------------- ★
    plot_loss_curve(history, os.path.join(RESULTS_DIR,f"{tag}_loss.png"))
    plot_confmat(y_true, y_pred, os.path.join(RESULTS_DIR,f"{tag}_cm.png"))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR,f"{tag}.pt"))
    return dict(exp_idx=args.exp_idx, dropout=args.dropout,
                pos_weight=args.pos_weight, lr=args.lr,
                val_loss=best_loss, val_acc=val_acc,
                test_loss=test_loss, test_acc=test_acc)

# ❺ CLI & 그리드 --------------------------------------------------------
def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_name',default='efflite0'); ap.add_argument('--exp_idx',type=int,default=1)
    ap.add_argument('--epochs',type=int,default=30);    ap.add_argument('--batch_size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=1e-3);    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--threshold',type=float,default=0.5); ap.add_argument('--pos_weight',type=float,default=1.4)
    ap.add_argument('--dropout',type=float,default=0.3);   ap.add_argument('--freq_mask',type=int,default=10)
    ap.add_argument('--time_mask',type=int,default=10);    ap.add_argument('--num_masks',type=int,default=1)
    ap.add_argument('--mixup_alpha',type=float,default=0.2)
    return ap

if __name__ == "__main__":
    parser=get_parser()
    if len(sys.argv)>1:                 # 단일 실험
        args=parser.parse_args()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        train_and_eval(args)
    else:                               # 그리드
        dropouts=[0.3,0.4]; lrs=[1e-3,5e-4]; pos_w=[1.4,1.55]
        grid=itertools.product(dropouts,lrs,pos_w,[42])
        csv_path=os.path.join(RESULTS_DIR,"efflite0_grid.csv")
        total=len(dropouts)*len(lrs)*len(pos_w); print("Total exp:",total)
        exp=1
        for d,lr,pw,_ in grid:
            tag=f"efflite0_exp{exp:03d}.pt"
            if os.path.exists(os.path.join(MODELS_DIR,tag)):
                print(f"[SKIP] exp{exp}"); exp+=1; continue
            arg=parser.parse_args([]); arg.exp_idx=exp
            arg.dropout=d; arg.lr=lr; arg.pos_weight=pw
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            print(f"\n=== EXP {exp}/{total} (drop={d},lr={lr},pw={pw}) ===")
            metrics=train_and_eval(arg)
            pd.DataFrame([metrics]).to_csv(
                csv_path, mode='a',
                header=not os.path.exists(csv_path),
                index=False
            )
            exp+=1
        print("✅ finished; grid CSV →",csv_path)
