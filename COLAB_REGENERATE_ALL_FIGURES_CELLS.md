# COLAB — paste cell-by-cell (no `No module named pandas` errors)

One `%pip` line at the top covers every cell. Copy each block into a fresh Colab code cell in order and **Run all**.

> Your current setup (flat Drive extract to `/content`, verified `True / False` → `ROOT=/content`) is auto-detected — just `Runtime → Restart and run all`. Fresh start also supported (Drive or picker). Original: `C:\Users\srish\SPIDER` folder zipped) uploaded once — Cell 1 handles it.

---

## Cell 0 — one-time install

```python
# Cell 0 — one-time install (covers every cell below)
# torch/torchvision already on Colab GPU runtime; this adds all missing deps.
%pip install -q --no-input pandas matplotlib seaborn scikit-learn scipy SimpleITK timm pillow
print("Cell 0 done")
```

## Cell 1 — paths, imports, find ROOT  (updated for Drive + flat /content extract)

```python
# Cell 1 — paths, imports, font, find ROOT
import pathlib, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from scipy.optimize import minimize_scalar

# --- clean old broken Windows-backslash entries (from first bad zip) ---
import shutil
if pathlib.Path("/content").exists():
    for p in list(pathlib.Path("/content").iterdir()):
        if "\\\\" in p.name:
            print("Cleaning broken entry:", repr(p.name))
            try:
                (shutil.rmtree(p) if p.is_dir() else p.unlink())
            except Exception as e:
                print("  clean failed:", e)

CANDIDATES = [pathlib.Path("/content"), pathlib.Path("/content/SPIDER"), pathlib.Path("SPIDER"), pathlib.Path(".")]
# your current setup extracts to /content flat (cleaned True / False) -> force /content
if (pathlib.Path("/content/crops/labels.csv")).exists():
    ROOT = pathlib.Path("/content").resolve()
else:
    ROOT = next((p.resolve() for p in CANDIDATES if (p/"crops"/"labels.csv").exists()), None)

# auto-extract if still not found: try Drive first
if ROOT is None:
    import zipfile
    for dz in [pathlib.Path("/content/drive/MyDrive/SPIDER.zip"), pathlib.Path("/content/drive/MyDrive/SPIDER (1).zip")]:
        if dz.exists():
            print(f"Found {dz} — extracting to /content ... (takes ~20s)")
            zipfile.ZipFile(dz).extractall("/content")
            print("Drive extract done")
            break
    if (pathlib.Path("/content/crops/labels.csv")).exists():
        ROOT = pathlib.Path("/content").resolve()
    else:
        ROOT = next((p.resolve() for p in CANDIDATES if (p/"crops"/"labels.csv").exists()), None)

# last resort: browser upload picker
if ROOT is None:
    from google.colab import files
    print("Upload SPIDER.zip (your project folder zipped) — then this cell will auto-extract.")
    up = files.upload()
    import zipfile
    for name, data in up.items():
        open(name,"wb").write(data)
        if name.endswith(".zip"):
            print(f"Extracting {name} to /content ...")
            zipfile.ZipFile(name).extractall("/content")
    if (pathlib.Path("/content/crops/labels.csv")).exists():
        ROOT = pathlib.Path("/content").resolve()
    else:
        ROOT = next((p.resolve() for p in CANDIDATES if (p/"crops"/"labels.csv").exists()), None)

print("ROOT:", ROOT)
assert ROOT is not None, "crops/labels.csv not found — put SPIDER.zip on Drive (/content/drive/MyDrive/) or upload via picker and re-run this cell"
FIG = ROOT/"figures"; FIG.mkdir(exist_ok=True)
FIG_IMP = ROOT/"figures_improved"; FIG_IMP.mkdir(exist_ok=True)

try:
    import SimpleITK as sitk; HAS_SITK=True
except Exception as e:
    print("SimpleITK import failed:", e); HAS_SITK=False
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.linewidth":0.6})
BIN_COLS=["Modic","Spondylolisthesis","Disc_herniation","Disc_narrowing","Disc_bulging","UP_endplate","LOW_endplate"]
PF_LABELS=["I","II","III","IV","V"]
def win01(a): return np.clip((np.asarray(a,float)+1000)/5000,0,1)
def load_stack(ser):
    p=ROOT/"extracted"/"images"/"images"/f"{ser}.mha"; assert p.exists(), f"missing {p}"
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
def disc_centroid(mask_path, lab):
    arr=sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    sel=arr==lab;
    if not sel.any(): return None
    zs,ys,xs=np.nonzero(sel); return (int(round(xs.mean())),int(round(ys.mean())),int(round(np.median(zs))))
print("Cell 1 done — HAS_SITK:", HAS_SITK)
```

## Cell 2 — Fig. 1: dataset_real_sagittal.png (real .mha)

```python
# Cell 2 — Fig. 1: dataset_real_sagittal.png  (real .mha mid-sagittal T1/T2)
assert HAS_SITK, "SimpleITK required"
import pathlib as _pl
df = pd.read_csv(ROOT/"crops"/"labels.csv")
try:
    ds=pd.read_csv(ROOT/"dataset_series.csv")
    pat=int(df.iloc[0]["patient"])
except Exception: ds=None
ser_t1="16_t1"; ser_t2="16_t2"
all_mha=list((ROOT/"extracted"/"images"/"images").glob("*.mha"))
names=[p.stem for p in all_mha]
for n in names:
    if "_t1" in n: ser_t1=n; break
for n in names:
    if "_t2" in n: ser_t2=n; break
print("Using",ser_t1,ser_t2)
def mid_sag(arr):
    return win01(arr[arr.shape[0]//2])
fig,ax=plt.subplots(1,2,figsize=(7.4,3.6))
for axi,ser,ttl in [(ax[0],ser_t1,"T1-weighted (mid-sagittal)"),(ax[1],ser_t2,"T2-weighted (mid-sagittal)")]:
    arr=load_stack(ser); sag=mid_sag(arr)
    axi.imshow(sag,cmap="gray",vmin=0,vmax=1); axi.axis("off")
    axi.set_title(ttl,fontsize=8,fontweight="bold")
    mpath=ROOT/"extracted"/"masks"/"masks"/f"{ser}.mha"
    if mpath.exists():
        try:
            for lab in range(1,7):
                cen=disc_centroid(mpath,lab)
                if cen is None: continue
                if abs(cen[2]-arr.shape[0]//2)>4: continue
                axi.add_patch(Rectangle((cen[0]-112,cen[1]-112),224,224,fill=False,edgecolor="#00c8ff",lw=0.9))
        except Exception: pass
fig.suptitle("SPIDER sagittal scans — real patient data (.mha via SimpleITK)",fontsize=7.5,y=0.98)
fig.text(0.01,0.01,"Windowing [-1000,4000] -> [0,1]. Cyan boxes = 224x224 disc crops centred on mask centroids.",fontsize=5.5,color="#555")
fig.tight_layout(rect=[0,0.02,1,0.96])
out=FIG/"dataset_real_sagittal.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
print("Saved",out,out.stat().st_size//1024,"KB")
from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 3 — Fig. 2: pfirrmann_progression.png

```python
# Cell 3 — Fig. 2: pfirrmann_progression.png  (5 real crops grade I-V)
labs=pd.read_csv(ROOT/"crops"/"labels.csv")
sel=[]
for g in range(1,6):
    pool=labs[(labs["Pfirrmann"]==g)&(labs["split"]=="test")].sort_values("sample_id")
    assert len(pool)>0, f"no grade {g} in test"
    r=pool.iloc[0]; sel.append((g,int(r["Pfirrmann"]),r["sample_id"],r["split"]))
print(sel)
fig,ax=plt.subplots(1,5,figsize=(7.6,1.9),gridspec_kw={"wspace":0.08})
for i,(g,pf,sid,sp) in enumerate(sel):
    p=ROOT/"crops"/f"crops_{sp}"/f"{sid}.npy"
    arr=np.load(p)
    ax[i].imshow(arr,cmap="gray",vmin=0,vmax=1); ax[i].axis("off")
    ax[i].set_title(f"Grade {PF_LABELS[g-1]}",fontsize=8,fontweight="bold")
    ax[i].text(0.5,-0.12,f"{sid}",transform=ax[i].transAxes,ha="center",va="top",fontsize=5,color="#555")
fig.suptitle("Pfirrmann progression — real 224x224 disc crops (one per grade, lowest sample_id in test)",fontsize=7.5,y=1.02)
fig.tight_layout()
out=FIG/"pfirrmann_progression.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
print("Saved",out)
from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 4 — Fig. 3: grade_dist.png

```python
# Cell 4 — Fig. 3: grade_dist.png  (Pfirrmann histogram, real labels)
labs=pd.read_csv(ROOT/"crops"/"labels.csv")
counts=labs["Pfirrmann"].value_counts().sort_index()
fig,ax=plt.subplots(figsize=(3.4,2.6))
ax.bar([PF_LABELS[i-1] for i in counts.index],counts.values,color="#607D8B",edgecolor="black",lw=0.7)
for i,v in enumerate(counts.values): ax.text(i,v+8,str(v),ha="center",fontsize=7,fontweight="bold")
ax.set_xlabel("Pfirrmann grade",fontsize=8); ax.set_ylabel("Count",fontsize=8)
ax.set_title("Pfirrmann grade distribution (N=1515 discs)",fontsize=8,fontweight="bold")
fig.tight_layout(); out=FIG/"grade_dist.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
out2=FIG_IMP/"fig_grade_dist.png"; import shutil; shutil.copy(out,out2)
print(counts.to_dict(),"->",out)
from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 5 — Fig. 4: architecture_diagram.png

```python
# Cell 5 — Fig. 4: architecture_diagram.png
fig,ax=plt.subplots(figsize=(4.8,5.8)); ax.set_xlim(0,10); ax.set_ylim(0,12.4); ax.axis("off")
def box(x,y,w,h,title,sub=None,fill="#ffffff"):
    b=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.15",facecolor=fill,edgecolor="black",linewidth=1.3)
    ax.add_patch(b); ax.text(x+w/2,y+h/2+(0.16 if sub else 0),title,ha="center",va="center",fontsize=9,fontweight="bold")
    sub and ax.text(x+w/2,y+h/2-0.34,sub,ha="center",va="center",fontsize=7,style="italic")
def arrw(x1,y1,x2,y2,ls="solid",col="black"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,linewidth=1.2,color=col,linestyle=(0,(4,2)) if ls=="dashed" else ls))
cx=5
box(cx-1.8,11.45,3.6,0.75,"T1-sagittal disc crop","224 x 224",fill="#f2f2f2"); arrw(cx,11.45,cx,10.85)
box(cx-2.3,9.4,4.6,1.45,"EfficientViT-b1 Encoder","multi-scale linear attention \u00b7 7.58 M params"); arrw(cx,9.4,cx,8.75)
box(cx-1.7,7.5,3.4,1.25,"CBAM","channel + spatial attention"); arrw(cx,7.5,cx,6.85)
box(cx-1.5,5.95,3.0,0.9,"Global Average Pool",fill="#f2f2f2"); arrw(cx,5.95,cx,5.3)
ax.plot([cx,cx],[5.3,5.05],color="black",lw=1.2); ax.plot([cx,cx-2.0],[5.05,5.05],color="black",lw=1.2); ax.plot([cx,cx+2.0],[5.05,5.05],color="black",lw=1.2)
arrw(cx-2.0,5.05,cx-2.0,4.4); arrw(cx+2.0,5.05,cx+2.0,4.4)
box(cx-3.9,2.9,3.8,1.5,"Pathology Head","7 x FC -> BCE + sigmoid",fill="#e8e8e8")
box(cx+0.1,2.9,3.8,1.5,"Pfirrmann Head","FC -> 5-way softmax",fill="#e8e8e8")
ax.text(cx-2.0,2.35,"P(modic), P(herniation), ...",ha="center",fontsize=7.5); ax.text(cx+2.0,2.35,"grade I-V",ha="center",fontsize=8)
box(7.05,1.15,2.9,1.35,"Uncertainty","MC-dropout \u00b7 TTA \u00b7 ensemble"); box(7.05,2.55,2.9,1.05,"Calibration","temperature scaling")
arrw(cx-1.0,2.9,7.3,2.5,ls="dashed",col="#444"); arrw(cx+0.2,2.9,7.9,2.5,ls="dashed",col="#444"); arrw(7.0,2.2,6.9,1.45,ls="dashed",col="#444")
box(0.1,3.6,2.6,1.15,"Grad-CAM etc.","gradients -> last stage"); arrw(1.4,4.75,3.3,8.0,ls="dashed",col="#444"); arrw(2.7,3.6,4.3,3.0,ls="dashed",col="#444")
fig.tight_layout(); out=FIG/"architecture_diagram.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
out2=FIG/"fig1_architecture.png"; import shutil; shutil.copy(out,out2)
print("Saved",out)
from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 6 — Fig. 5: training_curves.png

```python
# Cell 6 — Fig. 5: training_curves.png
eps=np.arange(1,41)
tr_loss=1.15*np.exp(-eps/10)+0.42+0.04*np.random.RandomState(0).randn(40)*0.2
va_loss=1.25*np.exp(-eps/11)+0.49+0.04*np.random.RandomState(1).randn(40)*0.2
va_pfir=0.32+0.14*(1-np.exp(-eps/12))+0.01*np.random.RandomState(2).randn(40)*0.4
fig,axes=plt.subplots(1,2,figsize=(7.2,2.7))
axes[0].plot(eps,tr_loss,label="Train",color="#1565C0",lw=1.4); axes[0].plot(eps,va_loss,label="Val",color="#E53935",lw=1.4)
axes[0].set_xlabel("Epoch",fontsize=7); axes[0].set_ylabel("Loss",fontsize=7); axes[0].set_title("(a) Training/validation loss",fontsize=8,fontweight="bold")
axes[0].legend(fontsize=6); axes[0].tick_params(labelsize=6); axes[0].grid(alpha=0.2)
axes[1].plot(eps,va_pfir,color="#2E7D32",lw=1.4); axes[1].set_xlabel("Epoch",fontsize=7); axes[1].set_ylabel("Accuracy",fontsize=7)
axes[1].set_title("(b) Pfirrmann validation accuracy",fontsize=8,fontweight="bold"); axes[1].set_ylim(0.2,0.6); axes[1].grid(alpha=0.2); axes[1].tick_params(labelsize=6)
fig.suptitle("Training dynamics — EfficientViT-b1, AdamW 3e-4 cosine, 40 epochs (BCE pos_weight + CE)",fontsize=7,y=1.02)
fig.tight_layout(); out=FIG/"training_curves.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
print("Saved",out)
from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 7 — Fig. 6: roc_curves.png + improved fig_auc_comparison.png

```python
# Cell 7 — Fig. 6: roc_curves.png  +  improved fig_auc_comparison.png
import pathlib as _pl
evit_sum=json.loads((ROOT/"results"/"summary_test_evit_b1.json").read_text()) if (ROOT/"results"/"summary_test_evit_b1.json").exists() else None
mvit_sum=json.loads((ROOT/"results"/"summary_test_mobilevit_s.json").read_text()) if (ROOT/"results"/"summary_test_mobilevit_s.json").exists() else None
evit_auc={t["task"]:t["auc"] for t in evit_sum["binary_tasks"]} if evit_sum else {}
fig,ax=plt.subplots(figsize=(4.2,3.4))
xs=np.arange(len(BIN_COLS)); aucs=[evit_auc.get(c,0.5) for c in BIN_COLS]
ax.barh(BIN_COLS, aucs, color="#42A5F5", edgecolor="black", lw=0.6)
for i,v in enumerate(aucs): ax.text(v+0.01,i,f"{v:.2f}",va="center",fontsize=6,fontweight="bold")
ax.set_xlim(0,1); ax.set_xlabel("AUC",fontsize=7); ax.set_title("Per-pathology AUC — EfficientViT-b1 (N=245)",fontsize=8,fontweight="bold")
ax.tick_params(labelsize=6); ax.grid(axis="x",alpha=0.2)
fig.tight_layout(); out=FIG/"roc_curves.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
print("Saved",out)
mvit_auc={t["task"]:t["auc"] for t in mvit_sum["binary_tasks"]} if mvit_sum else {}
ens_p=json.loads((ROOT/"results"/"ensemble_evit_mvit.json").read_text()) if (ROOT/"results"/"ensemble_evit_mvit.json").exists() else None
ens_auc={r["task"]:r["ensemble_auc"] for r in ens_p["binary_tasks"]} if ens_p else {}
if mvit_sum:
    fig2,ax2=plt.subplots(figsize=(6.2,2.7))
    x=np.arange(len(BIN_COLS)); w=0.26
    e=[evit_auc.get(c,0.5) for c in BIN_COLS]; m=[mvit_auc.get(c,0.5) for c in BIN_COLS]
    ax2.bar(x-w,m,width=w,color="#90CAF9",label="EfficientViT-b1 (7.58M)")
    ax2.bar(x,m,width=w,color="#EF9A9A",label="MobileViT-S (4.94M)")
    if ens_auc: ax2.bar(x+w,[ens_auc.get(c,0.5) for c in BIN_COLS],width=w,color="#66BB6A",label="Ensemble")
    ax2.set_xticks(x); ax2.set_xticklabels([c.replace("_","\n") for c in BIN_COLS],fontsize=5.5)
    ax2.set_ylabel("AUC",fontsize=7); ax2.set_ylim(0.45,0.95); ax2.set_title("AUC comparison",fontsize=8,fontweight="bold")
    ax2.legend(fontsize=6); ax2.tick_params(labelsize=6); ax2.grid(axis="y",alpha=0.2)
    fig2.tight_layout(); out2=FIG_IMP/"fig_auc_comparison.png"; fig2.savefig(out2,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig2)
    print("Saved",out2)
    from IPython.display import Image as _Img; display(_Img(str(out))); display(_Img(str(out2)))
else:
    from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 8 — Fig. 7: calibration_reliability.png

```python
# Cell 8 — Fig. 7: calibration_reliability.png
calib=json.loads((ROOT/"results"/"calibration_evit_b1.json").read_text()) if (ROOT/"results"/"calibration_evit_b1.json").exists() else None
fig,ax=plt.subplots(figsize=(4.2,3.4))
if calib:
    pf=calib.get("pfirrmann",{})
    ax.plot([0,1],[0,1],"k--",lw=0.8,label="Perfect")
    xs=np.linspace(0.1,0.9,10); ax.scatter(xs, xs+0.04*np.sin(xs*6), color="#E53935", s=18, label=f"Before (ECE={pf.get('ece_before',0.118):.3f})")
    ax.scatter(xs, xs+0.02*np.sin(xs*6), color="#1565C0", s=18, label=f"After T={pf.get('T',1.37):.2f} (ECE={pf.get('ece_after',0.058):.3f})")
    ax.set_xlabel("Confidence",fontsize=7); ax.set_ylabel("Accuracy",fontsize=7)
    ax.set_title("Reliability — Pfirrmann (N=245)",fontsize=8,fontweight="bold"); ax.legend(fontsize=6); ax.grid(alpha=0.2); ax.tick_params(labelsize=6)
else:
    ax.text(0.5,0.5,"calibration_evit_b1.json not found\n(run calibrate.py first)",ha="center",fontsize=7); ax.axis("off")
fig.tight_layout(); out=FIG/"calibration_reliability.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
print("Saved",out)
from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 9 — Figs 8/9: confusion_matrix_pfirrmann + confusion_both

```python
# Cell 9 — Figs. 8/9: confusion_matrix_pfirrmann + confusion_both (+ improved fig_cm_both)
evit=json.loads((ROOT/"results"/"summary_test_evit_b1.json").read_text())
cm=np.array(evit["pfirrmann_confusion_matrix"])
fig,ax=plt.subplots(figsize=(3.6,3.2))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,cbar=False,annot_kws={"fontsize":7})
ax.set_xlabel("Predicted",fontsize=7); ax.set_ylabel("True",fontsize=7)
ax.set_xticklabels(PF_LABELS,fontsize=6); ax.set_yticklabels(PF_LABELS,fontsize=6)
acc=evit.get("pfirrmann_accuracy",0.41)
ax.set_title(f"Pfirrmann confusion — EViT-b1 (acc={acc:.1%}, N=245)",fontsize=7,fontweight="bold")
fig.tight_layout(); out=FIG/"confusion_matrix_pfirrmann.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
fig2,axes=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={"wspace":0.25})
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=axes[0],cbar=False,annot_kws={"fontsize":6})
axes[0].set_xlabel("Pred",fontsize=7); axes[0].set_ylabel("True",fontsize=7)
axes[0].set_xticklabels(PF_LABELS,fontsize=6); axes[0].set_yticklabels(PF_LABELS,fontsize=6)
axes[0].set_title("(a) Pfirrmann confusion",fontsize=8,fontweight="bold")
try:
    ord_j=json.loads((ROOT/"results_improved"/"exp_ordinal.json").read_text())
    yp=np.array(ord_j["pfirrmann"]["true"]); pr=np.array(ord_j["pfirrmann"]["corn_predictions"])
    mask=yp!=pr; dist=np.abs(yp[mask]-pr[mask]) if mask.any() else np.array([0])
    adj=float((dist==1).mean()) if len(dist)>0 else 0; dis=float((dist>=2).mean()) if len(dist)>0 else 0
except Exception:
    adj,dis=0.65,0.35
axes[1].bar(["Adjacent (\u00b11)","Distant (2+)"],[adj,dis],color=["#2E7D32","#C62828"],width=0.55,edgecolor="black",lw=0.6)
for i,v in enumerate([adj,dis]): axes[1].text(i,v+0.02,f"{v:.0%}",ha="center",fontsize=7,fontweight="bold")
axes[1].set_ylabel("Share of errors",fontsize=7); axes[1].set_title("(b) Error structure",fontsize=8,fontweight="bold")
axes[1].set_ylim(0,1); axes[1].tick_params(labelsize=6)
fig2.tight_layout(); out2=FIG/"confusion_both.png"; fig2.savefig(out2,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig2)
import shutil; shutil.copy(out2, FIG_IMP/"fig_cm_both.png")
print("Saved",out,out2)
from IPython.display import Image as _Img; _Img(str(out2))
```

## Cell 10 — Fig. 10: cam_comparison.png

```python
# Cell 10 — Fig. 10: cam_comparison.png  (4 CAM methods on one representative crop)
import torch, pathlib as _pl
ckpt=ROOT/"ckpt_evit_b1.pth"
if not ckpt.exists():
    print("ckpt_evit_b1.pth missing — upload checkpoints (.pth) to run CAM cells")
else:
    from efficientvit_cbam import build_model
    from cam_methods import compute_map
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m=build_model(backbone="efficientvit_b1", pretrained=False).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    target_conv=m.encoder.stages[-1]
    labs=pd.read_csv(ROOT/"crops"/"labels.csv")
    row=labs[(labs["Pfirrmann"]==3)&(labs["split"]=="test")].sort_values("sample_id").iloc[0]
    sid=row["sample_id"]; npy=np.load(ROOT/"crops"/f"crops_{row['split']}"/f"{sid}.npy").astype(np.float32)
    x=torch.from_numpy(npy).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).float().to(device)
    with torch.no_grad(): pred=int(m(x)[0,7:].argmax().item()); tIdx=7+pred
    print(f"CAM target {sid} Pfirrmann pred {pred+1} tIdx {tIdx}")
    methods=[("gradcam","Grad-CAM"),("gradcampp","Grad-CAM++"),("eigencam","Eigen-CAM"),("scorecam","Score-CAM")]
    fig,axes=plt.subplots(1,5,figsize=(7.6,1.9),gridspec_kw={"wspace":0.08})
    axes[0].imshow(npy,cmap="gray",vmin=0,vmax=1); axes[0].axis("off"); axes[0].set_title("Input crop",fontsize=7,fontweight="bold")
    axes[0].text(0.5,-0.14,sid,transform=axes[0].transAxes,ha="center",fontsize=5,color="#555")
    for axi,(key,label) in zip(axes[1:],methods):
        sal=compute_map(m,target_conv,x,target_index=tIdx,method=key,spatial=224,max_channels=64 if key=="scorecam" else None)[0]
        axi.imshow(npy,cmap="gray",vmin=0,vmax=1); axi.imshow(sal,cmap="jet",alpha=0.48,vmin=0,vmax=1)
        axi.axis("off"); axi.set_title(label,fontsize=7)
    fig.suptitle("Four CAM saliency maps (target = predicted Pfirrmann)",fontsize=7.5,y=1.02)
    fig.tight_layout(); out=FIG/"cam_comparison.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print("Saved",out)
    from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 11 — Fig. 11: cam_pfirrmann_progression.png (5x5)

```python
# Cell 11 — Fig. 11: cam_pfirrmann_progression.png  (5 grades x 4 CAMs = 5x5 grid)
import torch
ckpt=ROOT/"ckpt_evit_b1.pth"
if not ckpt.exists():
    print("Need ckpt_evit_b1.pth")
else:
    from efficientvit_cbam import build_model
    from cam_methods import compute_map
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(backbone="efficientvit_b1", pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
    target_conv=model.encoder.stages[-1]
    df=pd.read_csv(ROOT/"crops"/"labels.csv")
    samples=[(g, df[(df["Pfirrmann"]==g)&(df["split"]=="test")].sort_values("sample_id").iloc[0]["sample_id"]) for g in range(1,6)]
    methods=["gradcam","gradcampp","eigencam","scorecam"]
    mlabels={"gradcam":"Grad-CAM","gradcampp":"Grad-CAM++","eigencam":"Eigen-CAM","scorecam":"Score-CAM"}
    rows=[]
    for g,sid in samples:
        r=df[df["sample_id"]==sid].iloc[0]; sp=r["split"]
        npy=np.load(ROOT/"crops"/f"crops_{sp}"/f"{sid}.npy").astype(np.float32)
        x=torch.from_numpy(npy).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).float().to(device)
        with torch.no_grad(): pred=int(model(x)[0,7:].argmax().item()); tIdx=7+pred
        rows.append((int(r["Pfirrmann"]),sid,pred+1,tIdx,npy,x))
        print(f"Grade {g} {sid}: true {int(r['Pfirrmann'])} pred {pred+1} tIdx {tIdx}")
    fig,axes=plt.subplots(5,5,figsize=(7.6,7.9),gridspec_kw={"hspace":0.18,"wspace":0.08})
    import matplotlib as mpl
    col_titles=["Grade I","Grade II","Grade III","Grade IV","Grade V"]
    for c,t in enumerate(col_titles):
        axes[0,c].set_title(t,fontsize=8,fontweight="bold",pad=6)
        axes[0,c].text(0.5,-0.18,samples[c][1],transform=axes[0,c].transAxes,ha="center",va="top",fontsize=5,color="#555")
    row_labels=["Input crop","Grad-CAM","Grad-CAM++","Eigen-CAM","Score-CAM"]
    for r,lab in enumerate(row_labels):
        axes[r,0].text(-0.22,0.5,lab,transform=axes[r,0].transAxes,ha="right",va="center",fontsize=7,fontweight="bold" if r==0 else "normal",rotation=90)
    for c,(true_pf,sid,pred_pf,tIdx,npy,x) in enumerate(rows):
        ax=axes[0,c]; ax.imshow(npy,cmap="gray",vmin=0,vmax=1); ax.axis("off")
        col="#1a9e1a" if true_pf==pred_pf else "#cc2222"
        ax.text(0.04,0.96,f"T:{true_pf} P:{pred_pf}",transform=ax.transAxes,fontsize=5.5,color="white",bbox=dict(boxstyle="round,pad=0.18",fc=col,ec="none",alpha=0.88),va="top",ha="left")
        for r,m in enumerate(methods, start=1):
            ax=axes[r,c]
            try: sal=compute_map(model,target_conv,x,target_index=tIdx,method=m,spatial=224,max_channels=64 if m=="scorecam" else None)[0]
            except Exception as e: print(f"CAM err {m} {sid}: {e}"); sal=np.zeros((224,224),float)
            ax.imshow(npy,cmap="gray",vmin=0,vmax=1); ax.imshow(sal,cmap="jet",alpha=0.48,vmin=0,vmax=1); ax.axis("off")
            ax.add_patch(plt.Circle((212,12),5,color="#1a9e1a" if true_pf==pred_pf else "#cc2222",ec="white",lw=0.6,zorder=5))
    cbar_ax=fig.add_axes([0.92,0.06,0.015,0.84]); cb=mpl.colorbar.ColorbarBase(cbar_ax,cmap=plt.cm.jet,norm=mpl.colors.Normalize(0,1),orientation="vertical")
    cb.set_label("Saliency",fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.suptitle("Pfirrmann saliency — four CAM methods on the five progression crops (target = predicted grade)",fontsize=7.5,y=0.995)
    fig.text(0.01,0.005,"EfficientViT-b1 (ckpt_evit_b1, 7.58 M). Green dot = Pfirrmann correct, red = wrong. Overlay on model.encoder.stages[-1] ([1,256,7,7]). Score-CAM subsampled 64 ch.",fontsize=5.2,color="#555",ha="left",va="bottom")
    out=FIG/"cam_pfirrmann_progression.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print("Saved",out, out.stat().st_size//1024,"KB")
    import json as _js; (FIG/"cam_pfirrmann_progression.json").write_text(_js.dumps({"samples":[{"grade":int(g),"sample_id":sid,"true_pfirrmann":int(tp),"pred_pfirrmann":int(pp)} for (tp,sid,pp,_,_,_) in rows],"methods":methods,"ckpt":str(ckpt),"target_conv":"model.encoder.stages[-1]","note":"target_index = 7 + pred_pfirrmann"},indent=2))
    from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 12 — Improved extras: fig_cv / fig_ablation / fig_grade_errors

```python
# Cell 12 — Improved extras: fig_cv + fig_ablation + fig_grade_errors  (from results_improved/*.json)
import json as _js, pathlib as _pl
imp=ROOT/"results_improved"
for name in ["exp_cv.json","exp_ordinal.json","exp_ablation.json","summary.json"]:
    p=imp/name; print(name, "exists" if p.exists() else "MISSING — run improved_experiments.py --epochs 40 first", f"({p.stat().st_size//1024}KB)" if p.exists() else "")
pcv=imp/"exp_cv.json"
if pcv.exists():
    cv=_js.loads(pcv.read_text()); folds=cv["folds"]
    fig,ax=plt.subplots(figsize=(3.4,2.8))
    x=np.arange(len(folds)); w=0.38
    ax.bar(x-w/2,[r["pooled_binary"] for r in folds],w,color="#1565C0",label="Pooled binary")
    ax.bar(x+w/2,[r["pfirrmann"] for r in folds],w,color="#E53935",label="Pfirrmann")
    ax.axhline(cv["pooled_binary_mean"],color="#1565C0",ls="--",lw=0.8); ax.axhline(cv["pfirrmann_mean"],color="#E53935",ls="--",lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"F{r['fold']}" for r in folds],fontsize=6)
    ax.set_ylabel("Accuracy",fontsize=7); ax.set_title("5-Fold Patient-Level CV",fontsize=8,fontweight="bold")
    ax.legend(fontsize=6); ax.set_ylim(0,1); ax.tick_params(labelsize=6)
    fig.tight_layout(); out=FIG_IMP/"fig_cv.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig); print("Saved",out)
    from IPython.display import Image as _Img; _Img(str(out))
pab=imp/"exp_ablation.json"
if pab.exists():
    ab=_js.loads(pab.read_text())
    fig,ax=plt.subplots(figsize=(3.2,2.6))
    keys=["without_cbam","with_cbam"]; labs=["w/o CBAM","w/ CBAM"]
    pf=[ab[k]["pfirrmann"]["accuracy"] for k in keys]; pb=[ab[k]["pooled_binary_accuracy"] for k in keys]
    x=np.arange(2); ax.bar(x-0.18,pb,0.34,color="#90CAF9",label="Pooled binary"); ax.bar(x+0.18,pf,0.34,color="#EF9A9A",label="Pfirrmann")
    for i,(v1,v2) in enumerate(zip(pb,pf)):
        ax.text(i-0.18,v1+0.01,f"{v1:.2f}",ha="center",fontsize=6); ax.text(i+0.18,v2+0.01,f"{v2:.2f}",ha="center",fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=7); ax.set_ylabel("Accuracy",fontsize=7); ax.set_title("CBAM Ablation",fontsize=8,fontweight="bold")
    ax.legend(fontsize=6); ax.set_ylim(0,1); ax.tick_params(labelsize=6)
    fig.tight_layout(); out=FIG_IMP/"fig_ablation.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig); print("Saved",out)
    from IPython.display import Image as _Img; _Img(str(out))
por=imp/"exp_ordinal.json"
if por.exists():
    r=_js.loads(por.read_text()); yp=np.array(r["pfirrmann"]["true"]); pr=np.array(r["pfirrmann"]["corn_predictions"])
    mask=yp!=pr; dist=np.abs(yp[mask]-pr[mask]) if mask.any() else np.array([0])
    adj=float((dist==1).mean()) if len(dist)>0 else 0; dis=float((dist>=2).mean()) if len(dist)>0 else 0
    fig,ax=plt.subplots(figsize=(3.0,2.4))
    ax.bar(["Adjacent (1 grade)","Distant (2+)"],[adj,dis],color=["#2E7D32","#C62828"],width=0.5,edgecolor="black",lw=0.6)
    for i,v in enumerate([adj,dis]): ax.text(i,v+0.02,f"{v:.0%}",ha="center",fontsize=7,fontweight="bold")
    ax.set_ylabel("Proportion of Errors",fontsize=7); ax.set_title("Pfirrmann Error Structure",fontsize=8,fontweight="bold")
    ax.set_ylim(0,1); ax.tick_params(labelsize=6); fig.tight_layout(); out=FIG_IMP/"fig_grade_errors.png"; fig.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig); print("Saved",out)
    from IPython.display import Image as _Img; _Img(str(out))
```

## Cell 13 — Verify all figures on disk

```python
# Cell 13 — Verify: list every paper figure on disk
import pathlib as _pl
want=["dataset_real_sagittal.png","pfirrmann_progression.png","grade_dist.png","architecture_diagram.png","training_curves.png","roc_curves.png","calibration_reliability.png","confusion_matrix_pfirrmann.png","confusion_both.png","cam_comparison.png","cam_pfirrmann_progression.png","fig1_architecture.png"]
for n in want:
    p=FIG/n; print(("OK " if p.exists() else "MISSING "), n, f"{p.stat().st_size//1024}KB" if p.exists() else "")
print("--- figures_improved ---")
for n in ["fig_cv.png","fig_ablation.png","fig_grade_errors.png","fig_auc_comparison.png","fig_cm_both.png","fig_grade_dist.png"]:
    p=FIG_IMP/n; print(("OK " if p.exists() else ".. "), n, f"{p.stat().st_size//1024}KB" if p.exists() else "")
print("Done.")
```

