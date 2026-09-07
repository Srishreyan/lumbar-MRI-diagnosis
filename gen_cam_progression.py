
import pathlib, argparse, json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from efficientvit_cbam import build_model
from cam_methods import compute_map

ROOT = pathlib.Path(__file__).resolve().parent
CKPT = ROOT/"ckpt_evit_b1.pth"
LABELS = ROOT/"crops"/"labels.csv"
CROPS = ROOT/"crops"/"crops_test"
OUT = ROOT/"figures"/"cam_pfirrmann_progression.png"

# Deterministic 5: lowest sample_id per grade from TEST (same 5 that should match pfirrmann_progression.png)
df=pd.read_csv(LABELS)
samples=[]
for g in range(1,6):
    pool=df[(df["Pfirrmann"]==g)&(df["split"]=="test")].sort_values("sample_id")
    sid=pool.iloc[0]["sample_id"]
    samples.append((g, sid, int(pool.iloc[0]["Pfirrmann"])) )
print("Samples:", samples)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=build_model(backbone="efficientvit_b1", pretrained=False).to(device)
state=torch.load(CKPT, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded {CKPT} on {device}, Pfirrmann head in")

# Hook target: last conv before CBAM. In EfficientViT it is encoder.stages[-1]; but cam_methods wants a conv module.
# We use the efficientvit_cbam encoder's final stage as target_conv via feature hook inside cam_methods.CAMBackend
# CAMBackend hooks that module; ensure we pass a nn.Module that has 4D output.
# Use model.encoder.stages[-1]
target_conv = model.encoder.stages[-1]
print("target:", target_conv)

METHODS = ["gradcam","gradcampp","eigencam","scorecam"]
METHOD_LABELS = {"gradcam":"Grad-CAM","gradcampp":"Grad-CAM++","eigencam":"Eigen-CAM","scorecam":"Score-CAM"}

# Compute predictions so we know Pfirrmann target_index per crop (7+pred)
rows=[]
for g, sid, true_pf in samples:
    npy = np.load(CROPS/f"{sid}.npy").astype(np.float32)  # [224,224] 0..1
    x = torch.from_numpy(npy).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).float().to(device)
    with torch.no_grad():
        logits = model(x)  # [1,12]
        pred_pf = int(logits[0,7:].argmax().item())
        # 7 bin + 5 pfir: Pfirrmann logits are indices 7..11, so overall target 7+pred_pf
        target_index = 7 + pred_pf
        prob = torch.softmax(logits[0,7:], dim=0)[pred_pf].item()
    print(f"Grade {g} {sid}: true={true_pf} pred={pred_pf+1} p={prob:.2f} tIdx={target_index}")
    rows.append((g, sid, true_pf, pred_pf+1, target_index, npy, x))

# Build 5x5 grid: row0 = raw crop + grade label, rows 1-4 = overlays (or 5 cols x 5 rows incl raw?)
# Layout: 5 columns (grades I-V), 5 rows (raw, Grad-CAM, Grad-CAM++, Eigen-CAM, Score-CAM) with jet overlay
fig, axes = plt.subplots(5, 5, figsize=(7.6, 7.9), gridspec_kw={"hspace":0.18, "wspace":0.08})
plt.rcParams.update({"font.family":"serif"})

cmap_heat = plt.cm.jet
# Title row
col_titles = ["Grade I","Grade II","Grade III","Grade IV","Grade V"]
for c, t in enumerate(col_titles):
    axes[0,c].set_title(t, fontsize=8, fontweight="bold", pad=6)
    axes[0,c].text(0.5, -0.18, f"{samples[c][1]}", transform=axes[0,c].transAxes, ha="center", va="top", fontsize=5, color="#555555")

row_labels = ["Input crop", "Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Score-CAM"]
for r, lab in enumerate(row_labels):
    axes[r,0].text(-0.22, 0.5, lab, transform=axes[r,0].transAxes, ha="right", va="center", fontsize=7, fontweight="bold" if r==0 else "normal", rotation=90)

for c, (g, sid, true_pf, pred_pf, tIdx, npy, x) in enumerate(rows):
    # row 0: raw
    ax = axes[0,c]
    ax.imshow(npy, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    # annotate true vs pred
    color = "#1a9e1a" if true_pf==pred_pf else "#cc2222"
    ax.text(0.04, 0.96, f"T:{true_pf} P:{pred_pf}", transform=ax.transAxes, fontsize=5.5, color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc=color, ec="none", alpha=0.88), va="top", ha="left")
    ax.text(0.96, 0.06, f"g={true_pf}", transform=ax.transAxes, fontsize=5, color="white",
            bbox=dict(boxstyle="round,pad=0.12", fc="black", ec="none", alpha=0.55), va="bottom", ha="right")

    # rows 1-4: CAM overlays explaining Pfirrmann prediction
    for r, m in enumerate(METHODS, start=1):
        ax = axes[r,c]
        try:
            sal = compute_map(model, target_conv, x, target_index=tIdx, method=m, spatial=224, max_channels=64 if m=="scorecam" else None)[0]  # [224,224]
        except Exception as e:
            print(f"CAM error {m} {sid}: {e}")
            import traceback; traceback.print_exc()
            sal = np.zeros((224,224), dtype=np.float32)
        ax.imshow(npy, cmap="gray", vmin=0, vmax=1)
        ax.imshow(sal, cmap="jet", alpha=0.48, vmin=0, vmax=1)
        ax.axis("off")
        # small correctness dot
        dot_c = "#1a9e1a" if true_pf==pred_pf else "#cc2222"
        ax.add_patch(plt.Circle((212, 12), 5, color=dot_c, ec="white", lw=0.6, zorder=5))

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.06, 0.015, 0.84])
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_heat, norm=norm, orientation="vertical")
cb.set_label("Saliency", fontsize=7)
cb.ax.tick_params(labelsize=6)

fig.suptitle("Pfirrmann saliency -- four CAM methods on the five progression crops (target = predicted Pfirrmann grade)", fontsize=7.5, y=0.995)
# footnote
fig.text(0.01, 0.005, "EfficientViT-b1 (ckptₑvitₛb₁, 7.58 M). Green dot = Pfirrmann correct, red = wrong. Overlays computed on model.encoder.stages[-1] ([1,256,7,7]). Score-CAM subsampled to 64 channels.", fontsize=5.2, color="#555555", ha="left", va="bottom")
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
# also save a JSON manifest
manifest={"samples":[{"grade":g,"sample_id":sid,"true_pfirrmann":true_pf,"pred_pfirrmann":pred_pf} for g,sid,true_pf,pred_pf,_,_,_ in rows], "methods":METHODS, "ckpt": str(CKPT), "target_conv":"model.encoder.stages[-1]", "note":"target_index = 7 + pred_pfirrmann (predicted grade)"}
(ROOT/"figures"/"cam_pfirrmann_progression.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
