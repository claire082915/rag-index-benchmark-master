import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Data ---
data = [
    {"recall": 0.0,   "em": 5.6,  "f1": 7.6,  "prec": 7.8,  "rec": 9.2},
    {"recall": 0.2,   "em": 15.9, "f1": 19.7, "prec": 20.3, "rec": 21.0},
    {"recall": 0.4,   "em": 25.4, "f1": 32.4, "prec": 33.1, "rec": 34.3},
    {"recall": 0.6,   "em": 39.0, "f1": 48.1, "prec": 49.8, "rec": 48.8},
    {"recall": 0.8,   "em": 51.8, "f1": 64.1, "prec": 66.2, "rec": 65.1},
    {"recall": 0.9,   "em": 57.3, "f1": 71.6, "prec": 74.1, "rec": 72.7},
    {"recall": 0.95,  "em": 62.4, "f1": 77.2, "prec": 79.6, "rec": 78.5},
    {"recall": 1.0,   "em": 67.5, "f1": 82.9, "prec": 85.7, "rec": 84.0},
]

x     = [d["recall"] for d in data]
em    = [d["em"]     for d in data]
f1    = [d["f1"]     for d in data]
prec  = [d["prec"]   for d in data]
rec   = [d["rec"]    for d in data]

COLORS = {
    "em":   "#1D9E75",
    "f1":   "#185FA5",
    "prec": "#BA7517",
    "rec":  "#534AB7",
}

def add_trendline(ax, x_vals, y_vals, color):
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(min(x_vals), max(x_vals), 200)
    ax.plot(x_line, m * x_line + b, color=color, linewidth=1.2,
            linestyle="--", alpha=0.5)

def style_ax(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=9)
    ax.yaxis.label.set_color("#555555")
    ax.xaxis.label.set_color("#555555")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("white")
fig.suptitle("Document Retrieval Recall vs. HotpotQA Scores (1000 queries)",
             fontsize=13, color="#333333", y=1.01)

# --- Left: full range ---
for y_vals, label, key in [
    (em,   "EM (Exact Match)",  "em"),
    (f1,   "F1 (Token Overlap)", "f1"),
    (prec, "Precision",         "prec"),
    (rec,  "Recall (Answer)",   "rec"),
]:
    ax1.scatter(x, y_vals, color=COLORS[key], s=40, zorder=3, label=label)
    add_trendline(ax1, x, y_vals, COLORS[key])

ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(0, 100)
ax1.set_xlabel("Document Retrieval Recall (fraction of required gold docs retrieved)", fontsize=10)
ax1.set_ylabel("QA Score (%)", fontsize=10)
ax1.set_title("Full range (retrieval recall: 0 → 1)", fontsize=10, color="#555555", pad=8)
ax1.legend(fontsize=9, frameon=False, ncol=1,
           loc="upper left", labelcolor="#444444")
style_ax(ax1)

# --- Right: zoomed 0.8–1.0 ---
zoom_mask = [d["recall"] >= 0.8 for d in data]
x_z    = [v for v, m in zip(x,    zoom_mask) if m]
em_z   = [v for v, m in zip(em,   zoom_mask) if m]
f1_z   = [v for v, m in zip(f1,   zoom_mask) if m]
prec_z = [v for v, m in zip(prec, zoom_mask) if m]
rec_z  = [v for v, m in zip(rec,  zoom_mask) if m]

for y_vals, label, key in [
    (em_z,   "EM (Exact Match)",   "em"),
    (f1_z,   "F1 (Token Overlap)", "f1"),
    (prec_z, "Precision",          "prec"),
    (rec_z,  "Recall (Answer)",    "rec"),
]:
    ax2.scatter(x_z, y_vals, color=COLORS[key], s=40, zorder=3, label=label)
    add_trendline(ax2, x_z, y_vals, COLORS[key])

ax2.set_xlim(0.785, 1.015)
ax2.set_ylim(50, 90)
ax2.set_xlabel("Document Retrieval Recall (fraction of required gold docs retrieved)", fontsize=10)
ax2.set_ylabel("QA Score (%)", fontsize=10)
ax2.set_title("Zoomed in (retrieval recall: 0.8 → 1.0)", fontsize=10, color="#555555", pad=8)
ax2.legend(fontsize=9, frameon=False, ncol=1,
           loc="upper left", labelcolor="#444444")
style_ax(ax2)

plt.tight_layout()
plt.savefig("hotpot_recall_vs_scores.png", dpi=150, bbox_inches="tight")
print("Saved to hotpot_recall_vs_scores.png")
plt.show()