import numpy as np
import pandas as pd
import cv2
from PIL import Image
from io import BytesIO
from sklearn.cluster import DBSCAN
from skimage.metrics import structural_similarity as ssim
import re  # <-- added

EMB_FILE = "logo_dinov2_embeddings.parquet"
LOGO_DIR = "logo_images"
OUT_HTML = "logo_clusters.html"

# ---------------- CONFIG ----------------
DBSCAN_EPS = 0.18
DBSCAN_MIN_SAMPLES = 2

SSIM_THRESHOLD = 0

COS_SIM_THRESHOLD = 0.85   # icon/global cosine
SSIM_SIZE = (256,256)

# ---------------- UTILS ----------------
def safe_filename(domain: str) -> str:  # <-- added (tiny helper for HTML paths)
    return re.sub(r"[^a-zA-Z0-9._-]", "_", domain)

def composite_alpha_to_white(img_rgba: Image.Image) -> Image.Image:  # <-- added (you used it)
    img_rgba = img_rgba.convert("RGBA")
    bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img_rgba).convert("RGB")

# ---------------- LOAD ----------------
df = pd.read_parquet(EMB_FILE).sort_index()
domains = list(df.index)
print(domains)

X_icon = np.stack(df["emb_icon"].values)
X_global = np.stack(df["emb_global"].values)

X_icon /= np.linalg.norm(X_icon, axis=1, keepdims=True)
X_global /= np.linalg.norm(X_global, axis=1, keepdims=True)


def make_graph(X):
    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine"
    ).fit_predict(X)

    dfc = pd.DataFrame({"domain": domains, "cluster": labels})
    for cid in sorted(dfc["cluster"].unique()):
        if cid == -1:
            continue

        group = dfc[dfc["cluster"] == cid]["domain"].tolist()

        for d in group:
            for o in group:
                if (d == o):
                    continue
                i1 = domains.index(d)
                i2 = domains.index(o)

                ok = False
                mx = max(np.dot(X_icon[i1], X_icon[i2]), np.dot(X_global[i1], X_global[i2]))
                mn = min(np.dot(X_icon[i1], X_icon[i2]), np.dot(X_global[i1], X_global[i2]))
               
                score = 0.85 * mx + 0.15 * mn
                if "nestle" in d and "nestle" in o:
                    print(score)
                if (score >= 0.85):
                    ok = True

                if ok:
                    G.add_edge(i1, i2)




# ---------------- STRICT SSIM REFINEMENT ----------------

class Graph:
    def __init__(self, n: int):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.vis = [False] * n

    def add_edge(self, u: int, v: int):
        self.adj[u].append(v)
        self.adj[v].append(u)

    def dfs(self, u: int, comp: list):
        self.vis[u] = True
        comp.append(u)
        for v in self.adj[u]:
            if not self.vis[v]:
                self.dfs(v, comp)

    def connected_components(self):
        comps = []
        for i in range(self.n):
            if not self.vis[i]:
                comp = []
                self.dfs(i, comp)
                comps.append(comp)
        return comps

final_clusters = []

G = Graph(len(domains))

make_graph(X_icon)
make_graph(X_global)


final_clusters = G.connected_components()
for i in range(len(final_clusters)):
    final_clusters[i] = [domains[idx] for idx in final_clusters[i]]

final_clusters = sorted(final_clusters, key=len, reverse=True)


# ---------------- HTML ----------------
html = "<html><body><h1>DINOv2 Logo Clusters (icon-aware)</h1>"

for i,c in enumerate(final_clusters):
    html += f"<h2>Cluster {i} ({len(c)})</h2><div>"
    for d in c:
        html += f"<div style='display:inline-block;margin:10px;text-align:center'>"
        # use the same safe naming you used when saving images
        html += f"<img src='{LOGO_DIR}/{safe_filename(d)}.png' width='100'><br>{d}</div>"
    html += "</div><hr>"

html += "</body></html>"

with open(OUT_HTML, "w") as f:
    f.write(html)

print("Clusters written to", OUT_HTML)
