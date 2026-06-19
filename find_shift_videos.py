import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Paths to the feature cache files
val_cache_path = "/tf/data/vjepa2/cache_features/val_val_b5d1d90d_strat_human_vjepa_nf_16_fs_4_ht_0.3_wt_0.41.pkl"
test_cache_path = "/tf/data/vjepa2/cache_features/test_test-dataset_c98ab4d7_strat_human_vjepa_nf_16_fs_4_ht_0.3_wt_0.41.pkl"

print("Loading validation features...")
with open(val_cache_path, "rb") as f:
    val_data = pickle.load(f)

print("Loading test features...")
with open(test_cache_path, "rb") as f:
    test_data = pickle.load(f)

X_val, y_val, val_meta = val_data["X"], val_data["y"], val_data["metadata"]
X_test, y_test, test_meta = (
    test_data["X"],
    test_data["y"],
    test_data["metadata"],
)


# 2. Helper to pool clips to video level (V-JEPA evaluation convention)
def aggregate_clips_to_videos(X, y, metadata, pool="mean"):
    groups = {}
    order = []
    for i, m in enumerate(metadata):
        vp = m["video"]
        if vp not in groups:
            groups[vp] = []
            order.append(vp)
        groups[vp].append(i)

    feats = []
    labels = []
    meta = []
    for vp in order:
        idxs = groups[vp]
        clip_feats = X[idxs]
        if pool == "max":
            pooled = clip_feats.max(axis=0)
        else:
            pooled = clip_feats.mean(axis=0)
        lbl = int(round(float(np.mean(y[idxs]))))
        feats.append(pooled)
        labels.append(lbl)
        meta.append({"video": vp, "ground_truth": lbl})
    return np.stack(feats, axis=0), np.array(labels), meta


print("\nAggregating clips to video level...")
X_val_v, y_val_v, val_meta_v = aggregate_clips_to_videos(
    X_val, y_val, val_meta
)
X_test_v, y_test_v, test_meta_v = aggregate_clips_to_videos(
    X_test, y_test, test_meta
)

print(
    f"Validation: {X_val_v.shape[0]} videos | Test: {X_test_v.shape[0]} videos"
)

# 3. Standardize on the combined set
print("\nStandardizing combined feature space...")
scaler = StandardScaler()
X_all_v = scaler.fit_transform(np.vstack([X_val_v, X_test_v]))
X_val_v_scaled = X_all_v[: len(X_val_v)]
X_test_v_scaled = X_all_v[len(X_val_v) :]

# 4. Project into shared PCA(2) space
print("Projecting features to PCA (n_components=2)...")
pca = PCA(n_components=2, random_state=42)
pca.fit(X_all_v)
pcs_val = pca.transform(X_val_v_scaled)
pcs_test = pca.transform(X_test_v_scaled)

print(
    f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}"
)

# 5. Compute the global centroid of the Validation Domain
c_val = pcs_val.mean(axis=0)
print(f"Validation Domain Centroid (c_val): {c_val}")

# 6. Partition Test Set using K-Means (K=2)
print("\nClustering Test Set (K-Means, K=2)...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
test_labels = kmeans.fit_predict(pcs_test)

# 7. Compute test sub-cluster centroids and distances
centroids = kmeans.cluster_centers_
distances = [np.linalg.norm(c_val - c) for c in centroids]

for k in range(2):
    count = np.sum(test_labels == k)
    print(
        f"Cluster {k}: {count} videos | Centroid: {centroids[k]} | Distance to c_val: {distances[k]:.4f}"
    )

# 8. Identify the target anomaly cluster (maximizing distance to c_val)
anomaly_cluster_idx = np.argmax(distances)
normal_cluster_idx = 1 - anomaly_cluster_idx
print(
    f"\nTarget Anomaly Cluster: Cluster {anomaly_cluster_idx} (Distance = {distances[anomaly_cluster_idx]:.4f})"
)

# 9. Extract all videos in the target anomaly cluster
anomalous_videos = []
for i, label in enumerate(test_labels):
    if label == anomaly_cluster_idx:
        video_path = test_meta_v[i]["video"]
        if not os.path.isabs(video_path):
            video_path = os.path.abspath(
                os.path.join("/tf/data/test-dataset", video_path)
            )
        anomalous_videos.append(
            {
                "video_path": video_path,
                "ground_truth": test_meta_v[i]["ground_truth"],
                "PC1": pcs_test[i, 0],
                "PC2": pcs_test[i, 1],
            }
        )

df_anomaly = pd.DataFrame(anomalous_videos)
output_csv = "/tf/data/vjepa2/anomalous_test_videos.csv"
df_anomaly.to_csv(output_csv, index=False)
print(f"Saved {len(df_anomaly)} anomalous videos to {output_csv}")

output_txt = "/tf/data/vjepa2/anomalous_video_paths.txt"
with open(output_txt, "w") as f:
    for video in anomalous_videos:
        f.write(f"{video['video_path']}\n")
print(f"Saved {len(anomalous_videos)} anomalous video paths to {output_txt}")

# 10. Sample 20 videos randomly from the anomaly cluster for audit
print("\n=== Random Sample of 20 Videos for Qualitative Audit ===")
sample_df = df_anomaly.sample(n=min(20, len(df_anomaly)), random_state=42)
for idx, row in sample_df.iterrows():
    print(
        f"- {row['video_path']} (GT Label: {row['ground_truth']}, PC1: {row['PC1']:.2f}, PC2: {row['PC2']:.2f})"
    )
