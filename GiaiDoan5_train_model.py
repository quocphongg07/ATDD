import os
os.environ["DGL_DISABLE_GRAPHBOLT"] = "1"
import pickle
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dgl.nn.pytorch import SAGEConv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

# --- CẤU HÌNH ---
INPUT_DIR = "final_subgraphs"
FEATURE_DIM = 20
HIDDEN_DIM = 128
K_VALUE = 40
NUM_EPOCHS = 150
BATCH_SIZE = 32


# ======================================================================
# PHẦN 1: BỘ NẠP DỮ LIỆU DGL TÙY CHỈNH
# ======================================================================

class CustomDGLDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        self.graphs = []
        self.labels = []
        print(f"Bắt đầu nạp {len(self.pkl_files)} tệp đồ thị con...")
        self._preload_data()
        print("Nạp dữ liệu hoàn tất.")

    def _preload_data(self):
        for file_name in self.pkl_files:
            file_path = os.path.join(self.data_dir, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            self.labels.append(data['label'])
            graph_data = data['graph']
            features_map = data['features']
            nodes = graph_data['nodes']
            node_to_id = {name: i for i, name in enumerate(nodes)}

            src_ids = []
            dst_ids = []
            for src_name, dst_name in graph_data['edges']:
                if src_name in node_to_id and dst_name in node_to_id:
                    src_ids.append(node_to_id[src_name])
                    dst_ids.append(node_to_id[dst_name])

            num_nodes = len(nodes)
            g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes)
            g = dgl.add_self_loop(g)

            feature_tensor = torch.zeros((num_nodes, FEATURE_DIM), dtype=torch.float32)
            for i, node_name in enumerate(nodes):
                feature_vector = features_map.get(node_name)
                if feature_vector is not None:
                    feature_tensor[i] = torch.tensor(feature_vector, dtype=torch.float32)

            g.ndata['feat'] = feature_tensor
            self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def collate_batch(samples):
    graphs, labels = zip(*samples)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, batched_labels


# ======================================================================
# PHẦN 2: KIẾN TRÚC MÔ HÌNH (MỤC 3.3)
# ======================================================================

class GCN_CNN_Model(nn.Module):
    def __init__(self, in_features, hidden_dim, k_value, num_classes=1):
        super(GCN_CNN_Model, self).__init__()
        self.k = k_value

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(SAGEConv(in_features, hidden_dim, 'mean'))
        self.gcn_layers.append(SAGEConv(hidden_dim, hidden_dim, 'mean'))
        self.gcn_layers.append(SAGEConv(hidden_dim, hidden_dim, 'mean'))

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4)

        self.fc1 = nn.Linear(hidden_dim * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g):
        h = g.ndata['feat']

        for layer in self.gcn_layers:
            h = F.relu(layer(g, h))
        g.ndata['h'] = h

        graphs = dgl.unbatch(g)
        h_list = []
        for gi in graphs:
            hi = gi.ndata["h"]
            score = hi.sum(dim=1)
            k = min(self.k, hi.shape[0])
            topk_idx = torch.topk(score, k=k).indices
            topk_feat = hi[topk_idx]

            if topk_feat.shape[0] < self.k:
                pad = torch.zeros(self.k - topk_feat.shape[0],
                                  topk_feat.shape[1],
                                  device=topk_feat.device,
                                  dtype=topk_feat.dtype)
                topk_feat = torch.cat([topk_feat, pad], dim=0)
            h_list.append(topk_feat)

        h_pooled = torch.stack(h_list)
        h_cnn = h_pooled.permute(0, 2, 1)

        h1 = F.relu(self.conv1(F.pad(h_cnn, (1, 0))))
        h2 = F.relu(self.conv2(F.pad(h_cnn, (1, 1))))
        h3 = F.relu(self.conv3(F.pad(h_cnn, (2, 1))))

        h1 = F.adaptive_max_pool1d(h1, 1).squeeze(-1)
        h2 = F.adaptive_max_pool1d(h2, 1).squeeze(-1)
        h3 = F.adaptive_max_pool1d(h3, 1).squeeze(-1)

        h_merged = torch.cat([h1, h2, h3], dim=1)
        h_fc = F.relu(self.fc1(h_merged))
        h_fc = self.dropout(h_fc)
        output = self.fc2(h_fc)
        return output.squeeze(1)


# ======================================================================
# PHẦN 3: HUẤN LUYỆN
# ======================================================================

def main():
    print("--- BẮT ĐẦU GIAI ĐOẠN 5: HUẤN LUYỆN ---")

    dataset = CustomDGLDataset(INPUT_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )

    model = GCN_CNN_Model(
        in_features=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        k_value=K_VALUE
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Bắt đầu huấn luyện {NUM_EPOCHS} epochs trên {len(dataset)} mẫu...")

    all_epoch_metrics = []
    summary_metrics = []

    # In tiêu đề bảng
    print(
        f"\n{'Epoch':<6} | {'Loss':<7} | {'Accuracy':<10} | {'Precision':<9} | {'F1-Score':<9} | {'AUC':<7} | {'TPR':<7} | {'FPR':<7}")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        model.train()

        epoch_labels = []
        epoch_preds = []
        epoch_logits = []
        total_loss = 0

        for batched_graph, labels in dataloader:
            logits = model(batched_graph)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).int()

            epoch_labels.extend(labels.cpu().numpy())
            epoch_preds.extend(preds.cpu().detach().numpy())
            epoch_logits.extend(logits.cpu().detach().numpy())
            total_loss += loss.item()

        y_true = np.array(epoch_labels)
        y_pred = np.array(epoch_preds)
        y_scores = np.array(epoch_logits)

        avg_loss = total_loss / len(dataloader)
        acc = (y_true == y_pred).mean()
        precision = precision_score(y_true, y_pred, zero_division=0)
        tpr_recall = recall_score(y_true, y_pred, zero_division=0)  # TPR
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        except Exception:
            fpr = 0.0

        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.5

        epoch_results = {
            'Loss': avg_loss,
            'Accuracy': acc,
            'Precision': precision,
            'F1-Score': f1,
            'AUC': auc,
            'TPR': tpr_recall,
            'FPR': fpr
        }
        all_epoch_metrics.append(epoch_results)

        if (epoch + 1) % 10 == 0:
            print(
                f"{(epoch + 1):<6} | {avg_loss:<7.4f} | {acc * 100:<10.2f}% | {precision:<9.4f} | {f1:<9.4f} | {auc:<7.4f} | {tpr_recall:<7.4f} | {fpr:<7.4f}")
            summary_metrics.append(epoch_results)

    print("--- HOÀN THÀNH HUẤN LUYỆN ---")

    print("\n--- BẢNG TỔNG HỢP (CÁC EPOCH ĐÃ GHI LẠI) ---")

    df_summary = pd.DataFrame(summary_metrics)

    df_summary.index = range(10, NUM_EPOCHS + 1, 10)
    df_summary.index.name = "Epoch"

    df_summary['Accuracy'] = df_summary['Accuracy'].map(lambda x: f"{x * 100:.2f}%")
    pd.set_option('display.precision', 4)
    print(df_summary)

    print("\n--- BẢNG TÍNH TRUNG BÌNH CÁC LẦN (TRÊN TẤT CẢ 150 EPOCH) ---")

    df_full = pd.DataFrame(all_epoch_metrics)
    print(df_full.mean())


if __name__ == "__main__":
    main()