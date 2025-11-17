import os
import pickle
import json
import networkx as nx
from joblib import Parallel, delayed
import logging

# Tắt log (nếu có)
logging.getLogger("androguard").setLevel(logging.CRITICAL) 
logging.getLogger("joblib").setLevel(logging.WARNING)

# --- CẤU HÌNH ---
INPUT_DIR = "processed_features_graphs"
API_RANK_FILE = "api_rank.json"
OUTPUT_DIR = "final_subgraphs"

# Tham số từ bài báo
TOP_K_APIS = 10     
HOP_DISTANCE = 2  
# --------------------------

def build_nx_graph(graph_data):
    """ Xây dựng lại đồ thị NetworkX từ dữ liệu đã lưu. """
    G = nx.DiGraph() # Đồ thị có hướng (FCG)
    G.add_nodes_from(graph_data['nodes'])
    G.add_edges_from(graph_data['edges'])
    return G

def extract_subgraph_for_apk(pkl_path, output_dir, ranked_api_map):
    """
    Triển khai Giai đoạn 4b: Trích xuất S-FCSG cho một APK.
    """
    try:
        # print(f"Bắt đầu trích xuất S-FCSG cho: {pkl_path}")

        # 1. Tải dữ liệu FCG đầy đủ (từ Giai đoạn 3)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        graph_data = data['graph']
        features_map = data['features']

        # 2. Xây dựng lại đồ thị NetworkX
        fcg = build_nx_graph(graph_data)

        # 3. Tìm Top K API nhạy cảm có trong đồ thị này
        apis_in_this_graph = []
        for node_name in graph_data['nodes']:
            if node_name in ranked_api_map:
                apis_in_this_graph.append((node_name, ranked_api_map[node_name]))

        apis_in_this_graph.sort(key=lambda x: x[1], reverse=True)
        top_k_nodes_in_graph = [api_name for api_name, score in apis_in_this_graph[:TOP_K_APIS]]

        if not top_k_nodes_in_graph:
            # print(f"CẢNH BÁO: Không tìm thấy API nhạy cảm nào trong {pkl_path}. Bỏ qua.")
            return False

        # 4. Triển khai Thuật toán 1: Lân cận 2-hop
        sensitive_node_set = set(top_k_nodes_in_graph)

        # Chuyển sang đồ thị vô hướng để tìm kiếm lân cận
        undirected_fcg = fcg.to_undirected()

        for start_node in top_k_nodes_in_graph:
            if start_node not in undirected_fcg:
                continue

            # nx.ego_graph tìm tất cả các nút trong bán kính 'HOP_DISTANCE' (2)
            two_hop_neighbors = nx.ego_graph(undirected_fcg, start_node, radius=HOP_DISTANCE).nodes()
            sensitive_node_set.update(two_hop_neighbors)

        # 5. Tạo S-FCSG (Đồ thị con)
        s_fcsg = fcg.subgraph(sensitive_node_set)

        # 6. Lọc các đặc trưng
        final_features = {}
        for node_name in s_fcsg.nodes():
            if node_name in features_map:
                final_features[node_name] = features_map[node_name]

        # 7. Lưu dữ liệu S-FCSG cuối cùng
        final_data = {
            "graph": {
                "nodes": list(s_fcsg.nodes()),
                "edges": list(s_fcsg.edges())
            },
            "features": final_features,
            "label": data['label']
        }

        base_name = os.path.splitext(os.path.basename(pkl_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_subgraph.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(final_data, f)

        print(f"Trích xuất S-FCSG thành công: {base_name}.pkl (Tổng số nút: {len(sensitive_node_set)})")
        return True

    except Exception as e:
        print(f"LỖI khi trích xuất S-FCSG cho {pkl_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        with open(API_RANK_FILE, 'r') as f:
            ranked_api_list = json.load(f)
        ranked_api_map = dict(ranked_api_list)
        print(f"Đã tải {len(ranked_api_map)} API từ bảng xếp hạng.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp {API_RANK_FILE}. Hãy chạy GiaiDoan4a trước.")
        return

    all_pkl_files = [
        os.path.join(INPUT_DIR, f) 
        for f in os.listdir(INPUT_DIR) 
        if f.endswith('.pkl')
    ]

    print(f"Tìm thấy {len(all_pkl_files)} tệp FCG đầy đủ để xử lý.")

    print("Bắt đầu trích xuất S-FCSG (Giai đoạn 4b - ĐÃ SỬA LỖI)...")
    results = Parallel(n_jobs=1)(
        delayed(extract_subgraph_for_apk)(pkl_path, OUTPUT_DIR, ranked_api_map) 
        for pkl_path in all_pkl_files
    )

    print("--- HOÀN THÀNH GIAI ĐOẠN 4b (ĐÃ SỬA LỖI) ---")
    print(f"Tổng số S-FCSG trích xuất thành công: {sum(results)}")
    print(f"Đã lưu các tệp đồ thị con cuối cùng vào: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
