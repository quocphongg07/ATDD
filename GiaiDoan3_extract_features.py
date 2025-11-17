import os
import pickle
import sys
import logging
import networkx as nx
import numpy as np
import json
from androguard.misc import AnalyzeAPK
import androguard.core.analysis.analysis 
from joblib import Parallel, delayed

logging.getLogger("androguard").setLevel(logging.CRITICAL) 

# --- CẤU HÌNH ---
INPUT_BENIGN_DIR = "data/benign_apks" 
INPUT_MALWARE_DIR = "data/malware_apks"   
OUTPUT_DIR = "processed_features_graphs_FINAL"
AOSP_MAP_FILE = "aosp_map.json"
# --------------------------

# ======================================================================
# PHẦN 1: CÁC HÀM ĐẶC TRƯNG
# ======================================================================

OPCODE_CATEGORIES = {
    'nop': 1, 'move': 2, 'return': 3, 'const': 4, 'monitor': 5, 'check': 6,
    'array': 7, 'throw': 8, 'goto': 9, 'switch': 9, 'if': 9,
    'cmpl': 10, 'cmpg': 10, 'cmp': 10,
    'iget': 11, 'iput': 11, 'sget': 11, 'sput': 11,
    'invoke': 12,
    'neg': 13, 'not': 13, 'int-to': 13, 'long-to': 13, 'float-to': 13, 
    'double-to': 13, 'int': 13, 'long': 13, 'float': 13, 'double': 13
}

def calculate_dalvik_features(method_analysis):
    """ Hàm 1: Tính đặc trưng Dalvik Opcodes (Mục 3.1.1) """
    vector = np.zeros(14, dtype=np.uint8)
    if method_analysis is None or (hasattr(method_analysis, 'is_external') and method_analysis.is_external()):
        vector[0] = 1 
        return vector

    try:
        for instruction in method_analysis.get_method().get_instructions():
            op_name = instruction.get_name()
            for category_prefix, index in OPCODE_CATEGORIES.items():
                if op_name.startswith(category_prefix):
                    vector[index] = 1
                    break 
    except Exception as e:
        pass 
    return vector

def calculate_importance_features(fcg, node):
    """ Hàm 2: Tính đặc trưng Tầm quan trọng Nút (Mục 3.1.2) """
    try:
        in_degree = fcg.in_degree(node)
        out_degree = fcg.out_degree(node)
        if not fcg.nodes():
             return np.zeros(3)
        centrality = nx.degree_centrality(fcg).get(node, 0.0)
        return np.array([in_degree, out_degree, centrality])
    except Exception as e:
        return np.zeros(3)

def calculate_function_weight_features(method_analysis, aosp_api_map):
    """
    Hàm 3: Tính đặc trưng Trọng số Hàm (Mục 3.1.3)
    *** PHIÊN BẢN HOÀN THIỆN ***
    Triển khai logic của Eq. 2, 3, 4
    """

    # Trường hợp 1: Nút là hàm BÊN NGOÀI (External API) [cite: 316, 325]
    if method_analysis is None or (hasattr(method_analysis, 'is_external') and method_analysis.is_external()):
        api_name = get_node_name(method_analysis) if method_analysis else "UNKNOWN"

        # Lấy trọng số (0 hoặc 1) từ map
        weight = aosp_api_map.get(api_name, 0)

        # Nếu trọng số > 0, tỷ lệ là 100%
        ratio = 1.0 if weight > 0 else 0.0
        product = weight * ratio

        return np.array([weight, ratio, product])

    # Trường hợp 2: Nút là hàm BÊN TRONG (Internal) [cite: 315, 324]
    total_external_calls = 0  # (denominator cho Eq. 4)
    dangerous_calls = 0       # (numerator cho Eq. 4)
    api_call_weight = 0       # (Eq. 2)

    try:
        for _call in method_analysis.get_method().get_xref_to():
            called_method_obj = _call[0] 

            if isinstance(called_method_obj, androguard.core.analysis.analysis.ExternalMethod):
                total_external_calls += 1
                api_name = called_method_obj.full_name 

                # Tra cứu trọng số (0 hoặc 1)
                weight = aosp_api_map.get(api_name, 0)

                if weight > 0:
                    dangerous_calls += 1
                    api_call_weight += weight # Eq. 2: cộng dồn trọng số
    except Exception:
        pass 

    # Tính toán Eq. 4 [cite: 332-334]
    ratio = (dangerous_calls / total_external_calls) if total_external_calls > 0 else 0.0

    # Tính toán tích
    product = api_call_weight * ratio

    # Trả về vector 3 chiều [cite: 340]
    return np.array([api_call_weight, ratio, product])

# ======================================================================
# PHẦN 2: KỊCH BẢN XỬ LÝ CHÍNH
# ======================================================================

def get_node_name(node):
    try:
        if hasattr(node, 'get_method'): 
            return node.get_method().full_name 
        else: 
            return node.full_name
    except Exception:
        return str(node) 

def process_single_apk(apk_path, output_dir, label, aosp_api_map):
    try:
        print(f"Bắt đầu xử lý (HOÀN THIỆN): {apk_path}")

        a, d, dx = AnalyzeAPK(apk_path)
        fcg = dx.get_call_graph()

        method_map = {m.get_method().full_name: m for m in dx.get_methods()}

        node_features_map = {} 
        all_nodes_map = {get_node_name(n): n for n in fcg.nodes()}

        for node_name, node_obj in all_nodes_map.items():

            default_obj = node_obj if (hasattr(node_obj, 'is_external') and node_obj.is_external()) else None
            method_obj = method_map.get(node_name, default_obj)

            vec_dalvik = calculate_dalvik_features(method_obj)
            vec_import = calculate_importance_features(fcg, node_obj)
            vec_weight = calculate_function_weight_features(method_obj, aosp_api_map)

            combined_vector = np.concatenate([vec_dalvik, vec_import, vec_weight])
            node_features_map[node_name] = combined_vector

        simple_fcg = {
            "nodes": list(all_nodes_map.keys()),
            "edges": [(get_node_name(u), get_node_name(v)) for u, v in fcg.edges()]
        }

        base_name = os.path.splitext(os.path.basename(apk_path))[0]
        data_to_save = {
            "graph": simple_fcg,
            "features": node_features_map,
            "label": label 
        }

        output_path = os.path.join(output_dir, f"{base_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"Xử lý thành công (features+graph HOÀN THIỆN): {apk_path} -> {output_path}")
        return True

    except Exception as e:
        print(f"LỖI khi xử lý {apk_path}: {type(e).__name__} - {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() 
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        with open(AOSP_MAP_FILE, 'r') as f:
            AOSP_API_MAP = json.load(f)
        print(f"Đã tải thành công {len(AOSP_API_MAP)} API từ {AOSP_MAP_FILE}")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp {AOSP_MAP_FILE}.")
        print("Vui lòng tạo tệp này (theo Bước 1) trước khi chạy.")
        return

    benign_files = [os.path.join(INPUT_BENIGN_DIR, f) for f in os.listdir(INPUT_BENIGN_DIR) if f.endswith('.apk')]
    malware_files = [os.path.join(INPUT_MALWARE_DIR, f) for f in os.listdir(INPUT_MALWARE_DIR) if f.endswith('.apk')]

    all_tasks = [(f, OUTPUT_DIR, 0, AOSP_API_MAP) for f in benign_files] + \
                [(f, OUTPUT_DIR, 1, AOSP_API_MAP) for f in malware_files]

    print(f"Tìm thấy tổng cộng {len(all_tasks)} tệp APK để xử lý.")

    print("Bắt đầu xử lý song song (Giai đoạn 3 HOÀN THIỆN)...")
    results = Parallel(n_jobs=-1)(
        delayed(process_single_apk)(task[0], task[1], task[2], task[3]) for task in all_tasks
    )

    print("--- HOÀN THÀNH GIAI ĐOẠN 3 (HOÀN THIỆN) ---")
    print(f"Tổng số tệp xử lý thành công: {sum(results)}")
    print(f"Đã lưu các tệp (graph + features) vào thư mục: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()