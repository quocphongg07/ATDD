import os
import pickle
import json
import math
from collections import defaultdict

# --- CẤU HÌNH ---
INPUT_DIR = "processed_features_graphs"
OUTPUT_RANK_FILE = "api_rank.json"

def load_all_data_and_build_api_set(input_dir):
    """
    Tải tất cả các tệp .pkl, đồng thời thu thập 3 thứ:
    1. Dữ liệu của từng tệp (graph, features, label)
    2. Toàn bộ các API bên ngoài (để làm SENSITIVE_API_SET)
    3. Đếm tổng số API được gọi trong từng tệp (cho TF-IDF)
    """
    all_app_data = []
    global_api_set = set()

    print(f"Bắt đầu tải dữ liệu từ: {INPUT_DIR}")

    for pkl_file in os.listdir(input_dir):
        if not pkl_file.endswith('.pkl'):
            continue

        file_path = os.path.join(input_dir, pkl_file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

            app_api_calls = defaultdict(int) # {api_name: count}

            # Lặp qua các nút để tìm API bên ngoài
            for node_name, features in data['features'].items():
                # Đặc trưng Dalvik[0] == 1 nghĩa là nút bên ngoài (external)
                if features[0] == 1:
                    global_api_set.add(node_name)
                    app_api_calls[node_name] += 1

            data['api_counts'] = app_api_calls
            data['total_api_calls'] = sum(app_api_calls.values())
            all_app_data.append(data)

    print(f"Đã tải {len(all_app_data)} tệp.")
    print(f"Tìm thấy tổng cộng {len(global_api_set)} API bên ngoài (SENSITIVE_API_SET tạm thời).")
    return all_app_data, list(global_api_set)

def calculate_api_coefficients(all_app_data, sensitive_api_set):
    """
    Triển khai Mục 3.2.2 - Tính Hệ số API (TF-IDF)
    """
    api_coefficients = {}

    # Tách dữ liệu thành 2 nhóm
    malware_apps = [d for d in all_app_data if d['label'] == 1]
    benign_apps = [d for d in all_app_data if d['label'] == 0]

    num_malware = len(malware_apps)
    num_benign = len(benign_apps)

    if num_malware == 0 or num_benign == 0:
        print("LỖI: Cần ít nhất 1 mẫu malware và 1 mẫu benign để tính TF-IDF.")
        return None

    print(f"Tính toán TF-IDF cho {len(sensitive_api_set)} API...")

    for api in sensitive_api_set:
        # --- Tính cho MALWARE ---
        # TF (Công thức 5) [cite: 444]
        tf_malware_sum = 0
        # IDF (Công thức 6) [cite: 445]
        num_malware_containing_api = 0

        for app in malware_apps:
            if api in app['api_counts']:
                num_malware_containing_api += 1
                if app['total_api_calls'] > 0:
                    tf_malware_sum += (app['api_counts'][api] / app['total_api_calls'])

        # (Thêm 1 vào mẫu số để tránh chia cho 0) [cite: 451]
        idf_malware = math.log(num_malware / (num_malware_containing_api + 1))
        tf_idf_malware = tf_malware_sum * idf_malware # (Công thức 7) [cite: 446]

        # --- Tính cho BENIGN ---
        tf_benign_sum = 0
        num_benign_containing_api = 0

        for app in benign_apps:
            if api in app['api_counts']:
                num_benign_containing_api += 1
                if app['total_api_calls'] > 0:
                    tf_benign_sum += (app['api_counts'][api] / app['total_api_calls'])

        idf_benign = math.log(num_benign / (num_benign_containing_api + 1))
        tf_idf_benign = tf_benign_sum * idf_benign

        # --- Tính Hệ số API cuối cùng (Công thức 8) [cite: 454] ---
        coefficient = (1 + tf_idf_malware) * (1 + tf_idf_benign)
        api_coefficients[api] = coefficient

    # Sắp xếp danh sách API theo hệ số giảm dần
    sorted_apis = sorted(api_coefficients.items(), key=lambda item: item[1], reverse=True)

    return sorted_apis

def main():
    all_data, api_set = load_all_data_and_build_api_set(INPUT_DIR)

    if not all_data:
        print(f"Không tìm thấy tệp .pkl nào trong {INPUT_DIR}")
        return

    ranked_apis = calculate_api_coefficients(all_data, api_set)

    if ranked_apis:
        print(f"--- API NHẠY CẢM NHẤT (Top 5) ---")
        for i, (api, score) in enumerate(ranked_apis[:5]):
            print(f"#{i+1}: {api} (Score: {score:.4f})")

        # Lưu toàn bộ danh sách xếp hạng vào tệp JSON
        with open(OUTPUT_RANK_FILE, 'w') as f:
            json.dump(ranked_apis, f, indent=4)

        print(f"\n--- HOÀN THÀNH GIAI ĐOẠN 4a ---")
        print(f"Đã lưu danh sách xếp hạng API vào: {OUTPUT_RANK_FILE}")

if __name__ == "__main__":
    main()
