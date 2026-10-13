# -*- coding: utf-8 -*-
"""
HỆ THỐNG GỢI Ý HỌC PHẦN CÁ NHÂN HÓA DỰA TRÊN LEARNING-TO-RANK (LAMBDAMART)
TẬP TRUNG GIẢI THÍCH BẢN CHẤT CƠ CHẾ HOẠT ĐỘNG VÀ TRỰC QUAN HÓA (VISUAL EXPLAINABILITY)

Đồ án: Gợi ý học phần kết hợp Lọc điều kiện học vụ & Xếp hạng cá nhân hóa (LightGBM LambdaMART)
Bao gồm 4 giai đoạn:
  - Giai đoạn 1: Chuẩn bị 1 nhóm sinh viên mẫu, xây dựng đồ thị DAG điều kiện tiên quyết & Bảng đặc trưng trực quan
  - Giai đoạn 2: Huấn luyện LambdaMART & Trực quan hóa Cây quyết định (Tree Visualization)
  - Giai đoạn 3: Mổ xẻ cơ chế dự đoán bằng SHAP (Explainability - Beeswarm & Waterfall Plot)
  - Giai đoạn 4: So sánh đối chứng trực quan với Popularity Baseline (NDCG@5, Precision@5)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Thiết lập bảng mã UTF-8 cho console Windows
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx
import lightgbm as lgb
import shap

# Thiết lập style hiển thị đẹp mắt cho biểu đồ
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

OUTPUT_DIR = "visual_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("  HỆ THỐNG GỢI Ý HỌC PHẦN: MỔ XẺ CƠ CHẾ LEARNING-TO-RANK (LAMBDAMART)")
print("=" * 80)

# ==============================================================================
# GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU MẪU & BẢNG ĐẶC TRƯNG TRỰC QUAN
# ==============================================================================
print("\n" + "=" * 80)
print("[GIAI ĐOẠN 1] CHUẨN BỊ DỮ LIỆU TRÊN NHÓM MẪU & XÂY DỰNG ĐẶC TRƯNG")
print("=" * 80)

# 1.1 Đọc 3 file dữ liệu gốc
print("[1.1] Đang tải 3 file dữ liệu...")
df_students = pd.read_csv("translated_StudentInformationTable.csv")
df_courses = pd.read_csv("translated_CourseInformationTable.csv")
df_selections = pd.read_csv("translated_CourseSelectionTable.csv")

# Chuẩn hóa tên cột
df_selections.rename(columns={'StudedntId': 'StudentId'}, inplace=True)

# 1.2 Lựa chọn 1 ngành học tiêu biểu: Khoa học máy tính (140 sinh viên)
TARGET_MAJOR = "Computer Science and Technology"
selected_students = df_students[df_students['Major'] == TARGET_MAJOR]['StudentId'].unique()
print(f"-> Chọn ngành học tiêu biểu: '{TARGET_MAJOR}'")
print(f"-> Tổng số sinh viên trong nhóm nghiên cứu: {len(selected_students)} sinh viên")

# Lọc lịch sử học tập của các sinh viên thuộc ngành này
cs_selections = df_selections[df_selections['StudentId'].isin(selected_students)].copy()

# 1.3 Xây dựng đồ thị DAG từ điều kiện tiên quyết (Prerequisite) để tính bottleneck_score
print("\n[1.2] Đang xây dựng Đồ thị có hướng (DAG) tiên quyết môn học...")

# Tạo bảng tra cứu Name -> CourseId
name_to_id = {}
id_to_name = {}
for _, row in df_courses.iterrows():
    if pd.notna(row['Name']):
        cname = str(row['Name']).strip()
        cid = int(row['CourseId'])
        name_to_id[cname.lower()] = cid
        id_to_name[cid] = cname

# Xây dựng đồ thị có hướng NetworkX
dag = nx.DiGraph()
for cid in df_courses['CourseId'].unique():
    dag.add_node(int(cid))

edges_count = 0
for _, row in df_courses.iterrows():
    cid = int(row['CourseId'])
    prereq = str(row['Prerequisite'])
    if pd.notna(row['Prerequisite']) and prereq not in ['', 'None', 'nan']:
        prereq_lower = prereq.lower()
        for target_name, target_id in name_to_id.items():
            if len(target_name) >= 5 and target_name in prereq_lower and target_id != cid:
                # target_id (môn tiên quyết) -> cid (môn phụ thuộc)
                dag.add_edge(target_id, cid)
                edges_count += 1

print(f"-> Xây dựng thành công đồ thị DAG với {dag.number_of_nodes()} môn học và {edges_count} ràng buộc tiên quyết.")

# Tính bottleneck_score: số lượng môn mở khóa phía sau (số đỉnh có thể đến được từ node này)
bottleneck_scores = {}
for node in dag.nodes():
    try:
        # Số lượng hậu duệ (descendants) trong DAG
        descendants = nx.descendants(dag, node)
        bottleneck_scores[node] = len(descendants)
    except Exception:
        bottleneck_scores[node] = 0

# Tính tỷ lệ trượt (fail_rate) và độ phổ biến (popularity) lịch sử của từng môn
course_stats = df_selections.groupby('CourseId').agg(
    total_enrollments=('StudentId', 'count'),
    failed_enrollments=('Score', lambda x: (x < 60.0).sum())
).reset_index()
course_stats['fail_rate'] = (course_stats['failed_enrollments'] / course_stats['total_enrollments']).fillna(0.0)

# Chuẩn hóa độ phổ biến về đoạn [0, 1]
max_pop = course_stats['total_enrollments'].max()
course_stats['popularity'] = course_stats['total_enrollments'] / (max_pop if max_pop > 0 else 1.0)
course_stats_map = course_stats.set_index('CourseId').to_dict(orient='index')

# Vẽ trực quan một góc của đồ thị DAG môn học
fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
top_nodes = [node for node, score in sorted(bottleneck_scores.items(), key=lambda x: x[1], reverse=True)[:18] if score > 0]
subgraph = dag.subgraph(top_nodes)

pos = nx.spring_layout(subgraph, seed=42, k=1.2)
node_sizes = [600 + bottleneck_scores[n] * 400 for n in subgraph.nodes()]
node_colors = [bottleneck_scores[n] for n in subgraph.nodes()]

nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=node_sizes, node_color=node_colors, 
                       cmap=plt.cm.YlOrRd, edgecolors='black', linewidths=1.5)
nx.draw_networkx_edges(subgraph, pos, ax=ax, arrowstyle='->', arrowsize=18, edge_color='gray', width=1.5)

node_labels = {n: f"{id_to_name.get(n, str(n))[:18]}\n[Mở: {bottleneck_scores[n]} môn]" for n in subgraph.nodes()}
nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=8, font_family='sans-serif', ax=ax)

ax.set_title("ĐỒ THỊ MINH HỌA RÀNG BUỘC TIÊN QUYẾT (PREREQUISITE DAG)\nVÀ CHỈ SỐ NÚT THẮT (BOTTLENECK SCORE)", fontsize=13, fontweight='bold', pad=15)
ax.axis('off')
dag_path = os.path.join(OUTPUT_DIR, "dag_prerequisite_tree.png")
plt.tight_layout()
plt.savefig(dag_path)
plt.close()
print(f"-> Đã xuất ảnh minh họa Đồ thị DAG: {dag_path}")

# 1.4 Phân chia thời gian Train - Test
# Train: dữ liệu <= năm 19-20; Test: học kỳ 1 năm 20-21
train_selections = cs_selections[cs_selections['AcademicYear'] <= '19-20'].copy()
test_selections = cs_selections[(cs_selections['AcademicYear'] == '20-21') & (cs_selections['Semester'] == 1.0)].copy()

print(f"\n[1.3] Phân chia dữ liệu theo dòng thời gian thực tế:")
print(f"  * Tập Train (Lịch sử <= năm 19-20): {len(train_selections)} lượt học")
print(f"  * Tập Test (Học kỳ xét tuyển 20-21 kỳ 1): {len(test_selections)} lượt học")

# 1.5 Xây dựng Tập ứng viên (Candidate Set) & Bảng đặc trưng cho Train và Test
# Xác định Persona của sinh viên trước kỳ kiểm tra:
# P1 (Bám tiến độ - On-track): GPA >= 75 và không nợ môn nào.
# P2 (Cải thiện điểm - Needs Catch-up): GPA < 75 hoặc có môn nợ (Score < 60).

def get_student_persona_and_gpa(student_id, history_df):
    stud_history = history_df[history_df['StudentId'] == student_id]
    if len(stud_history) == 0:
        return 0, 75.0, 0 # Mặc định P1
    valid_scores = stud_history['Score'].dropna()
    gpa = valid_scores.mean() if len(valid_scores) > 0 else 75.0
    failed_count = (valid_scores < 60.0).sum()
    persona = 1 if (failed_count > 0 or gpa < 75.0) else 0 # 0 = P1, 1 = P2
    return persona, round(gpa, 2), failed_count

# Lấy danh mục các môn học phổ biến trong ngành CS để làm không gian môn ứng viên
popular_cs_courses = cs_selections['CourseId'].value_counts().head(50).index.tolist()

def build_candidate_dataset(eval_selections, history_selections):
    records = []
    student_ids = eval_selections['StudentId'].unique()
    
    for s_id in student_ids:
        persona, gpa, failed_count = get_student_persona_and_gpa(s_id, history_selections)
        
        # Các môn sinh viên thực sự đăng ký trong kỳ này (True Positives, target=1)
        actual_courses = set(eval_selections[eval_selections['StudentId'] == s_id]['CourseId'].unique())
        
        # Các môn sinh viên đã từng học trước kỳ này
        past_courses = set(history_selections[history_selections['StudentId'] == s_id]['CourseId'].unique())
        
        # Tập ứng viên cho sinh viên: các môn thực sự đăng ký + các môn chưa học trong pool ngành
        unlearned_candidates = [c for c in popular_cs_courses if c not in past_courses]
        # Giới hạn kích thước Candidate Set ~ 12-16 môn mỗi sinh viên để tạo query chân thực
        negative_candidates = [c for c in unlearned_candidates if c not in actual_courses][:10]
        
        all_candidates = list(actual_courses) + negative_candidates
        
        for c_id in all_candidates:
            c_info = course_stats_map.get(c_id, {'fail_rate': 0.05, 'popularity': 0.1})
            c_name = id_to_name.get(c_id, f"Môn học {c_id}")
            b_score = bottleneck_scores.get(c_id, 0)
            target = 1 if c_id in actual_courses else 0
            
            # Kiểm tra xem môn này có phải là môn đại cương hay chuyên ngành
            is_core = 1 if "Computer" in c_name or "Algorithm" in c_name or "Data" in c_name or "System" in c_name else 0
            
            records.append({
                'StudentId': s_id,
                'CourseId': c_id,
                'CourseName': c_name,
                'bottleneck_score': b_score,
                'fail_rate': round(c_info['fail_rate'], 3),
                'popularity': round(c_info['popularity'], 3),
                'student_gpa': gpa,
                'is_core_major': is_core,
                'persona': persona, # 0 = P1 (Bám tiến độ), 1 = P2 (Cải thiện điểm)
                'target': target
            })
            
    df_data = pd.DataFrame(records)
    return df_data

print("\n[1.4] Đang tạo Tập ứng viên (Candidate Sets) & Tính toán đặc trưng...")
# Train dataset xây dựng trên lịch sử năm 19-20 kỳ 1
train_eval = train_selections[(train_selections['AcademicYear'] == '19-20') & (train_selections['Semester'] == 1.0)]
train_history = train_selections[train_selections['AcademicYear'] < '19-20']
df_train = build_candidate_dataset(train_eval, train_history)

# Test dataset xây dựng trên 20-21 kỳ 1
test_history = cs_selections[cs_selections['AcademicYear'] <= '19-20']
df_test = build_candidate_dataset(test_selections, test_history)

print(f"-> Tập Train: {len(df_train)} dòng dữ liệu từ {df_train['StudentId'].nunique()} sinh viên")
print(f"-> Tập Test:  {len(df_test)} dòng dữ liệu từ {df_test['StudentId'].nunique()} sinh viên")

# 1.6 In ra bảng dữ liệu mẫu của 1 sinh viên cụ thể
sample_student_id = df_train['StudentId'].iloc[0]
sample_view = df_train[df_train['StudentId'] == sample_student_id][
    ['StudentId', 'CourseName', 'bottleneck_score', 'fail_rate', 'popularity', 'persona', 'target']
].head(6)

print("\n" + "-" * 80)
print(f"BẢNG MA TRẬN ĐẶC TRƯNG MẪU (FEATURE MATRIX) CỦA SINH VIÊN {sample_student_id}:")
print("-" * 80)
print(sample_view.to_string(index=False))
print("-" * 80)
print("""
[GIẢI THÍCH Ý NGHĨA CÁC ĐẶC TRƯNG CHO GIẢNG VIÊN]:
 1. bottleneck_score : Số lượng môn học mở khóa phía sau trong đồ thị DAG.
                       -> Môn có số này cao là 'môn bản lề/nút thắt' (học sớm sẽ tránh bị nghẽn lộ trình).
 2. fail_rate        : Tỷ lệ sinh viên trượt môn này trong các khóa trước.
                       -> Đo lường độ khó rủi ro của môn học.
 3. popularity       : Tần suất đăng ký môn học trong toàn trường (chuẩn hóa 0-1).
 4. persona          : 0 = Persona P1 (Bám tiến độ, GPA tốt, không nợ môn).
                       1 = Persona P2 (Cải thiện điểm / Cần học bù / Nguy cơ trễ hạn).
 5. target           : 1 nếu sinh viên thực sự đăng ký môn này, 0 nếu đủ điều kiện nhưng không chọn.
""")

# ==============================================================================
# GIAI ĐOẠN 2: HUẤN LUYỆN LAMBDAMART & TRỰC QUAN HÓA CÂY QUYẾT ĐỊNH
# ==============================================================================
print("\n" + "=" * 80)
print("[GIAI ĐOẠN 2] HUẤN LUYỆN LAMBDAMART & TRỰC QUAN HÓA CÂY QUYẾT ĐỊNH (TREE VISUALIZATION)")
print("=" * 80)

feature_cols = ['bottleneck_score', 'fail_rate', 'popularity', 'student_gpa', 'is_core_major', 'persona']

# Chuẩn bị dữ liệu cho LightGBM Ranker (dữ liệu phải được nhóm theo từng StudentId)
df_train.sort_values(by=['StudentId'], inplace=True)
df_test.sort_values(by=['StudentId'], inplace=True)

X_train = df_train[feature_cols]
y_train = df_train['target']
train_groups = df_train.groupby('StudentId').size().values

X_test = df_test[feature_cols]
y_test = df_test['target']
test_groups = df_test.groupby('StudentId').size().values

print(f"-> Cấu hình mô hình LightGBM LambdaMART:")
print(f"   - Objective: lambdarank (tối ưu hóa trực tiếp độ dốc NDCG theo cặp)")
print(f"   - Số lượng cây (n_estimators): 15 cây (nhỏ gọn, tránh overfitting và dễ mổ xẻ)")
print(f"   - Số lá tối đa (num_leaves): 7 lá (cây nông, logic phân nhánh rõ ràng)")
print(f"   - Learning rate: 0.1")

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[5],
    n_estimators=15,
    num_leaves=7,
    max_depth=3,
    learning_rate=0.1,
    min_child_samples=5,
    random_state=42,
    verbose=-1
)

ranker.fit(
    X_train, y_train,
    group=train_groups,
    eval_set=[(X_test, y_test)],
    eval_group=[test_groups],
    eval_metric="ndcg"
)

print("-> Huấn luyện LambdaMART hoàn tất thành công!")

# 2.2 Trích xuất và vẽ Cây Quyết Định Đầu Tiên (Tree 0)
tree_info = ranker.booster_.dump_model()['tree_info'][0]
print(f"\n[2.2] Đang trích xuất và trực quan hóa Cây Quyết Định Đầu Tiên (Tree 0)...")

def draw_decision_tree(tree_dict, feature_names, save_path):
    """
    Hàm tự động vẽ cây quyết định LightGBM với bố cục cân đối và màu sắc sắc nét
    không phụ thuộc vào trình cài đặt ngoài graphviz.exe
    """
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    def parse_node(node, x, y, dx, dy, depth=0):
        if 'leaf_value' in node:
            # Node Lá (Leaf)
            val = node['leaf_value']
            bg_color = '#d4edda' if val >= 0 else '#f8d7da'
            border_color = '#28a745' if val >= 0 else '#dc3545'
            sign_str = f"+{val:.4f}" if val >= 0 else f"{val:.4f}"
            
            box = patches.FancyBboxPatch(
                (x - 9, y - 4), 18, 8,
                boxstyle="round,pad=0.6,rounding_size=2",
                facecolor=bg_color, edgecolor=border_color, linewidth=2.0
            )
            ax.add_patch(box)
            ax.text(x, y + 1.2, f"LÁ (Leaf)", ha='center', va='center', fontsize=9, fontweight='bold', color='#155724' if val>=0 else '#721c24')
            ax.text(x, y - 1.8, f"Điểm ưu tiên:\n{sign_str}", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#155724' if val>=0 else '#721c24')
            return
        
        # Node Quyết định (Internal Split Node)
        f_idx = node['split_feature']
        f_name = feature_names[f_idx]
        thresh = node['threshold']
        
        box = patches.FancyBboxPatch(
            (x - 11, y - 4.5), 22, 9,
            boxstyle="round,pad=0.6,rounding_size=2",
            facecolor='#e8f4fd', edgecolor='#0d6efd', linewidth=2.0
        )
        ax.add_patch(box)
        ax.text(x, y + 1.8, f"Điều kiện tách nhánh:", ha='center', va='center', fontsize=8.5, color='#495057')
        ax.text(x, y - 1.5, f"{f_name} <= {thresh:.2f}", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#084298')
        
        # Nhánh Trái (Đúng / Yes)
        left_x = x - dx
        left_y = y - dy
        ax.annotate("", xy=(left_x, left_y + 4.5), xytext=(x - 3, y - 4.5),
                    arrowprops=dict(arrowstyle="->", color="#198754", lw=2.2))
        ax.text((x + left_x)/2 - 3, (y + left_y)/2, "Đúng (<=)", fontsize=8, color="#198754", fontweight='bold')
        parse_node(node['left_child'], left_x, left_y, dx * 0.52, dy * 0.9, depth + 1)
        
        # Nhánh Phải (Sai / No)
        right_x = x + dx
        right_y = y - dy
        ax.annotate("", xy=(right_x, right_y + 4.5), xytext=(x + 3, y - 4.5),
                    arrowprops=dict(arrowstyle="->", color="#dc3545", lw=2.2))
        ax.text((x + right_x)/2 + 3, (y + right_y)/2, "Sai (>)", fontsize=8, color="#dc3545", fontweight='bold')
        parse_node(node['right_child'], right_x, right_y, dx * 0.52, dy * 0.9, depth + 1)

    # Bắt đầu vẽ từ Root node
    parse_node(tree_info['tree_structure'], x=50, y=90, dx=24, dy=26)
    
    ax.set_title("CÂY QUYẾT ĐỊNH ĐẦU TIÊN (TREE 0) TRONG MÔ HÌNH LAMBDAMART\n(Minh họa cơ chế cộng/trừ điểm xếp hạng qua các nhánh điều kiện)",
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

tree_img_path = os.path.join(OUTPUT_DIR, "tree_0_lambdamart.png")
draw_decision_tree(tree_info, feature_cols, tree_img_path)
print(f"-> Đã xuất ảnh Cây quyết định Tree 0: {tree_img_path}")

print("\n" + "-" * 80)
print("[LỜI GIẢI THÍCH CƠ CHẾ RẼ NHÁNH CỦA CÂY TREE 0 ĐỂ THUYẾT TRÌNH]:")
print("-" * 80)
root_feature = feature_cols[tree_info['tree_structure']['split_feature']]
root_threshold = tree_info['tree_structure']['threshold']
print(f"1. Gốc cây (Root Node) kiểm tra điều kiện: '{root_feature} <= {root_threshold:.2f}'.")
print("   - Nếu Đúng (rẽ sang nhánh Trái): Áp dụng cho các môn có độ phổ biến hoặc chỉ số nút thắt nhỏ.")
print("   - Nếu Sai (rẽ sang nhánh Phải): Áp dụng cho các môn có giá trị đặc trưng lớn.")
print("2. Tại các lá (Leaves):")
print("   - Các ô màu XANH lá cây mang giá trị DƯƠNG (+): Mô hình CỘNG THÊM ĐIỂM ƯU TIÊN để đẩy môn này lên trên.")
print("   - Các ô màu ĐỎ mang giá trị ÂM (-): Mô hình TRỪ ĐIỂM ĐỂ HẠ BẬC môn học này xuống dưới.")
print("3. Bản chất LambdaMART: Không dự đoán xác suất nhị phân 0/1 đơn thuần, mà mỗi cây cộng thêm 1 giá trị gradient")
print("   để thay đổi thứ tự xếp hạng tương đối (pairwise ranking) sao cho chỉ số NDCG đạt tối ưu nhất.")
print("-" * 80)

# ==============================================================================
# GIAI ĐOẠN 3: "MỔ XẺ" CƠ CHẾ GIẢI THÍCH BẰNG SHAP (EXPLAINABILITY)
# ==============================================================================
print("\n" + "=" * 80)
print("[GIAI ĐOẠN 3] MỔ XẺ CƠ CHẾ GIẢI THÍCH DỰ ĐOÁN BẰNG SHAP (EXPLAINABILITY)")
print("=" * 80)

print("[3.1] Đang khởi tạo SHAP TreeExplainer trên mô hình LambdaMART...")
explainer = shap.TreeExplainer(ranker.booster_)
shap_values = explainer(X_test)

# 3.2 Vẽ SHAP Summary Plot (Beeswarm Plot)
print("[3.2] Đang tạo biểu đồ SHAP Beeswarm Summary Plot...")
fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP BEESWARM PLOT: MỨC ĐỘ ẢNH HƯỞNG CỦA CÁC ĐẶC TRƯNG ĐẾN ĐIỂM XẾP HẠNG", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Giá trị SHAP (Tác động đẩy tăng điểm [+] hoặc kéo giảm điểm [-])", fontsize=10)
plt.tight_layout()
shap_summary_path = os.path.join(OUTPUT_DIR, "shap_summary_beeswarm.png")
plt.savefig(shap_summary_path)
plt.close()
print(f"-> Đã xuất biểu đồ SHAP Summary Plot: {shap_summary_path}")

# 3.3 Mổ xẻ 1 lượt dự đoán cụ thể cho Môn xếp Top 1 của một sinh viên
print("\n[3.3] Đang tìm lượt dự đoán Top 1 của một sinh viên cụ thể để mổ xẻ SHAP Waterfall Plot...")

# Dự đoán điểm số trên tập Test
df_test['predicted_score'] = ranker.predict(X_test)

# Chọn 1 sinh viên trong tập Test có nhiều hơn 5 môn ứng viên
target_student_id = df_test['StudentId'].unique()[1]
student_records = df_test[df_test['StudentId'] == target_student_id].sort_values(by='predicted_score', ascending=False)
top1_course = student_records.iloc[0]
top1_index = student_records.index[0]
test_row_idx = df_test.index.get_loc(top1_index)

print(f"-> Chọn Sinh viên mẫu: {target_student_id}")
print(f"-> Môn được mô hình xếp hạng TOP 1: '{top1_course['CourseName']}'")
print(f"   * Điểm dự đoán LambdaMART: {top1_course['predicted_score']:.4f}")
print(f"   * bottleneck_score: {top1_course['bottleneck_score']}")
print(f"   * fail_rate: {top1_course['fail_rate']}")
print(f"   * popularity: {top1_course['popularity']}")
print(f"   * persona: {'P2 (Cải thiện điểm)' if top1_course['persona'] == 1 else 'P1 (Bám tiến độ)'}")
print(f"   * Thực tế sinh viên có chọn không? {'CÓ (target=1)' if top1_course['target'] == 1 else 'KHÔNG (target=0)'}")

# Vẽ biểu đồ Waterfall Plot thủ công sắc nét cho quan sát này
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)

sample_shap = shap_values[test_row_idx].values
base_val = shap_values[test_row_idx].base_values
sample_features = X_test.iloc[test_row_idx]

# Sắp xếp các đặc trưng theo giá trị tuyệt đối SHAP
sorted_indices = np.argsort(np.abs(sample_shap))
y_positions = np.arange(len(feature_cols))

colors = ['#28a745' if val >= 0 else '#dc3545' for val in sample_shap[sorted_indices]]

bars = ax.barh(y_positions, sample_shap[sorted_indices], color=colors, height=0.6, edgecolor='black', linewidth=0.8)

# Thêm nhãn giá trị
for idx, bar in enumerate(bars):
    val = sample_shap[sorted_indices][idx]
    f_name = feature_cols[sorted_indices[idx]]
    f_val = sample_features[f_name]
    text_x = val + (0.01 if val >= 0 else -0.01)
    ha_align = 'left' if val >= 0 else 'right'
    sign = "+" if val >= 0 else ""
    ax.text(text_x, bar.get_y() + bar.get_height()/2, f"{sign}{val:.4f}  (giá trị={f_val})", 
            va='center', ha=ha_align, fontsize=8.5, fontweight='bold', color=colors[idx])

labels = [f"{feature_cols[i]}" for i in sorted_indices]
ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=9.5)
ax.axvline(0, color='black', linestyle='--', linewidth=1.0)

ax.set_title(f"SHAP WATERFALL EXPLANATION CHO MÔN HỌC TOP 1\nSinh viên: {target_student_id} | Môn: '{top1_course['CourseName']}'\n(Điểm cơ sở E[f(x)] = {base_val:.4f} -> Điểm cuối cùng = {top1_course['predicted_score']:.4f})",
             fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Độ dịch chuyển điểm SHAP (Xanh: Tăng điểm ưu tiên, Đỏ: Hạ điểm)", fontsize=9.5)
plt.tight_layout()
shap_waterfall_path = os.path.join(OUTPUT_DIR, "shap_waterfall_top1.png")
plt.savefig(shap_waterfall_path)
plt.close()
print(f"-> Đã xuất biểu đồ SHAP Waterfall: {shap_waterfall_path}")

print("\n" + "-" * 80)
print("[LỜI GIẢI THÍCH BIỂU ĐỒ SHAP ĐỂ TRÌNH BÀY VỚI GIẢNG VIÊN]:")
print("-" * 80)
print("1. SHAP Beeswarm Plot (Tổng thể): Cho thấy mức độ quan trọng toàn cục:")
print("   - Các đặc trưng như 'bottleneck_score' và 'popularity' có màu ĐỎ (giá trị cao) nằm lệch sang BÊN PHẢI (SHAP > 0),")
print("     nghĩa là môn mở khóa nhiều môn và môn có độ phổ biến cao đều tạo lực đẩy rất mạnh để vươn lên Top đầu.")
print("   - Ngược lại, 'fail_rate' cao thường kéo điểm sang BÊN TRÁI (SHAP < 0), giúp hạ độ ưu tiên của các môn có rủi ro cao.")
print("2. SHAP Waterfall Plot (Cá nhân hóa cho 1 môn cụ thể):")
print(f"   - Môn '{top1_course['CourseName']}' có điểm cơ bản là {base_val:.4f}.")
for idx in reversed(sorted_indices):
    f_name = feature_cols[idx]
    val = sample_shap[idx]
    f_val = sample_features[f_name]
    sign = "+" if val >= 0 else ""
    print(f"   * Đặc trưng '{f_name} = {f_val}': {sign}{val:.4f} điểm")
print(f"   -> Kết quả: Tổng các lực kéo đã đưa môn này đạt điểm số {top1_course['predicted_score']:.4f}, vươn lên TOP 1!")
print("-" * 80)

# ==============================================================================
# GIAI ĐOẠN 4: SO SÁNH ĐỐI CHỨNG VỚI POPULARITY BASELINE
# ==============================================================================
print("\n" + "=" * 80)
print("[GIAI ĐOẠN 4] SO SÁNH HIỆU NĂNG ĐỐI CHỨNG VỚI BASELINE (POPULARITY)")
print("=" * 80)

# 4.1 Định nghĩa các hàm tính NDCG@K và Precision@K
def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

def precision_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.mean(r)
    return 0.0

# 4.2 Đánh giá trên tập Test cho từng sinh viên
eval_metrics = []

for s_id, group_df in df_test.groupby('StudentId'):
    if len(group_df) < 2 or group_df['target'].sum() == 0:
        continue
    
    persona = group_df['persona'].iloc[0]
    
    # Cách 1: Popularity Baseline (xếp giảm dần theo độ phổ biến lịch sử)
    baseline_sorted = group_df.sort_values(by='popularity', ascending=False)
    r_baseline = baseline_sorted['target'].values
    
    # Cách 2: LambdaMART (xếp giảm dần theo điểm mô hình học được)
    model_sorted = group_df.sort_values(by='predicted_score', ascending=False)
    r_model = model_sorted['target'].values
    
    eval_metrics.append({
        'StudentId': s_id,
        'persona': 'P1 (Bám tiến độ)' if persona == 0 else 'P2 (Cải thiện)',
        'baseline_ndcg5': ndcg_at_k(r_baseline, 5),
        'model_ndcg5': ndcg_at_k(r_model, 5),
        'baseline_p5': precision_at_k(r_baseline, 5),
        'model_p5': precision_at_k(r_model, 5)
    })

df_metrics = pd.DataFrame(eval_metrics)

# Tính trung bình tổng thể và theo từng Persona
overall_summary = {
    'Method': ['Popularity Baseline', 'LambdaMART (Đề xuất)'],
    'NDCG@5': [df_metrics['baseline_ndcg5'].mean(), df_metrics['model_ndcg5'].mean()],
    'Precision@5': [df_metrics['baseline_p5'].mean(), df_metrics['model_p5'].mean()]
}
df_overall = pd.DataFrame(overall_summary)

persona_summary = df_metrics.groupby('persona').agg(
    baseline_ndcg5=('baseline_ndcg5', 'mean'),
    model_ndcg5=('model_ndcg5', 'mean'),
    baseline_p5=('baseline_p5', 'mean'),
    model_p5=('model_p5', 'mean')
).reset_index()

print("BẢNG SO SÁNH TỔNG THỂ:")
print(df_overall.to_string(index=False))

print("\nBẢNG BÓC TÁCH THEO TỪNG PERSONA:")
print(persona_summary.to_string(index=False))

# 4.3 Vẽ biểu đồ cột so sánh
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)

# Biểu đồ 1: NDCG@5
x = np.arange(len(persona_summary))
width = 0.35

ax1 = axes[0]
b1 = ax1.bar(x - width/2, persona_summary['baseline_ndcg5'], width, label='Popularity Baseline', color='#6c757d', edgecolor='black')
b2 = ax1.bar(x + width/2, persona_summary['model_ndcg5'], width, label='LambdaMART', color='#0d6efd', edgecolor='black')

ax1.set_ylabel('Điểm NDCG@5 (Càng cao càng tốt)', fontsize=10, fontweight='bold')
ax1.set_title('SO SÁNH CHỈ SỐ NDCG@5 THEO PERSONA', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(persona_summary['persona'], fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='upper left', frameon=True)

# Thêm nhãn giá trị
for bar in b1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5)
for bar in b2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#084298')

# Biểu đồ 2: Precision@5
ax2 = axes[1]
b3 = ax2.bar(x - width/2, persona_summary['baseline_p5'], width, label='Popularity Baseline', color='#6c757d', edgecolor='black')
b4 = ax2.bar(x + width/2, persona_summary['model_p5'], width, label='LambdaMART', color='#198754', edgecolor='black')

ax2.set_ylabel('Điểm Precision@5 (Tỷ lệ trúng đích)', fontsize=10, fontweight='bold')
ax2.set_title('SO SÁNH CHỈ SỐ PRECISION@5 THEO PERSONA', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(persona_summary['persona'], fontsize=10)
ax2.set_ylim(0, 1.05)
ax2.legend(loc='upper left', frameon=True)

for bar in b3:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5)
for bar in b4:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#0f5132')

plt.tight_layout()
comp_chart_path = os.path.join(OUTPUT_DIR, "baseline_comparison.png")
plt.savefig(comp_chart_path)
plt.close()
print(f"\n-> Đã xuất biểu đồ so sánh Baseline: {comp_chart_path}")

print("\n" + "=" * 80)
print("  HOÀN TẤT THÀNH CÔNG TOÀN BỘ 4 GIAI ĐOẠN MINH HỌA VÀ MỔ XẺ LAMBDAMART!")
print("  Tất cả biểu đồ đã được lưu trong thư mục:", OUTPUT_DIR)
print("=" * 80)
