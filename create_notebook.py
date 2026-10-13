# -*- coding: utf-8 -*-
import json

notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.14.2'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

def add_md(source):
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in source.split('\n')]
    })

def add_code(source):
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in source.split('\n')]
    })

add_md('''# ĐỒ ÁN: HỆ THỐNG GỢI Ý HỌC PHẦN CÁ NHÂN HÓA (LEARNING-TO-RANK LAMBDAMART)
## MỤC TIÊU: MỔ XẺ BẢN CHẤT CƠ CHẾ HOẠT ĐỘNG VÀ TRỰC QUAN HÓA (EXPLAINABILITY NOTEBOOK)

Notebook này được thiết kế theo đúng quy trình 4 bước nhằm phục vụ **báo cáo và phản biện với Giảng viên hướng dẫn**. Thay vì xem mô hình như một 'hộp đen' (black box), chúng ta mổ xẻ tường tận:
1. **Giai đoạn 1**: Chuẩn bị nhóm sinh viên mẫu (Khoa học máy tính - 140 SV), xây dựng Đồ thị có hướng (DAG) tiên quyết để tính `bottleneck_score`, và tạo Bảng ma trận đặc trưng.
2. **Giai đoạn 2**: Huấn luyện mô hình LambdaMART (LightGBM Ranker) và trực quan hóa Cây quyết định đầu tiên (Tree 0) để thấy rõ thuật toán rẽ nhánh cộng/trừ điểm thế nào.
3. **Giai đoạn 3**: Mổ xẻ cơ chế giải thích bằng SHAP (SHAP Beeswarm toàn cục & SHAP Waterfall cho môn Top 1 của một sinh viên cụ thể).
4. **Giai đoạn 4**: So sánh đối chứng trực quan với Baseline truyền thống (Popularity) trên các chỉ số NDCG@5 và Precision@5 (tổng thể và bóc tách theo Persona P1, P2).''')

add_code('''import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx
import lightgbm as lgb
import shap

# Thiết lập bảng mã và giao diện biểu đồ
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("Đã nạp đầy đủ các thư viện phân tích và trực quan hóa!")''')

add_md('''---
## GIAI ĐOẠN 1: CHUẨN BỊ 1 NHÓM SINH VIÊN MẪU & BẢNG ĐẶC TRƯNG TRỰC QUAN
- Chọn ngành: **Khoa học máy tính (Computer Science and Technology)** (140 sinh viên).
- Phân chia thời gian: Train (<= năm 19-20), Test (học kỳ 1 năm 20-21).
- Xây dựng đồ thị DAG để tính `bottleneck_score` (số môn mở khóa phía sau).''')

add_code('''# 1.1 Tải dữ liệu từ 3 file CSV
df_students = pd.read_csv('translated_StudentInformationTable.csv')
df_courses = pd.read_csv('translated_CourseInformationTable.csv')
df_selections = pd.read_csv('translated_CourseSelectionTable.csv')
df_selections.rename(columns={'StudedntId': 'StudentId'}, inplace=True)

# 1.2 Lọc nhóm ngành Khoa học máy tính
TARGET_MAJOR = 'Computer Science and Technology'
selected_students = df_students[df_students['Major'] == TARGET_MAJOR]['StudentId'].unique()
cs_selections = df_selections[df_selections['StudentId'].isin(selected_students)].copy()
print(f"Ngành nghiên cứu: {TARGET_MAJOR} | Số sinh viên: {len(selected_students)}")''')

add_code('''# 1.3 Xây dựng đồ thị DAG & tính bottleneck_score
name_to_id = {}
id_to_name = {}
for _, row in df_courses.iterrows():
    if pd.notna(row['Name']):
        cname = str(row['Name']).strip()
        cid = int(row['CourseId'])
        name_to_id[cname.lower()] = cid
        id_to_name[cid] = cname

dag = nx.DiGraph()
for cid in df_courses['CourseId'].unique():
    dag.add_node(int(cid))

for _, row in df_courses.iterrows():
    cid = int(row['CourseId'])
    prereq = str(row['Prerequisite'])
    if pd.notna(row['Prerequisite']) and prereq not in ['', 'None', 'nan']:
        prereq_lower = prereq.lower()
        for target_name, target_id in name_to_id.items():
            if len(target_name) >= 5 and target_name in prereq_lower and target_id != cid:
                dag.add_edge(target_id, cid)

bottleneck_scores = {}
for node in dag.nodes():
    try:
        descendants = nx.descendants(dag, node)
        bottleneck_scores[node] = len(descendants)
    except Exception:
        bottleneck_scores[node] = 0

# Tính tỷ lệ trượt (fail_rate) và độ phổ biến (popularity)
course_stats = df_selections.groupby('CourseId').agg(
    total_enrollments=('StudentId', 'count'),
    failed_enrollments=('Score', lambda x: (x < 60.0).sum())
).reset_index()
course_stats['fail_rate'] = (course_stats['failed_enrollments'] / course_stats['total_enrollments']).fillna(0.0)
max_pop = course_stats['total_enrollments'].max()
course_stats['popularity'] = course_stats['total_enrollments'] / (max_pop if max_pop > 0 else 1.0)
course_stats_map = course_stats.set_index('CourseId').to_dict(orient='index')

print(f"Đồ thị DAG hoàn tất: {dag.number_of_nodes()} môn học, {dag.number_of_edges()} ràng buộc tiên quyết.")''')

add_code('''# Trực quan hóa một phần đồ thị DAG tiên quyết
fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
top_nodes = [node for node, score in sorted(bottleneck_scores.items(), key=lambda x: x[1], reverse=True)[:18] if score > 0]
subgraph = dag.subgraph(top_nodes)

pos = nx.spring_layout(subgraph, seed=42, k=1.2)
node_sizes = [600 + bottleneck_scores[n] * 400 for n in subgraph.nodes()]
node_colors = [bottleneck_scores[n] for n in subgraph.nodes()]

nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=node_sizes, node_color=node_colors, 
                       cmap=plt.cm.YlOrRd, edgecolors='black', linewidths=1.5)
nx.draw_networkx_edges(subgraph, pos, ax=ax, arrowstyle='->', arrowsize=18, edge_color='gray', width=1.5)

node_labels = {n: f"{id_to_name.get(n, str(n))[:18]}\\n[Mở: {bottleneck_scores[n]} môn]" for n in subgraph.nodes()}
nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=8, font_family='sans-serif', ax=ax)

ax.set_title("ĐỒ THỊ MINH HỌA RÀNG BUỘC TIÊN QUYẾT (DAG) & CHỈ SỐ NÚT THẮT (BOTTLENECK SCORE)", fontsize=12, fontweight='bold', pad=15)
ax.axis('off')
plt.tight_layout()
plt.show()''')

add_code('''# 1.4 Tạo Candidate Set & Bảng Ma trận đặc trưng
train_selections = cs_selections[cs_selections['AcademicYear'] <= '19-20'].copy()
test_selections = cs_selections[(cs_selections['AcademicYear'] == '20-21') & (cs_selections['Semester'] == 1.0)].copy()

popular_cs_courses = cs_selections['CourseId'].value_counts().head(50).index.tolist()

def get_student_persona_and_gpa(student_id, history_df):
    stud_history = history_df[history_df['StudentId'] == student_id]
    if len(stud_history) == 0:
        return 0, 75.0, 0
    valid_scores = stud_history['Score'].dropna()
    gpa = valid_scores.mean() if len(valid_scores) > 0 else 75.0
    failed_count = (valid_scores < 60.0).sum()
    persona = 1 if (failed_count > 0 or gpa < 75.0) else 0
    return persona, round(gpa, 2), failed_count

def build_candidate_dataset(eval_selections, history_selections):
    records = []
    student_ids = eval_selections['StudentId'].unique()
    for s_id in student_ids:
        persona, gpa, failed_count = get_student_persona_and_gpa(s_id, history_selections)
        actual_courses = set(eval_selections[eval_selections['StudentId'] == s_id]['CourseId'].unique())
        past_courses = set(history_selections[history_selections['StudentId'] == s_id]['CourseId'].unique())
        unlearned_candidates = [c for c in popular_cs_courses if c not in past_courses]
        negative_candidates = [c for c in unlearned_candidates if c not in actual_courses][:10]
        all_candidates = list(actual_courses) + negative_candidates
        
        for c_id in all_candidates:
            c_info = course_stats_map.get(c_id, {'fail_rate': 0.05, 'popularity': 0.1})
            c_name = id_to_name.get(c_id, f"Môn học {c_id}")
            b_score = bottleneck_scores.get(c_id, 0)
            target = 1 if c_id in actual_courses else 0
            is_core = 1 if ('Computer' in c_name or 'Algorithm' in c_name or 'Data' in c_name or 'System' in c_name) else 0
            records.append({
                'StudentId': s_id,
                'CourseId': c_id,
                'CourseName': c_name,
                'bottleneck_score': b_score,
                'fail_rate': round(c_info['fail_rate'], 3),
                'popularity': round(c_info['popularity'], 3),
                'student_gpa': gpa,
                'is_core_major': is_core,
                'persona': persona,
                'target': target
            })
    return pd.DataFrame(records)

train_eval = train_selections[(train_selections['AcademicYear'] == '19-20') & (train_selections['Semester'] == 1.0)]
train_history = train_selections[train_selections['AcademicYear'] < '19-20']
df_train = build_candidate_dataset(train_eval, train_history)

test_history = cs_selections[cs_selections['AcademicYear'] <= '19-20']
df_test = build_candidate_dataset(test_selections, test_history)

# In bảng dữ liệu mẫu của 1 sinh viên cụ thể
sample_sid = df_train['StudentId'].iloc[0]
print(f"BẢNG MA TRẬN ĐẶC TRƯNG MẪU (FEATURE MATRIX) CHO SINH VIÊN {sample_sid}:")
display(df_train[df_train['StudentId'] == sample_sid][
    ['StudentId', 'CourseName', 'bottleneck_score', 'fail_rate', 'popularity', 'persona', 'target']
].head(6))''')

add_md('''---
## GIAI ĐOẠN 2: HUẤN LUYỆN LAMBDAMART & TRỰC QUAN HÓA CÂY QUYẾT ĐỊNH
- Huấn luyện `lightgbm.LGBMRanker` với hàm mục tiêu `lambdarank`.
- Vẽ Cây quyết định đầu tiên (Tree 0) để thấy rõ điều kiện rẽ nhánh và điểm số lá.''')

add_code('''feature_cols = ['bottleneck_score', 'fail_rate', 'popularity', 'student_gpa', 'is_core_major', 'persona']
df_train.sort_values(by=['StudentId'], inplace=True)
df_test.sort_values(by=['StudentId'], inplace=True)

X_train = df_train[feature_cols]
y_train = df_train['target']
train_groups = df_train.groupby('StudentId').size().values

X_test = df_test[feature_cols]
y_test = df_test['target']
test_groups = df_test.groupby('StudentId').size().values

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
print("Huấn luyện LambdaMART hoàn tất thành công!")''')

add_code('''# Trực quan hóa Cây Quyết Định Tree 0
tree_info = ranker.booster_.dump_model()['tree_info'][0]

fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

def parse_node(node, x, y, dx, dy, depth=0):
    if 'leaf_value' in node:
        val = node['leaf_value']
        bg_color = '#d4edda' if val >= 0 else '#f8d7da'
        border_color = '#28a745' if val >= 0 else '#dc3545'
        sign_str = f"+{val:.4f}" if val >= 0 else f"{val:.4f}"
        box = patches.FancyBboxPatch((x - 9, y - 4), 18, 8, boxstyle="round,pad=0.6,rounding_size=2", facecolor=bg_color, edgecolor=border_color, linewidth=2.0)
        ax.add_patch(box)
        ax.text(x, y + 1.2, "LÁ (Leaf)", ha='center', va='center', fontsize=9, fontweight='bold', color='#155724' if val>=0 else '#721c24')
        ax.text(x, y - 1.8, f"Điểm ưu tiên:\\n{sign_str}", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#155724' if val>=0 else '#721c24')
        return
    f_idx = node['split_feature']
    f_name = feature_cols[f_idx]
    thresh = node['threshold']
    box = patches.FancyBboxPatch((x - 11, y - 4.5), 22, 9, boxstyle="round,pad=0.6,rounding_size=2", facecolor='#e8f4fd', edgecolor='#0d6efd', linewidth=2.0)
    ax.add_patch(box)
    ax.text(x, y + 1.8, "Điều kiện tách nhánh:", ha='center', va='center', fontsize=8.5, color='#495057')
    ax.text(x, y - 1.5, f"{f_name} <= {thresh:.2f}", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#084298')
    
    left_x, left_y = x - dx, y - dy
    ax.annotate("", xy=(left_x, left_y + 4.5), xytext=(x - 3, y - 4.5), arrowprops=dict(arrowstyle="->", color="#198754", lw=2.2))
    ax.text((x + left_x)/2 - 3, (y + left_y)/2, "Đúng (<=)", fontsize=8, color="#198754", fontweight='bold')
    parse_node(node['left_child'], left_x, left_y, dx * 0.52, dy * 0.9, depth + 1)
    
    right_x, right_y = x + dx, y - dy
    ax.annotate("", xy=(right_x, right_y + 4.5), xytext=(x + 3, y - 4.5), arrowprops=dict(arrowstyle="->", color="#dc3545", lw=2.2))
    ax.text((x + right_x)/2 + 3, (y + right_y)/2, "Sai (>)", fontsize=8, color="#dc3545", fontweight='bold')
    parse_node(node['right_child'], right_x, right_y, dx * 0.52, dy * 0.9, depth + 1)

parse_node(tree_info['tree_structure'], x=50, y=90, dx=24, dy=26)
ax.set_title("CÂY QUYẾT ĐỊNH ĐẦU TIÊN (TREE 0) TRONG MÔ HÌNH LAMBDAMART\\n(Minh họa cơ chế cộng/trừ điểm xếp hạng qua từng nhánh điều kiện)", fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()''')

add_md('''---
## GIAI ĐOẠN 3: "MỔ XẺ" CƠ CHẾ GIẢI THÍCH BẰNG SHAP (EXPLAINABILITY)
- Biểu đồ **SHAP Beeswarm Plot**: Đánh giá toàn cục các đặc trưng quyết định thứ bậc.
- Biểu đồ **SHAP Waterfall Plot**: Giải thích chi tiết môn học Top 1 của một sinh viên được cộng/trừ bao nhiêu điểm từ các đặc trưng.''')

add_code('''explainer = shap.TreeExplainer(ranker.booster_)
shap_values = explainer(X_test)

# SHAP Summary Beeswarm Plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP BEESWARM PLOT: MỨC ĐỘ ẢNH HƯỞNG CỦA CÁC ĐẶC TRƯNG ĐẾN ĐIỂM XẾP HẠNG", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Giá trị SHAP (Tác động đẩy tăng điểm [+] hoặc kéo giảm điểm [-])", fontsize=10)
plt.tight_layout()
plt.show()''')

add_code('''# SHAP Waterfall Plot cho môn học Top 1 của một sinh viên
df_test['predicted_score'] = ranker.predict(X_test)
target_sid = df_test['StudentId'].unique()[1]
student_rec = df_test[df_test['StudentId'] == target_sid].sort_values(by='predicted_score', ascending=False)
top1_course = student_rec.iloc[0]
test_row_idx = df_test.index.get_loc(student_rec.index[0])

sample_shap = shap_values[test_row_idx].values
base_val = shap_values[test_row_idx].base_values
sample_features = X_test.iloc[test_row_idx]
sorted_indices = np.argsort(np.abs(sample_shap))
y_positions = np.arange(len(feature_cols))
colors = ['#28a745' if val >= 0 else '#dc3545' for val in sample_shap[sorted_indices]]

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
bars = ax.barh(y_positions, sample_shap[sorted_indices], color=colors, height=0.6, edgecolor='black', linewidth=0.8)

for idx, bar in enumerate(bars):
    val = sample_shap[sorted_indices][idx]
    f_name = feature_cols[sorted_indices[idx]]
    f_val = sample_features[f_name]
    sign = "+" if val >= 0 else ""
    text_x = val + (0.01 if val >= 0 else -0.01)
    ha_align = 'left' if val >= 0 else 'right'
    ax.text(text_x, bar.get_y() + bar.get_height()/2, f"{sign}{val:.4f}  (giá trị={f_val})", 
            va='center', ha=ha_align, fontsize=8.5, fontweight='bold', color=colors[idx])

ax.set_yticks(y_positions)
ax.set_yticklabels([f"{feature_cols[i]}" for i in sorted_indices], fontsize=9.5)
ax.axvline(0, color='black', linestyle='--', linewidth=1.0)
ax.set_title(f"SHAP WATERFALL EXPLANATION CHO MÔN HỌC TOP 1\\nSinh viên: {target_sid} | Môn: '{top1_course['CourseName']}'\\n(Điểm cơ sở E[f(x)] = {base_val:.4f} -> Điểm cuối cùng = {top1_course['predicted_score']:.4f})",
             fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Độ dịch chuyển điểm SHAP (Xanh: Tăng điểm ưu tiên, Đỏ: Hạ điểm)", fontsize=9.5)
plt.tight_layout()
plt.show()''')

add_md('''---
## GIAI ĐOẠN 4: SO SÁNH ĐỐI CHỨNG VỚI BASELINE (POPULARITY)
- Đối chứng giữa **Popularity Baseline** và **LambdaMART (Personalized)**.
- So sánh trên chỉ số xếp hạng chuẩn **NDCG@5** và **Precision@5** (tổng thể và bóc tách theo Persona P1 & P2).''')

add_code('''def dcg_at_k(r, k):
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

eval_metrics = []
for s_id, group_df in df_test.groupby('StudentId'):
    if len(group_df) < 2 or group_df['target'].sum() == 0:
        continue
    persona = group_df['persona'].iloc[0]
    
    baseline_sorted = group_df.sort_values(by='popularity', ascending=False)
    r_baseline = baseline_sorted['target'].values
    
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
persona_summary = df_metrics.groupby('persona').agg(
    baseline_ndcg5=('baseline_ndcg5', 'mean'),
    model_ndcg5=('model_ndcg5', 'mean'),
    baseline_p5=('baseline_p5', 'mean'),
    model_p5=('model_p5', 'mean')
).reset_index()

# Vẽ biểu đồ cột so sánh
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=160)
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

for bar in b1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5)
for bar in b2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#084298')

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
plt.show()''')

with open('lambdamart_visual_explanation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print('Generated lambdamart_visual_explanation.ipynb successfully!')
