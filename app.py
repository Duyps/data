# -*- coding: utf-8 -*-
"""
HỆ THỐNG GỢI Ý HỌC PHẦN CÁ NHÂN HÓA (LAMBDAMART RANKING & EXPLAINABLE AI)
WEB DEMO TRỰC QUAN: TÍCH HỢP BỘ SOI LUỒNG DỮ LIỆU & PHÉP TÍNH TOÁN HỌC (DATA PIPELINE INSPECTOR)

Chạy bằng lệnh: python app.py
Truy cập tại:   http://localhost:5000
"""

import os
import sys
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Thiết lập UTF-8
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import networkx as nx
import lightgbm as lgb
import shap

print("=" * 75)
print("  KHỞI TẠO HỆ THỐNG GỢI Ý HỌC PHẦN CÁ NHÂN HÓA (LAMBDAMART & PIPELINE INSPECTOR)")
print("=" * 75)

# 1. ĐỌC DỮ LIỆU
print("[1/4] Đang tải dữ liệu từ CSV...")
df_students = pd.read_csv("translated_StudentInformationTable.csv")
df_courses = pd.read_csv("translated_CourseInformationTable.csv")
df_selections = pd.read_csv("translated_CourseSelectionTable.csv")
df_selections.rename(columns={'StudedntId': 'StudentId'}, inplace=True)

TARGET_MAJOR = "Computer Science and Technology"
selected_students = df_students[df_students['Major'] == TARGET_MAJOR]['StudentId'].unique()
cs_selections = df_selections[df_selections['StudentId'].isin(selected_students)].copy()

# 2. XÂY DỰNG ĐỒ THỊ DAG VÀ TÍNH BOTTLENECK SCORE
print("[2/4] Đang xây dựng đồ thị DAG tiên quyết và các đặc trưng...")
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
descendant_map = {}
for node in dag.nodes():
    try:
        descendants = nx.descendants(dag, node)
        bottleneck_scores[node] = len(descendants)
        descendant_map[node] = [id_to_name.get(d, str(d)) for d in list(descendants)[:4]]
    except Exception:
        bottleneck_scores[node] = 0
        descendant_map[node] = []

course_stats = df_selections.groupby('CourseId').agg(
    total_enrollments=('StudentId', 'count'),
    failed_enrollments=('Score', lambda x: (x < 60.0).sum())
).reset_index()
course_stats['fail_rate'] = (course_stats['failed_enrollments'] / course_stats['total_enrollments']).fillna(0.0)
max_pop = course_stats['total_enrollments'].max()
course_stats['popularity'] = course_stats['total_enrollments'] / (max_pop if max_pop > 0 else 1.0)
course_stats_map = course_stats.set_index('CourseId').to_dict(orient='index')

# Phân chia dữ liệu Train (<= 19-20) và Test (20-21 kỳ 1)
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
            c_info = course_stats_map.get(c_id, {'fail_rate': 0.05, 'popularity': 0.1, 'total_enrollments': 100, 'failed_enrollments': 5})
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
                'target': target,
                'total_enrollments': c_info.get('total_enrollments', 100),
                'failed_enrollments': c_info.get('failed_enrollments', 5)
            })
    return pd.DataFrame(records)

train_eval = train_selections[(train_selections['AcademicYear'] == '19-20') & (train_selections['Semester'] == 1.0)]
train_history = train_selections[train_selections['AcademicYear'] < '19-20']
df_train = build_candidate_dataset(train_eval, train_history)

test_history = cs_selections[cs_selections['AcademicYear'] <= '19-20']
df_test = build_candidate_dataset(test_selections, test_history)

# 3. HUẤN LUYỆN LAMBDAMART
print("[3/4] Đang huấn luyện mô hình LightGBM LambdaMART...")
feature_cols = ['bottleneck_score', 'fail_rate', 'popularity', 'student_gpa', 'is_core_major', 'persona']
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
ranker.fit(X_train, y_train, group=train_groups)

df_test['predicted_score'] = ranker.predict(X_test)

# 4. KHỞI TẠO SHAP EXPLAINER
print("[4/4] Đang khởi tạo SHAP TreeExplainer...")
explainer = shap.TreeExplainer(ranker.booster_)
shap_values = explainer(X_test)

student_list = []
test_student_ids = sorted(df_test['StudentId'].unique())

for s_id in test_student_ids:
    sub = df_test[df_test['StudentId'] == s_id]
    p = int(sub['persona'].iloc[0])
    gpa = float(sub['student_gpa'].iloc[0])
    student_list.append({
        'student_id': int(s_id),
        'gpa': gpa,
        'persona': p,
        'persona_label': "P1 (Bám tiến độ)" if p == 0 else "P2 (Cần cải thiện)",
        'num_candidates': len(sub)
    })

def get_recommendation_and_pipeline(s_id):
    sub = df_test[df_test['StudentId'] == s_id].copy()
    if len(sub) == 0:
        return None
    
    # 1. Thông tin lịch sử học tập
    stud_history = test_history[test_history['StudentId'] == s_id].copy()
    history_sample = []
    for _, h in stud_history.head(6).iterrows():
        history_sample.append({
            'academic_year': str(h.get('AcademicYear', '19-20')),
            'semester': int(h.get('Semester', 1)),
            'course_name': str(h.get('CourseName', '')),
            'score': float(h.get('Score', 0)) if pd.notna(h.get('Score')) else None
        })
    
    valid_scores = stud_history['Score'].dropna()
    total_credits = len(stud_history)
    failed_count = int((valid_scores < 60.0).sum())
    gpa_val = round(float(valid_scores.mean()), 2) if len(valid_scores) > 0 else 75.0
    persona_val = 1 if (failed_count > 0 or gpa_val < 75.0) else 0

    # 2. Xếp hạng LambdaMART & SHAP
    sub_lm = sub.sort_values(by='predicted_score', ascending=False)
    sub_base = sub.sort_values(by='popularity', ascending=False)
    
    lm_results = []
    feature_matrix_sample = []
    
    for rank, (idx, row) in enumerate(sub_lm.iterrows(), start=1):
        test_row_idx = df_test.index.get_loc(idx)
        shap_vals = shap_values[test_row_idx].values
        base_val = float(shap_values[test_row_idx].base_values)
        
        shap_breakdown = []
        for f_name, f_val, s_val in zip(feature_cols, row[feature_cols], shap_vals):
            shap_breakdown.append({
                'feature': f_name,
                'feature_val': float(f_val),
                'shap_val': round(float(s_val), 4)
            })
            
        unlocked = descendant_map.get(int(row['CourseId']), [])
        
        item_data = {
            'rank': rank,
            'course_id': int(row['CourseId']),
            'course_name': row['CourseName'],
            'score': round(float(row['predicted_score']), 4),
            'bottleneck_score': int(row['bottleneck_score']),
            'unlocked_courses': unlocked,
            'fail_rate': round(float(row['fail_rate']) * 100, 1),
            'raw_failed': int(row.get('failed_enrollments', 0)),
            'raw_total': int(row.get('total_enrollments', 100)),
            'popularity': round(float(row['popularity']), 3),
            'is_core': int(row['is_core_major']),
            'actually_registered': int(row['target']),
            'base_value': round(base_val, 4),
            'shap_breakdown': shap_breakdown
        }
        lm_results.append(item_data)
        
        feature_matrix_sample.append({
            'course_name': row['CourseName'],
            'bottleneck_score': int(row['bottleneck_score']),
            'fail_rate': round(float(row['fail_rate']), 3),
            'popularity': round(float(row['popularity']), 3),
            'student_gpa': gpa_val,
            'is_core': int(row['is_core_major']),
            'persona': persona_val,
            'target': int(row['target'])
        })
        
    base_results = []
    for rank, (idx, row) in enumerate(sub_base.iterrows(), start=1):
        base_results.append({
            'rank': rank,
            'course_id': int(row['CourseId']),
            'course_name': row['CourseName'],
            'popularity': round(float(row['popularity']), 3),
            'actually_registered': int(row['target'])
        })
        
    return {
        'student_id': int(s_id),
        'gpa': gpa_val,
        'persona': persona_val,
        'persona_label': "P1: Bám tiến độ (GPA ≥ 75, không nợ môn)" if persona_val == 0 else "P2: Cần cải thiện điểm (Có môn nợ hoặc GPA < 75)",
        'total_past_courses': total_credits,
        'failed_count': failed_count,
        'history_sample': history_sample,
        'feature_matrix': feature_matrix_sample,
        'lambdamart_top5': lm_results[:5],
        'baseline_top5': base_results[:5],
        'total_candidates': len(sub)
    }

# GIAO DIỆN HTML/CSS/JS VỚI TAB SOI LUỒNG DỮ LIỆU
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ Thống Gợi Ý Học Phần Cá Nhân Hóa (LambdaMART & Data Pipeline Inspector)</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #eff6ff;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --gray-50: #f8fafc;
            --gray-100: #f1f5f9;
            --gray-200: #e2e8f0;
            --gray-300: #cbd5e1;
            --gray-700: #334155;
            --gray-800: #1e293b;
            --gray-900: #0f172a;
            --card-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.06);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background: #f8fafc; color: var(--gray-900); padding-bottom: 60px; }
        
        .header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white; padding: 24px 40px; border-bottom: 3px solid var(--primary);
            display: flex; align-items: center; justify-content: space-between;
        }
        .header h1 { font-size: 22px; font-weight: 800; }
        .header p { color: #94a3b8; font-size: 13px; margin-top: 4px; }
        
        .container { max-width: 1300px; margin: 20px auto; padding: 0 20px; }
        
        /* Navigation Tabs */
        .tabs { display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 2px solid var(--gray-200); padding-bottom: 2px; }
        .tab-btn {
            padding: 12px 20px; border: none; background: transparent; font-size: 14px;
            font-weight: 700; color: #64748b; cursor: pointer; border-radius: 8px 8px 0 0;
            display: flex; align-items: center; gap: 8px; transition: all 0.15s;
        }
        .tab-btn:hover { color: var(--primary); background: #f1f5f9; }
        .tab-btn.active { color: var(--primary); border-bottom: 3px solid var(--primary); background: white; }
        
        .tab-pane { display: none; }
        .tab-pane.active { display: block; animation: fadeIn 0.2s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

        /* Control Bar */
        .control-card {
            background: white; border-radius: 12px; padding: 18px 24px;
            box-shadow: var(--card-shadow); display: flex; align-items: center; justify-content: space-between;
            gap: 20px; margin-bottom: 20px; border: 1px solid var(--gray-200);
        }
        .control-group { display: flex; align-items: center; gap: 12px; }
        label { font-weight: 600; font-size: 14px; color: var(--gray-700); }
        select {
            padding: 10px 16px; border: 1.5px solid var(--gray-300); border-radius: 8px;
            font-size: 14px; font-weight: 600; outline: none; background: white; cursor: pointer; min-width: 280px;
        }
        select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,0.15); }
        .quick-btn {
            padding: 8px 14px; border: 1px solid var(--gray-200); background: #f8fafc;
            border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s;
        }
        .quick-btn:hover { background: #e2e8f0; color: var(--primary); }
        
        /* Profile Banner */
        .profile-banner {
            background: white; border-radius: 12px; padding: 18px 24px;
            box-shadow: var(--card-shadow); margin-bottom: 24px;
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
            border-left: 5px solid var(--primary);
        }
        .profile-item h4 { font-size: 12px; text-transform: uppercase; color: #64748b; margin-bottom: 4px; }
        .profile-item p { font-size: 18px; font-weight: 800; color: var(--gray-900); }
        
        .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }
        .badge-p1 { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
        .badge-p2 { background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }
        .badge-target { background: #dcfce7; color: #166534; font-size: 11px; }

        /* Grid */
        .grid-2 { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 24px; }
        .column-card {
            background: white; border-radius: 12px; padding: 22px;
            box-shadow: var(--card-shadow); border: 1px solid var(--gray-200);
        }
        .column-title {
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid var(--gray-100);
        }
        .column-title h2 { font-size: 16px; font-weight: 700; }

        /* Course Cards */
        .course-card {
            background: #ffffff; border: 1.5px solid var(--gray-200);
            border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;
            transition: all 0.2s;
        }
        .course-card:hover { border-color: var(--primary); box-shadow: 0 6px 12px -2px rgba(37,99,235,0.1); transform: translateY(-1px); }
        .course-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
        .rank-circle {
            width: 26px; height: 26px; border-radius: 50%; background: #1e293b; color: white;
            display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 12px;
        }
        .rank-circle.top1 { background: #eab308; }
        .course-name { font-size: 14.5px; font-weight: 700; color: #0f172a; margin-left: 10px; flex-grow: 1; }
        .metric-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
        .tag {
            background: var(--gray-100); padding: 4px 8px; border-radius: 6px;
            font-size: 11.5px; font-weight: 600; color: #475569; display: flex; align-items: center; gap: 4px;
        }
        .tag-bottleneck { background: #fee2e2; color: #991b1b; }
        .tag-safe { background: #ecfdf5; color: #065f46; }

        .btn-explain {
            margin-top: 10px; background: #eff6ff; color: var(--primary); border: 1px solid #bfdbfe;
            padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 700; cursor: pointer;
            transition: all 0.15s; width: 100%; text-align: center;
        }
        .btn-explain:hover { background: var(--primary); color: white; }

        /* =================== PIPELINE INSPECTOR STYLES =================== */
        .pipeline-step {
            background: white; border-radius: 12px; padding: 22px; margin-bottom: 22px;
            border: 1px solid var(--gray-200); box-shadow: var(--card-shadow);
            position: relative;
        }
        .step-header { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
        .step-num {
            width: 32px; height: 32px; border-radius: 8px; background: var(--primary); color: white;
            display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px;
        }
        .step-title { font-size: 16px; font-weight: 700; color: var(--gray-900); }
        .formula-box {
            background: #f8fafc; border-left: 4px solid var(--primary); padding: 12px 18px;
            border-radius: 0 8px 8px 0; font-family: 'Fira Code', monospace; font-size: 13px;
            color: #1e293b; margin: 12px 0;
        }
        .math-result {
            background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 16px;
            margin-top: 10px; font-size: 13.5px; color: #166534; font-weight: 600;
        }
        
        table.data-table {
            width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px;
        }
        table.data-table th, table.data-table td {
            padding: 8px 12px; border: 1px solid var(--gray-200); text-align: left;
        }
        table.data-table th { background: #f8fafc; font-weight: 700; color: #475569; }
        table.data-table tr:hover { background: #f1f5f9; }

        /* Gallery */
        .gallery-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .gallery-card {
            background: white; border-radius: 12px; padding: 18px; border: 1px solid var(--gray-200);
            box-shadow: var(--card-shadow);
        }
        .gallery-card h3 { font-size: 15px; margin-bottom: 10px; color: #0f172a; }
        .gallery-card img { width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; }

        /* Modal SHAP */
        .modal {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.6); z-index: 1000; align-items: center; justify-content: center;
        }
        .modal-content {
            background: white; border-radius: 14px; max-width: 650px; width: 90%; padding: 26px;
            box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.2); animation: popIn 0.2s ease-out;
        }
        @keyframes popIn { from { transform: scale(0.95); opacity: 1; } }
        .shap-bar-row { display: flex; align-items: center; margin: 10px 0; font-size: 13px; }
        .shap-name { width: 160px; font-weight: 600; color: #334155; }
        .shap-bar-track { flex-grow: 1; height: 18px; background: #f1f5f9; border-radius: 4px; margin: 0 12px; }
        .shap-fill { height: 100%; border-radius: 4px; }
        .shap-fill.pos { background: #16a34a; }
        .shap-fill.neg { background: #dc2626; }
        .shap-val { width: 70px; text-align: right; font-weight: 700; font-family: monospace; }
        .close-btn {
            background: #1e293b; color: white; border: none; padding: 8px 18px;
            border-radius: 6px; font-weight: 600; cursor: pointer; float: right; margin-top: 14px;
        }
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>HỆ THỐNG GỢI Ý HỌC PHẦN CÁ NHÂN HÓA</h1>
            <p>Mô hình LambdaMART & Minh Bạch Hóa Luồng Dữ Liệu Từng Bước (Explainable Data Pipeline)</p>
        </div>
        <div>
            <span class="badge" style="background: #2563eb; color: white; padding: 6px 14px; font-size: 13px;">Đồ Án Tốt Nghiệp</span>
        </div>
    </div>

    <div class="container">
        <!-- Control Box -->
        <div class="control-card">
            <div class="control-group">
                <label for="studentSelect">Chọn Sinh viên kiểm thử:</label>
                <select id="studentSelect" onchange="loadStudent(this.value)"></select>
            </div>
            <div class="control-group">
                <span style="font-size: 13px; color: #64748b; font-weight: 600;">Xem nhanh:</span>
                <button class="quick-btn" onclick="selectStudent(2825)">SV 2825 (P1 - Bám tiến độ)</button>
                <button class="quick-btn" onclick="selectStudent(133)">SV 133 (P2 - Cần gỡ điểm)</button>
            </div>
        </div>

        <!-- Student Profile Banner -->
        <div class="profile-banner">
            <div class="profile-item">
                <h4>Mã Sinh Viên</h4>
                <p id="profId">---</p>
            </div>
            <div class="profile-item">
                <h4>Điểm tích lũy (GPA)</h4>
                <p id="profGpa">---</p>
            </div>
            <div class="profile-item">
                <h4>Nhóm Persona</h4>
                <div id="profPersona" style="margin-top: 4px;">---</div>
            </div>
            <div class="profile-item">
                <h4>Số môn ứng viên (Kỳ 20-21 K1)</h4>
                <p id="profCandidates">---</p>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('tab-recommend')">
                🎯 Bảng Gợi Ý & Xếp Hạng
            </button>
            <button class="tab-btn" onclick="switchTab('tab-pipeline')">
                🔬 Luồng Xử Lý & Phép Tính Dữ Liệu (Data Pipeline Inspector)
            </button>
            <button class="tab-btn" onclick="switchTab('tab-gallery')">
                📈 Kho Biểu Đồ Minh Họa (Visual Artifacts)
            </button>
        </div>

        <!-- TAB 1: RECOMMENDATIONS -->
        <div id="tab-recommend" class="tab-pane active">
            <div class="grid-2">
                <!-- LambdaMART -->
                <div class="column-card" style="border-top: 4px solid var(--primary);">
                    <div class="column-title">
                        <h2>🎯 Đề Xuất Bởi LambdaMART (Cá Nhân Hóa)</h2>
                        <span class="badge" style="background:#dbeafe; color:#1e40af;">NDCG@5: 0.744</span>
                    </div>
                    <div id="lambdamartList">Đang tải...</div>
                </div>

                <!-- Baseline -->
                <div class="column-card" style="border-top: 4px solid #64748b; background: #fafafa;">
                    <div class="column-title">
                        <h2 style="color: #475569;">📊 Popularity Baseline (Theo Số Đông)</h2>
                        <span class="badge" style="background:#f1f5f9; color:#475569;">NDCG@5: 0.278</span>
                    </div>
                    <div id="baselineList">Đang tải...</div>
                </div>
            </div>
        </div>

        <!-- TAB 2: PIPELINE & MATH INSPECTOR -->
        <div id="tab-pipeline" class="tab-pane">
            
            <!-- Bước 1: Dữ liệu thô -->
            <div class="pipeline-step">
                <div class="step-header">
                    <div class="step-num">1</div>
                    <div class="step-title">Dữ Liệu Thô Đầu Vào (Raw Data Ingestion)</div>
                </div>
                <p style="font-size: 13.5px; color: #475569;">
                    Hệ thống trích xuất lịch sử đăng ký của sinh viên đang chọn từ file <code>translated_CourseSelectionTable.csv</code>:
                </p>
                <div style="overflow-x: auto;">
                    <table class="data-table" id="tableRawHistory">
                        <thead>
                            <tr>
                                <th>Học kỳ</th>
                                <th>Môn đã từng học</th>
                                <th>Điểm số (Score / 100)</th>
                                <th>Trạng thái</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>

            <!-- Bước 2: Phép tính GPA & Persona -->
            <div class="pipeline-step">
                <div class="step-header">
                    <div class="step-num">2</div>
                    <div class="step-title">Phép Tính Học Vụ & Phân Loại Persona (Student Persona Logic)</div>
                </div>
                <p style="font-size: 13.5px; color: #475569;">
                    Máy tính chạy phép toán tổng hợp điểm và rà soát các môn có nguy cơ trượt:
                </p>
                <div class="formula-box">
                    GPA = (Σ Điểm các môn đã học) / (Tổng số môn có điểm)<br>
                    Failed_Count = Đếm(Môn có Score &lt; 60.0)<br>
                    IF (Failed_Count &gt; 0 HOẶC GPA &lt; 75.0) THEN Persona = P2 (Cần cải thiện điểm)<br>
                    ELSE Persona = P1 (Bám tiến độ)
                </div>
                <div class="math-result" id="pipelinePersonaResult">
                    Đang tính toán...
                </div>
            </div>

            <!-- Bước 3: Đồ thị DAG & Bottleneck Score -->
            <div class="pipeline-step">
                <div class="step-header">
                    <div class="step-num">3</div>
                    <div class="step-title">Đồ Thị Có Hướng (DAG) & Phép Tính Nút Thắt (Bottleneck Score)</div>
                </div>
                <p style="font-size: 13.5px; color: #475569;">
                    Từ cột <code>Prerequisite</code> của 5,591 môn học, hệ thống dựng cây quan hệ có hướng $A \rightarrow B$.
                    Chỉ số nút thắt của môn học $c$ được tính bằng kích thước tập hợp các đỉnh hậu duệ (descendants):
                </p>
                <div class="formula-box">
                    bottleneck_score(c) = |Descendants(c)| = Số lượng môn học mở khóa trực tiếp và gián tiếp phía sau
                </div>
                <div id="unlockedCoursesInfo" style="margin-top: 10px; font-size: 13px;"></div>
            </div>

            <!-- Bước 4: Thống kê lịch sử môn học -->
            <div class="pipeline-step">
                <div class="step-header">
                    <div class="step-num">4</div>
                    <div class="step-title">Phép Tính Độ Khó (Fail Rate) & Độ Phổ Biến (Popularity)</div>
                </div>
                <p style="font-size: 13.5px; color: #475569;">
                    Đo lường rủi ro của môn học dựa trên lịch sử toàn trường của các khóa trước:
                </p>
                <div class="formula-box">
                    fail_rate(c) = (Số lượt sinh viên trượt môn c) / (Tổng lượt đăng ký môn c)<br>
                    popularity(c) = (Tổng lượt đăng ký môn c) / max(Tổng lượt đăng ký toàn trường)
                </div>
            </div>

            <!-- Bước 5: Ma trận đặc trưng số nạp vào mô hình -->
            <div class="pipeline-step">
                <div class="step-header">
                    <div class="step-num">5</div>
                    <div class="step-title">Ma Trận Đặc Trưng Số (Feature Matrix) Nạp Vào LambdaMART</div>
                </div>
                <p style="font-size: 13.5px; color: #475569;">
                    Sau khi qua các phép tính trên, mỗi môn trong Candidate Set của sinh viên được nén thành 1 vector số học 6 chiều:
                </p>
                <div style="overflow-x: auto;">
                    <table class="data-table" id="tableFeatureMatrix">
                        <thead>
                            <tr>
                                <th>Tên môn ứng viên</th>
                                <th>bottleneck_score</th>
                                <th>fail_rate</th>
                                <th>popularity</th>
                                <th>student_gpa</th>
                                <th>is_core</th>
                                <th>persona</th>
                                <th>target (Thực tế)</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>

            <!-- Bước 6: Suy luận mô hình LambdaMART & SHAP -->
            <div class="pipeline-step">
                <div class="step-header">
                    <div class="step-num">6</div>
                    <div class="step-title">Suy Luận LambdaMART & Minh Bạch Hóa SHAP (Attribution Equation)</div>
                </div>
                <p style="font-size: 13.5px; color: #475569;">
                    Mỗi môn học đi qua 15 Cây Boosting (Trees). Điểm số xếp hạng và lực kéo giải thích SHAP thỏa mãn định lý:
                </p>
                <div class="formula-box">
                    Điểm_Xếp_Hạng(c) = Base_Value + Σ [SHAP_Value của từng đặc trưng]<br>
                    f(x) = E[f(x)] + φ_bottleneck + φ_fail_rate + φ_popularity + φ_gpa + φ_is_core + φ_persona
                </div>
                <div style="font-size: 13px; color: #475569;">
                    👉 Môn nào có tổng điểm <code>f(x)</code> cao nhất sẽ được đặt ở vị trí <strong>Top 1</strong>!
                </div>
            </div>

        </div>

        <!-- TAB 3: VISUAL ARTIFACTS GALLERY -->
        <div id="tab-gallery" class="tab-pane">
            <div class="gallery-grid">
                <div class="gallery-card">
                    <h3>1. Đồ Thị Tiên Quyết DAG & Nút Thắt</h3>
                    <img src="/artifacts/dag_prerequisite_tree.png" alt="DAG Tree">
                    <p style="font-size: 12.5px; color: #64748b; margin-top: 8px;">
                        Môn có chấm to màu đỏ mở khóa nhiều môn phía sau trong lộ trình đào tạo.
                    </p>
                </div>
                <div class="gallery-card">
                    <h3>2. Cây Quyết Định Đầu Tiên (Tree 0)</h3>
                    <img src="/artifacts/tree_0_lambdamart.png" alt="Tree 0">
                    <p style="font-size: 12.5px; color: #64748b; margin-top: 8px;">
                        Nhánh Đúng/Sai và các ô lá màu xanh (+) cộng điểm, màu đỏ (-) trừ điểm xếp hạng.
                    </p>
                </div>
                <div class="gallery-card">
                    <h3>3. SHAP Beeswarm Plot (Toàn Cục)</h3>
                    <img src="/artifacts/shap_summary_beeswarm.png" alt="SHAP Beeswarm">
                    <p style="font-size: 12.5px; color: #64748b; margin-top: 8px;">
                        Độ phổ biến và nút thắt là 2 lực đẩy đưa môn học lên Top đầu mạnh mẽ nhất.
                    </p>
                </div>
                <div class="gallery-card">
                    <h3>4. So Sánh Hiệu Năng Với Baseline</h3>
                    <img src="/artifacts/baseline_comparison.png" alt="Baseline Comparison">
                    <p style="font-size: 12.5px; color: #64748b; margin-top: 8px;">
                        LambdaMART đạt NDCG@5 = 0.744, vượt trội hơn hẳn Baseline (0.278), đặc biệt trên nhóm P2.
                    </p>
                </div>
            </div>
        </div>

    </div>

    <!-- SHAP Modal -->
    <div id="shapModal" class="modal">
        <div class="modal-content">
            <h3 id="modalCourseTitle" style="font-size: 18px; margin-bottom: 6px;">Mổ xẻ SHAP Waterfall</h3>
            <p id="modalCourseSub" style="color: #64748b; font-size: 13px; margin-bottom: 18px;"></p>
            
            <div style="background: #f8fafc; padding: 12px; border-radius: 8px; margin-bottom: 14px; font-size: 13px;">
                <strong>Điểm cơ sở ban đầu E[f(x)]:</strong> <span id="modalBaseVal"></span> điểm
            </div>

            <div id="shapRows"></div>

            <div style="background: #eff6ff; padding: 12px; border-radius: 8px; margin-top: 14px; font-size: 13px; font-weight: 700; color: #1e40af;">
                = Điểm số xếp hạng cuối cùng: <span id="modalFinalScore"></span>
            </div>

            <button class="close-btn" onclick="closeModal()">Đóng lại</button>
        </div>
    </div>

    <script>
        let currentStudentData = null;

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            event.currentTarget.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        async function init() {
            const res = await fetch('/api/students');
            const students = await res.json();
            const select = document.getElementById('studentSelect');
            select.innerHTML = students.map(s => 
                `<option value="${s.student_id}">SV ${s.student_id} - ${s.persona_label} (GPA: ${s.gpa})</option>`
            ).join('');
            
            if (students.length > 0) {
                loadStudent(students[0].student_id);
            }
        }

        function selectStudent(id) {
            document.getElementById('studentSelect').value = id;
            loadStudent(id);
        }

        async function loadStudent(id) {
            const res = await fetch(`/api/recommend?student_id=${id}`);
            const data = await res.json();
            currentStudentData = data;

            document.getElementById('profId').innerText = data.student_id;
            document.getElementById('profGpa').innerText = data.gpa + " / 100";
            document.getElementById('profCandidates').innerText = data.total_candidates + " môn";

            const pBadge = document.getElementById('profPersona');
            if (data.persona === 0) {
                pBadge.innerHTML = `<span class="badge badge-p1">P1: Bám tiến độ</span>`;
            } else {
                pBadge.innerHTML = `<span class="badge badge-p2">P2: Cần cải thiện điểm</span>`;
            }

            // 1. Render Tab Gợi Ý
            const lmHtml = data.lambdamart_top5.map(c => `
                <div class="course-card">
                    <div class="course-header">
                        <div class="rank-circle ${c.rank === 1 ? 'top1' : ''}">#${c.rank}</div>
                        <div class="course-name">${c.course_name}</div>
                        ${c.actually_registered ? '<span class="badge badge-target">✓ Đã đăng ký</span>' : ''}
                    </div>
                    <div class="metric-tags">
                        <span class="tag tag-bottleneck">🔑 Mở khóa: <strong>${c.bottleneck_score}</strong> môn</span>
                        <span class="tag tag-safe">🛡️ Tỷ lệ trượt: <strong>${c.fail_rate}%</strong> (${c.raw_failed}/${c.raw_total})</span>
                        <span class="tag">🔥 Độ phổ biến: <strong>${(c.popularity * 100).toFixed(0)}%</strong></span>
                        <span class="tag" style="background:#e0f2fe; color:#0369a1;">⭐ Điểm rank: <strong>${c.score}</strong></span>
                    </div>
                    <button class="btn-explain" onclick="showShap(${c.rank - 1})">
                        🔍 Mổ xẻ lý do gợi ý (SHAP Explainability)
                    </button>
                </div>
            `).join('');
            document.getElementById('lambdamartList').innerHTML = lmHtml;

            const baseHtml = data.baseline_top5.map(c => `
                <div class="course-card" style="opacity: 0.9;">
                    <div class="course-header">
                        <div class="rank-circle" style="background: #94a3b8;">#${c.rank}</div>
                        <div class="course-name" style="font-weight: 600;">${c.course_name}</div>
                        ${c.actually_registered ? '<span class="badge badge-target">✓ Đã đăng ký</span>' : ''}
                    </div>
                    <div class="metric-tags">
                        <span class="tag">🔥 Độ phổ biến toàn trường: <strong>${(c.popularity * 100).toFixed(0)}%</strong></span>
                    </div>
                </div>
            `).join('');
            document.getElementById('baselineList').innerHTML = baseHtml;

            // 2. Render Tab Pipeline
            // Table Raw History
            const histBody = document.querySelector('#tableRawHistory tbody');
            histBody.innerHTML = data.history_sample.map(h => `
                <tr>
                    <td>Năm ${h.academic_year} K${h.semester}</td>
                    <td><strong>${h.course_name}</strong></td>
                    <td>${h.score !== null ? h.score : 'Chưa có'}</td>
                    <td>${h.score !== null ? (h.score >= 60 ? '<span style="color:#16a34a; font-weight:700;">✓ Qua môn</span>' : '<span style="color:#dc2626; font-weight:700;">✗ Trượt</span>') : '---'}</td>
                </tr>
            `).join('');

            // Persona Result
            document.getElementById('pipelinePersonaResult').innerHTML = `
                📊 Kết quả tính toán cho Sinh viên ${data.student_id}:<br>
                • Tổng số môn đã tích lũy: <strong>${data.total_past_courses} môn</strong><br>
                • Điểm GPA trung bình tích lũy: <strong>${data.gpa} / 100</strong><br>
                • Số môn từng bị trượt (&lt; 60 điểm): <strong>${data.failed_count} môn</strong><br>
                ➔ Kết luận xếp loại: <strong>${data.persona_label}</strong>
            `;

            // Unlocked Courses
            const topUnlocked = data.lambdamart_top5.filter(c => c.bottleneck_score > 0);
            let unlockHtml = `<strong>Minh họa mở khóa môn học trong đồ thị DAG:</strong><br>`;
            if (topUnlocked.length > 0) {
                unlockHtml += topUnlocked.map(c => 
                    `• Môn <em>"${c.course_name}"</em> có <strong>bottleneck_score = ${c.bottleneck_score}</strong> (Mở khóa các môn: ${c.unlocked_courses.slice(0, 3).join(', ')}...)`
                ).join('<br>');
            } else {
                unlockHtml += `Các môn trong Candidate Set của sinh viên này đều là môn cuối nhánh (bottleneck_score = 0).`;
            }
            document.getElementById('unlockedCoursesInfo').innerHTML = unlockHtml;

            // Feature Matrix Table
            const featBody = document.querySelector('#tableFeatureMatrix tbody');
            featBody.innerHTML = data.feature_matrix.slice(0, 8).map(f => `
                <tr>
                    <td><strong>${f.course_name}</strong></td>
                    <td style="color:#dc2626; font-weight:700;">${f.bottleneck_score}</td>
                    <td>${f.fail_rate}</td>
                    <td>${f.popularity}</td>
                    <td>${f.student_gpa}</td>
                    <td>${f.is_core}</td>
                    <td>${f.persona}</td>
                    <td>${f.target === 1 ? '<span class="badge badge-target">1 (Đã chọn)</span>' : '<span style="color:#94a3b8;">0</span>'}</td>
                </tr>
            `).join('');
        }

        function showShap(index) {
            const course = currentStudentData.lambdamart_top5[index];
            document.getElementById('modalCourseTitle').innerText = `#${course.rank} - ${course.course_name}`;
            document.getElementById('modalCourseSub').innerText = `Sinh viên: ${currentStudentData.student_id} | Điểm GPA: ${currentStudentData.gpa}`;
            document.getElementById('modalBaseVal').innerText = course.base_value;
            document.getElementById('modalFinalScore').innerText = course.score;

            const rowsHtml = course.shap_breakdown.map(b => {
                const isPos = b.shap_val >= 0;
                const widthPercent = Math.min(Math.abs(b.shap_val) * 35, 100);
                return `
                    <div class="shap-bar-row">
                        <div class="shap-name">${b.feature} (${b.feature_val})</div>
                        <div class="shap-bar-track">
                            <div class="shap-fill ${isPos ? 'pos' : 'neg'}" style="width: ${widthPercent}%;"></div>
                        </div>
                        <div class="shap-val" style="color: ${isPos ? '#16a34a' : '#dc2626'}">
                            ${isPos ? '+' : ''}${b.shap_val}
                        </div>
                    </div>
                `;
            }).join('');

            document.getElementById('shapRows').innerHTML = rowsHtml;
            document.getElementById('shapModal').style.display = 'flex';
        }

        function closeModal() {
            document.getElementById('shapModal').style.display = 'none';
        }

        window.onclick = function(event) {
            if (event.target == document.getElementById('shapModal')) {
                closeModal();
            }
        }

        init();
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            
        elif path.startswith("/artifacts/"):
            img_name = os.path.basename(path)
            img_path = os.path.join("visual_artifacts", img_name)
            if os.path.exists(img_path):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.end_headers()
                with open(img_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                
        elif path == "/api/students":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(student_list).encode('utf-8'))
            
        elif path == "/api/recommend":
            s_id = int(query.get('student_id', [student_list[0]['student_id']])[0])
            result = get_recommendation_and_pipeline(s_id)
            if result:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=5000):
    server = HTTPServer(('127.0.0.1', port), RequestHandler)
    url = f"http://localhost:{port}"
    print(f"\n" + "=" * 75)
    print(f"  [HOÀN TẤT] MÁY CHỦ WEB DEMO ĐANG CHẠY TẠI: {url}")
    print(f"  Bấm Ctrl + C trong terminal để dừng server.")
    print("=" * 75 + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng máy chủ thành công.")

if __name__ == '__main__':
    run_server(5000)
