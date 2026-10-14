# -*- coding: utf-8 -*-
"""
HỆ THỐNG GỢI Ý HỌC PHẦN THÔNG MINH (LAMBDAMART RANKING & 6-PERSONA ADAPTIVE SYSTEM)
WEB DEMO TRỰC QUAN: 6 CHÂN DUNG SINH VIÊN (P1 -> P6), TẬP TRUNG SỐ LIỆU ĐỐI SÁNH & SHAP

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
print("  KHỞI TẠO HỆ THỐNG GỢI Ý HỌC PHẦN THÔNG MINH (6-PERSONA ADAPTIVE SYSTEM)")
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

# ĐỊNH NGHĨA 6 PERSONA HỌC VỤ
PERSONAS = {
    1: {
        'id': 1,
        'code': 'P1',
        'name': 'Tốt nghiệp đúng hạn',
        'criteria': 'Tích lũy tín chỉ bằng chuẩn; ít môn nợ',
        'strategy': 'Ưu tiên môn nút thắt & bắt buộc còn thiếu',
        'color': '#2563eb'
    },
    2: {
        'id': 2,
        'code': 'P2',
        'name': 'Cải thiện kết quả',
        'criteria': 'GPA dưới ngưỡng; có môn chưa đạt (< 60 điểm)',
        'strategy': 'Tránh môn tỷ lệ rớt cao; ưu tiên môn an toàn để gỡ điểm GPA',
        'color': '#d97706'
    },
    3: {
        'id': 3,
        'code': 'P3',
        'name': 'Khám phá đa dạng',
        'criteria': 'Tiến độ ổn; còn nhiều tín chỉ tự chọn ngoài ngành',
        'strategy': 'Mở rộng tự chọn đa dạng; khai thác mô tả học phần',
        'color': '#8b5cf6'
    },
    4: {
        'id': 4,
        'code': 'P4',
        'name': 'Chuyên sâu ngành',
        'criteria': 'Tự chọn tập trung sâu vào một nhóm chuyên ngành',
        'strategy': 'Ưu tiên tối đa môn chuyên ngành cốt lõi và nâng cao',
        'color': '#0d9488'
    },
    5: {
        'id': 5,
        'code': 'P5',
        'name': 'Học vượt tiến độ',
        'criteria': 'GPA xuất sắc (≥ 84); tích lũy nhanh hơn chuẩn',
        'strategy': 'Nới giới hạn tín chỉ; mở đường học sớm môn năm trên',
        'color': '#16a34a'
    },
    6: {
        'id': 6,
        'code': 'P6',
        'name': 'Năm cuối nợ môn',
        'criteria': 'Kỳ học cao (≥ 3 năm); còn môn bắt buộc chưa hoàn thành',
        'strategy': 'Báo động đỏ: Ưu tiên tuyệt đối môn bắt buộc còn thiếu để ra trường',
        'color': '#dc2626'
    }
}

def detect_student_persona(stud_history):
    if len(stud_history) == 0:
        return 1
    valid_scores = stud_history['Score'].dropna()
    gpa = valid_scores.mean() if len(valid_scores) > 0 else 75.0
    failed_count = (valid_scores < 60.0).sum()
    total_credits = len(stud_history)
    unique_years = len(stud_history['AcademicYear'].unique())
    
    # 1. P6: Năm cuối nợ môn (Học từ 3 năm trở lên mà vẫn có môn nợ)
    if unique_years >= 3 and failed_count > 0:
        return 6
    # 2. P5: Học vượt (GPA >= 84, không nợ môn, tích lũy nhiều môn)
    if gpa >= 84.0 and failed_count == 0 and total_credits >= 10:
        return 5
    # 3. P2: Cải thiện kết quả (Có môn rớt hoặc GPA < 72.0)
    if failed_count > 0 or gpa < 72.0:
        return 2
    # 4. P3: Khám phá (Nhiều môn tự chọn ngoài chuyên ngành)
    core_count = sum(1 for _, r in stud_history.iterrows() if any(k in str(r.get('CourseName', '')).lower() for k in ['computer', 'algorithm', 'data', 'system', 'programming']))
    non_core_ratio = 1.0 - (core_count / total_credits) if total_credits > 0 else 0
    if non_core_ratio > 0.4:
        return 3
    # 5. P4: Chuyên sâu (Tập trung rất sâu vào Core CS, GPA khá giỏi)
    if core_count >= 5 and gpa >= 78.0:
        return 4
    # 6. P1: Tốt nghiệp đúng hạn (Chuẩn)
    return 1

# Phân chia dữ liệu Train (<= 19-20) và Test (20-21 kỳ 1)
train_selections = cs_selections[cs_selections['AcademicYear'] <= '19-20'].copy()
test_selections = cs_selections[(cs_selections['AcademicYear'] == '20-21') & (cs_selections['Semester'] == 1.0)].copy()

popular_cs_courses = cs_selections['CourseId'].value_counts().head(50).index.tolist()

def get_student_stats(student_id, history_df):
    stud_history = history_df[history_df['StudentId'] == student_id]
    if len(stud_history) == 0:
        return 1, 75.0, 0
    valid_scores = stud_history['Score'].dropna()
    gpa = valid_scores.mean() if len(valid_scores) > 0 else 75.0
    failed_count = (valid_scores < 60.0).sum()
    persona_id = detect_student_persona(stud_history)
    return persona_id, round(gpa, 2), failed_count

def build_candidate_dataset(eval_selections, history_selections):
    records = []
    student_ids = eval_selections['StudentId'].unique()
    for s_id in student_ids:
        persona_id, gpa, failed_count = get_student_stats(s_id, history_selections)
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
                'persona': persona_id,
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
    p_id = int(sub['persona'].iloc[0])
    p_info = PERSONAS.get(p_id, PERSONAS[1])
    gpa = float(sub['student_gpa'].iloc[0])
    student_list.append({
        'student_id': int(s_id),
        'gpa': gpa,
        'persona_id': p_id,
        'persona_code': p_info['code'],
        'persona_name': p_info['name'],
        'num_candidates': len(sub)
    })

def compute_persona_adjustment(row, persona_id):
    b_score = row['bottleneck_score']
    f_rate = row['fail_rate']
    pop = row['popularity']
    is_core = row['is_core_major']
    
    if persona_id == 1:
        # P1: Đúng hạn -> Ưu tiên nút thắt mở đường
        bonus = 0.55 * (b_score / 3.0) + 0.15 * is_core
    elif persona_id == 2:
        # P2: Cải thiện kết quả -> Phạt nặng môn khó, cộng điểm môn an toàn
        bonus = -1.6 * f_rate + 0.45 * (1.0 - f_rate) - 0.25 * b_score
    elif persona_id == 3:
        # P3: Khám phá đa dạng -> Mở rộng tự chọn ngoài chuyên ngành
        bonus = 0.9 * (1.0 - is_core) + 0.35 * pop - 0.2 * b_score
    elif persona_id == 4:
        # P4: Chuyên sâu ngành -> Ưu tiên tối đa môn cốt lõi
        bonus = 0.95 * is_core + 0.3 * (1.0 - f_rate) + 0.15 * pop
    elif persona_id == 5:
        # P5: Học vượt -> Mở đường xa, chấp nhận môn khó
        bonus = 0.75 * (b_score / 2.0) + 0.5 * is_core + 0.25 * f_rate
    elif persona_id == 6:
        # P6: Năm cuối nợ môn -> Ưu tiên tuyệt đối môn tiên quyết/bắt buộc
        bonus = 2.2 * (1.0 if b_score > 0 else 0.0) + 0.9 * is_core - 0.6 * (1.0 - is_core)
    else:
        bonus = 0.0
        
    return round(bonus, 4)

def get_recommendation_and_pipeline(s_id, active_persona_id=None):
    sub = df_test[df_test['StudentId'] == s_id].copy()
    if len(sub) == 0:
        return None
    
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
    
    detected_persona_id = int(sub['persona'].iloc[0])
    selected_persona_id = int(active_persona_id) if active_persona_id is not None else detected_persona_id
    persona_meta = PERSONAS.get(selected_persona_id, PERSONAS[1])

    # Tính điểm Persona Adjusted Score
    sub['persona_bonus'] = sub.apply(lambda r: compute_persona_adjustment(r, selected_persona_id), axis=1)
    sub['final_rank_score'] = sub['predicted_score'] + sub['persona_bonus']

    sub_lm = sub.sort_values(by='final_rank_score', ascending=False)
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
        
        # Thêm lực kéo của Persona Policy vào SHAP
        shap_breakdown.append({
            'feature': f'persona_policy ({persona_meta["code"]})',
            'feature_val': selected_persona_id,
            'shap_val': round(float(row['persona_bonus']), 4)
        })
            
        unlocked = descendant_map.get(int(row['CourseId']), [])
        
        item_data = {
            'rank': rank,
            'course_id': int(row['CourseId']),
            'course_name': row['CourseName'],
            'score': round(float(row['final_rank_score']), 4),
            'raw_lm_score': round(float(row['predicted_score']), 4),
            'persona_bonus': round(float(row['persona_bonus']), 4),
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
            'persona': selected_persona_id,
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
        'detected_persona_id': detected_persona_id,
        'active_persona_id': selected_persona_id,
        'persona_meta': persona_meta,
        'all_personas': PERSONAS,
        'total_past_courses': total_credits,
        'failed_count': failed_count,
        'history_sample': history_sample,
        'feature_matrix': feature_matrix_sample,
        'lambdamart_top5': lm_results[:5],
        'baseline_top5': base_results[:5],
        'total_candidates': len(sub)
    }

# GIAO DIỆN HTML/CSS/JS HUMANIZED, DATA-CENTRIC & 6-PERSONA INTERACTIVE
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Gợi Ý Môn Học Cá Nhân Hóa - 6 Personas & LambdaMART</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #eff6ff;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --slate-50: #f8fafc;
            --slate-100: #f1f5f9;
            --slate-200: #e2e8f0;
            --slate-300: #cbd5e1;
            --slate-600: #475569;
            --slate-700: #334155;
            --slate-800: #1e293b;
            --slate-900: #0f172a;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.04);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background: #f1f5f9; color: var(--slate-900); padding-bottom: 50px; }

        /* Topbar */
        .topbar {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white; padding: 18px 36px; display: flex; align-items: center; justify-content: space-between;
            box-shadow: var(--shadow-md);
        }
        .topbar-left { display: flex; align-items: center; gap: 14px; }
        .logo-icon {
            width: 42px; height: 42px; border-radius: 10px; background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 800;
        }
        .topbar h1 { font-size: 19px; font-weight: 800; letter-spacing: -0.3px; }
        .topbar p { font-size: 12.5px; color: #94a3b8; margin-top: 2px; }
        .tag-badge {
            background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4);
            color: #93c5fd; padding: 5px 12px; border-radius: 99px; font-size: 12px; font-weight: 700;
        }

        .main-container { max-width: 1320px; margin: 20px auto; padding: 0 20px; }

        /* Executive Metrics */
        .metrics-hero {
            display: grid; grid-template-columns: 1fr 1fr 0.9fr; gap: 16px; margin-bottom: 20px;
        }
        .metric-card {
            background: white; border-radius: 14px; padding: 18px 22px; box-shadow: var(--shadow-sm);
            border: 1px solid var(--slate-200); position: relative; overflow: hidden;
        }
        .metric-card.accent-lm { border-left: 5px solid var(--primary); }
        .metric-card.accent-base { border-left: 5px solid var(--slate-400); }
        .metric-card.accent-lift {
            background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); color: white; border: none;
        }
        .card-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--slate-600); }
        .metric-card.accent-lift .card-label { color: #a7f3d0; }
        .score-val { font-size: 34px; font-weight: 800; margin: 4px 0; font-family: 'JetBrains Mono', monospace; }
        .score-sub { font-size: 12.5px; color: var(--slate-600); }
        .metric-card.accent-lift .score-sub { color: #d1fae5; }
        
        .progress-track {
            height: 7px; background: var(--slate-100); border-radius: 99px; margin-top: 10px; overflow: hidden;
        }
        .progress-fill { height: 100%; border-radius: 99px; }

        /* Controls */
        .action-bar {
            background: white; border-radius: 14px; padding: 16px 22px; box-shadow: var(--shadow-sm);
            border: 1px solid var(--slate-200); margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; gap: 16px;
        }
        .select-group { display: flex; align-items: center; gap: 10px; flex-grow: 1; max-width: 520px; }
        .select-group label { font-size: 13.5px; font-weight: 700; color: var(--slate-700); white-space: nowrap; }
        select {
            width: 100%; padding: 10px 14px; border-radius: 10px; border: 1.5px solid var(--slate-300);
            background: white; font-size: 13.5px; font-weight: 600; outline: none; cursor: pointer;
            transition: all 0.15s;
        }
        select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,0.12); }
        .quick-group { display: flex; align-items: center; gap: 8px; }
        .btn-quick {
            padding: 8px 14px; border-radius: 8px; border: 1px solid var(--slate-300); background: var(--slate-50);
            font-size: 12.5px; font-weight: 700; color: var(--slate-700); cursor: pointer; transition: all 0.15s;
        }
        .btn-quick:hover { background: var(--primary-light); color: var(--primary); border-color: var(--primary); }

        /* ================= 6-PERSONA INTERACTIVE STRIP ================= */
        .persona-panel {
            background: white; border-radius: 14px; padding: 18px 22px; box-shadow: var(--shadow-sm);
            border: 1px solid var(--slate-200); margin-bottom: 20px;
        }
        .persona-panel-head {
            display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;
        }
        .persona-title { font-size: 14.5px; font-weight: 800; color: var(--slate-900); display: flex; align-items: center; gap: 8px; }
        .auto-tag {
            background: #dbeafe; color: #1e40af; padding: 4px 10px; border-radius: 99px;
            font-size: 11.5px; font-weight: 700;
        }
        .persona-cards-grid {
            display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;
        }
        .persona-btn {
            background: var(--slate-50); border: 2px solid var(--slate-200); border-radius: 10px;
            padding: 10px 8px; text-align: center; cursor: pointer; transition: all 0.18s; position: relative;
        }
        .persona-btn:hover { border-color: var(--primary); background: white; transform: translateY(-1px); }
        .persona-btn.active {
            border-color: var(--primary); background: #eff6ff; box-shadow: 0 0 0 3px rgba(37,99,235,0.15);
        }
        .p-code { font-size: 12px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: var(--slate-600); }
        .persona-btn.active .p-code { color: var(--primary); }
        .p-name { font-size: 12.5px; font-weight: 700; color: var(--slate-800); margin-top: 3px; }
        .p-detected-dot {
            width: 8px; height: 8px; border-radius: 50%; background: #10b981; position: absolute;
            top: 6px; right: 6px;
        }

        .active-strategy-banner {
            margin-top: 14px; background: #f8fafc; border-left: 4px solid var(--primary);
            border-radius: 0 8px 8px 0; padding: 10px 14px; display: flex; align-items: center; justify-content: space-between;
            font-size: 13px; color: var(--slate-700);
        }

        /* Profile Mini Grid */
        .profile-strip {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;
        }
        .strip-item {
            background: white; border-radius: 12px; padding: 14px 18px; border: 1px solid var(--slate-200);
            box-shadow: var(--shadow-sm);
        }
        .strip-item .label { font-size: 11.5px; font-weight: 700; color: var(--slate-600); text-transform: uppercase; }
        .strip-item .value { font-size: 19px; font-weight: 800; color: var(--slate-900); margin-top: 2px; }

        /* Tabs */
        .tabs-nav {
            display: flex; gap: 8px; margin-bottom: 18px; border-bottom: 2px solid var(--slate-200); padding-bottom: 2px;
        }
        .tab-item {
            padding: 10px 18px; border: none; background: transparent; font-size: 13.5px; font-weight: 700;
            color: var(--slate-600); cursor: pointer; border-radius: 8px 8px 0 0; display: flex; align-items: center;
            gap: 7px; transition: all 0.15s;
        }
        .tab-item:hover { color: var(--primary); background: var(--slate-200); }
        .tab-item.active { color: var(--primary); border-bottom: 3px solid var(--primary); background: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.2s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }

        /* Course Cards */
        .grid-columns { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 20px; }
        .col-box {
            background: white; border-radius: 16px; padding: 20px; border: 1px solid var(--slate-200);
            box-shadow: var(--shadow-sm);
        }
        .col-header {
            display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;
            padding-bottom: 12px; border-bottom: 2px solid var(--slate-100);
        }
        .col-header h3 { font-size: 16px; font-weight: 800; display: flex; align-items: center; gap: 8px; }
        .badge-pill {
            padding: 4px 10px; border-radius: 99px; font-size: 12px; font-weight: 700; font-family: 'JetBrains Mono', monospace;
        }

        .course-card {
            background: white; border: 1.5px solid var(--slate-200); border-radius: 12px;
            padding: 14px 16px; margin-bottom: 12px; transition: all 0.2s; position: relative;
        }
        .course-card:hover {
            border-color: var(--primary); box-shadow: var(--shadow-md); transform: translateY(-1px);
        }
        .card-top { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .rank-badge {
            width: 26px; height: 26px; border-radius: 8px; background: var(--slate-800); color: white;
            font-size: 12px; font-weight: 800; display: flex; align-items: center; justify-content: center;
            font-family: 'JetBrains Mono', monospace; flex-shrink: 0;
        }
        .rank-badge.top1 { background: #eab308; color: #713f12; }
        .course-title { font-size: 14.5px; font-weight: 700; color: var(--slate-900); flex-grow: 1; }
        .tag-registered {
            background: #dcfce7; color: #15803d; border: 1px solid #86efac; padding: 3px 8px;
            border-radius: 6px; font-size: 11.5px; font-weight: 700; white-space: nowrap;
        }

        .chips-grid {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; background: var(--slate-50);
            padding: 8px; border-radius: 8px; border: 1px solid var(--slate-100);
        }
        .chip { text-align: center; }
        .chip-label { font-size: 10.5px; color: var(--slate-600); font-weight: 600; }
        .chip-val { font-size: 13px; font-weight: 800; color: var(--slate-800); font-family: 'JetBrains Mono', monospace; margin-top: 1px; }

        .btn-shap {
            margin-top: 10px; width: 100%; padding: 7px 12px; background: var(--primary-light);
            border: 1px solid #bfdbfe; color: var(--primary-dark); border-radius: 8px; font-size: 12px;
            font-weight: 700; cursor: pointer; transition: all 0.15s; display: flex; align-items: center;
            justify-content: center; gap: 6px;
        }
        .btn-shap:hover { background: var(--primary); color: white; border-color: var(--primary); }

        /* Pipeline Tables */
        .stepper-container { display: flex; flex-direction: column; gap: 16px; }
        .stepper-card {
            background: white; border-radius: 14px; padding: 18px 22px; border: 1px solid var(--slate-200);
            box-shadow: var(--shadow-sm);
        }
        .stepper-head { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
        .step-circle {
            width: 28px; height: 28px; border-radius: 8px; background: var(--primary); color: white;
            font-weight: 800; font-size: 13px; display: flex; align-items: center; justify-content: center;
        }
        .step-title { font-size: 15px; font-weight: 700; color: var(--slate-900); }
        .step-meta { font-size: 12px; color: var(--slate-600); margin-left: auto; }

        table.compact-table {
            width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px;
        }
        table.compact-table th, table.compact-table td {
            padding: 8px 10px; border: 1px solid var(--slate-200); text-align: left;
        }
        table.compact-table th { background: var(--slate-50); font-weight: 700; color: var(--slate-600); }
        table.compact-table tr:hover { background: #f8fafc; }

        /* Gallery */
        .gallery-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
        .gallery-item {
            background: white; border-radius: 14px; padding: 16px; border: 1px solid var(--slate-200);
            box-shadow: var(--shadow-sm);
        }
        .gallery-item h4 { font-size: 14.5px; font-weight: 700; margin-bottom: 8px; color: var(--slate-800); }
        .gallery-item img { width: 100%; border-radius: 8px; border: 1px solid var(--slate-200); display: block; }
        .gallery-item p { font-size: 12px; color: var(--slate-600); margin-top: 6px; }

        /* SHAP Modal */
        .modal-overlay {
            display: none; position: fixed; inset: 0; background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(2px);
            z-index: 1000; align-items: center; justify-content: center; padding: 20px;
        }
        .modal-box {
            background: white; border-radius: 16px; max-width: 620px; width: 100%; padding: 24px;
            box-shadow: var(--shadow-lg); animation: popUp 0.18s ease-out;
        }
        @keyframes popUp { from { transform: scale(0.96); opacity: 0; } to { transform: scale(1); opacity: 1; } }
        .modal-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
        .modal-title { font-size: 16.5px; font-weight: 800; color: var(--slate-900); }
        .modal-sub { font-size: 12.5px; color: var(--slate-600); margin-top: 2px; }
        .close-x {
            background: none; border: none; font-size: 20px; color: var(--slate-400); cursor: pointer; line-height: 1;
        }
        .close-x:hover { color: var(--slate-700); }
        
        .score-formula-box {
            background: var(--slate-50); border: 1px solid var(--slate-200); border-radius: 10px;
            padding: 10px 14px; font-size: 13px; font-family: 'JetBrains Mono', monospace; margin-bottom: 16px;
            display: flex; justify-content: space-between;
        }
        .shap-row { display: flex; align-items: center; margin-bottom: 8px; font-size: 12.5px; }
        .shap-feat { width: 180px; font-weight: 600; color: var(--slate-700); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .shap-bar-bg { flex-grow: 1; height: 16px; background: var(--slate-100); border-radius: 4px; margin: 0 10px; overflow: hidden; }
        .shap-bar-val { height: 100%; border-radius: 4px; }
        .shap-bar-val.pos { background: #10b981; }
        .shap-bar-val.neg { background: #ef4444; }
        .shap-num { width: 65px; text-align: right; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        .btn-close-modal {
            margin-top: 16px; width: 100%; padding: 10px; background: var(--slate-800); color: white;
            border: none; border-radius: 8px; font-weight: 700; font-size: 13px; cursor: pointer;
        }
        .btn-close-modal:hover { background: var(--slate-900); }

        @media (max-width: 992px) {
            .persona-cards-grid { grid-template-columns: repeat(3, 1fr); }
            .metrics-hero { grid-template-columns: 1fr; }
            .grid-columns { grid-template-columns: 1fr; }
            .profile-strip { grid-template-columns: 1fr 1fr; }
            .gallery-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <!-- TOPBAR -->
    <header class="topbar">
        <div class="topbar-left">
            <div class="logo-icon">AI</div>
            <div>
                <h1>HỆ THỐNG GỢI Ý MÔN HỌC THÍCH ỨNG THEO 6 PERSONAS</h1>
                <p>Mô hình Học máy LambdaMART kết hợp Chiến lược Chân dung Sinh viên (P1 -> P6)</p>
            </div>
        </div>
        <span class="tag-badge">Khóa Luận / Đồ Án Tốt Nghiệp</span>
    </header>

    <main class="main-container">
        
        <!-- 1. SCOREBOARD -->
        <section class="metrics-hero">
            <div class="metric-card accent-lm">
                <div class="card-label">Mô hình Đề xuất (LambdaMART)</div>
                <div class="score-val" style="color: var(--primary);">0.744</div>
                <div class="score-sub">Chỉ số NDCG@5 (Độ chính xác Top 5)</div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: 74.4%; background: linear-gradient(90deg, #3b82f6, #1d4ed8);"></div>
                </div>
            </div>

            <div class="metric-card accent-base">
                <div class="card-label">Mô hình Cơ sở (Popularity Baseline)</div>
                <div class="score-val" style="color: #64748b;">0.278</div>
                <div class="score-sub">Chỉ số NDCG@5 (Theo số đông)</div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: 27.8%; background: #94a3b8;"></div>
                </div>
            </div>

            <div class="metric-card accent-lift">
                <div class="card-label">Mức Độ Cải Thiện</div>
                <div class="score-val" style="color: #a7f3d0;">+167%</div>
                <div class="score-sub">AI vượt trội gấp 2.68 lần Baseline</div>
                <div style="margin-top: 10px; font-size: 11.5px; opacity: 0.9;">
                    Đánh giá trên toàn bộ sinh viên tập Test độc lập
                </div>
            </div>
        </section>

        <!-- 2. CONTROLS: CHỌN SINH VIÊN -->
        <section class="action-bar">
            <div class="select-group">
                <label for="studentSelect">Chọn sinh viên kiểm thử:</label>
                <select id="studentSelect" onchange="onStudentChange(this.value)"></select>
            </div>
            <div class="quick-group">
                <span style="font-size: 12.5px; color: var(--slate-600); font-weight: 700;">Xem nhanh:</span>
                <button class="btn-quick" onclick="selectStudent(2825)">SV 2825 (P4 · Chuyên sâu)</button>
                <button class="btn-quick" onclick="selectStudent(133)">SV 133 (P2 · Cần gỡ điểm)</button>
            </div>
        </section>

        <!-- 3. INTERACTIVE 6-PERSONA SELECTOR STRIP -->
        <section class="persona-panel">
            <div class="persona-panel-head">
                <div class="persona-title">
                    <span>🧭 Chân Dung Sinh Viên & Chiến Lược Cá Nhân Hóa (6 Personas)</span>
                    <span class="auto-tag" id="detectedPersonaTag">🤖 Nhận diện tự động: P1</span>
                </div>
                <span style="font-size: 12px; color: var(--slate-600); font-weight: 600;">
                    Click để đổi Persona và xem AI thay đổi thứ hạng tức thì:
                </span>
            </div>

            <div class="persona-cards-grid" id="personaCardsGrid">
                <!-- Sẽ được render bởi JS -->
            </div>

            <div class="active-strategy-banner">
                <div>
                    <strong>Chiến lược đang áp dụng:</strong> <span id="strategyText">---</span>
                </div>
                <div style="font-size: 11.5px; color: var(--slate-600);">
                    Điều kiện nhận diện: <span id="criteriaText" style="font-style: italic;">---</span>
                </div>
            </div>
        </section>

        <!-- 4. PROFILE STRIP -->
        <section class="profile-strip">
            <div class="strip-item">
                <div class="label">Mã Sinh Viên</div>
                <div class="value" id="profId">---</div>
            </div>
            <div class="strip-item">
                <div class="label">Điểm Tích Lũy (GPA)</div>
                <div class="value" id="profGpa">---</div>
            </div>
            <div class="strip-item">
                <div class="label">Số Môn Từng Trượt</div>
                <div class="value" id="profFailed" style="color: #dc2626;">---</div>
            </div>
            <div class="strip-item">
                <div class="label">Môn Khả Dụng Kỳ Tới</div>
                <div class="value" id="profCandidates">---</div>
            </div>
        </section>

        <!-- 5. TABS -->
        <nav class="tabs-nav">
            <button class="tab-item active" onclick="switchTab('tab-recommend')">
                🎯 Bảng So Sánh Gợi Ý
            </button>
            <button class="tab-item" onclick="switchTab('tab-pipeline')">
                🔬 Luồng Số Liệu Từng Bước (Pipeline Inspector)
            </button>
            <button class="tab-item" onclick="switchTab('tab-gallery')">
                📈 Biểu Đồ Minh Họa
            </button>
        </nav>

        <!-- TAB 1: RECOMMENDATIONS -->
        <div id="tab-recommend" class="tab-content active">
            <div class="grid-columns">
                <!-- Cột LambdaMART -->
                <div class="col-box" style="border-top: 4px solid var(--primary);">
                    <div class="col-header">
                        <h3>🎯 Đề Xuất Bởi LambdaMART (<span id="activePersonaHeader">P1</span>)</h3>
                        <span class="badge-pill" style="background: #dbeafe; color: #1e40af;">NDCG@5: 0.744</span>
                    </div>
                    <div id="lambdamartList">Đang tải danh sách...</div>
                </div>

                <!-- Cột Baseline -->
                <div class="col-box" style="border-top: 4px solid #94a3b8; background: #fafafa;">
                    <div class="col-header">
                        <h3 style="color: #475569;">📊 Popularity Baseline (Số Đông)</h3>
                        <span class="badge-pill" style="background: #e2e8f0; color: #475569;">NDCG@5: 0.278</span>
                    </div>
                    <div id="baselineList">Đang tải danh sách...</div>
                </div>
            </div>
        </div>

        <!-- TAB 2: PIPELINE INSPECTOR -->
        <div id="tab-pipeline" class="tab-content">
            <div class="stepper-container">
                
                <div class="stepper-card">
                    <div class="stepper-head">
                        <div class="step-circle">1</div>
                        <div class="step-title">Lịch Sử Học Tập Thực Tế (Selection Table)</div>
                        <div class="step-meta" id="historySummary">Đang tính...</div>
                    </div>
                    <div style="overflow-x: auto;">
                        <table class="compact-table" id="tableRawHistory">
                            <thead>
                                <tr>
                                    <th>Học kỳ</th>
                                    <th>Môn đã từng học</th>
                                    <th>Điểm số (/100)</th>
                                    <th>Trạng thái</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>

                <div class="stepper-card">
                    <div class="stepper-head">
                        <div class="step-circle">2</div>
                        <div class="step-title">Vector Đặc Trưng Số Nạp Vào AI (Feature Vector)</div>
                        <div class="step-meta">Mỗi môn ứng viên được số hóa thành 6 chiều</div>
                    </div>
                    <div style="overflow-x: auto;">
                        <table class="compact-table" id="tableFeatureMatrix">
                            <thead>
                                <tr>
                                    <th>Môn Ứng Viên</th>
                                    <th>Nút Thắt (Bottleneck)</th>
                                    <th>Tỷ Lệ Trượt (Fail Rate)</th>
                                    <th>Độ Phổ Biến</th>
                                    <th>GPA SV</th>
                                    <th>Cùng Ngành</th>
                                    <th>Persona</th>
                                    <th>Thực Tế Chọn (Ground Truth)</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>

                <div class="stepper-card">
                    <div class="stepper-head">
                        <div class="step-circle">3</div>
                        <div class="step-title">Cơ Chế Điều Chỉnh Thứ Hạng Persona-Aware Re-ranking</div>
                    </div>
                    <div style="background: var(--slate-50); padding: 12px 16px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--slate-800); border: 1px solid var(--slate-200);">
                        Score_Cuối_Cùng(c) = Score_LambdaMART(c) + Bonus_Persona(c, Persona_ID)
                    </div>
                    <p style="font-size: 12.5px; color: var(--slate-600); margin-top: 8px;">
                        • Khi bạn bấm đổi Persona ở trên, <code>Bonus_Persona</code> sẽ tự động tính toán lại theo chiến lược của Persona đó.<br>
                        • Môn nào có tổng điểm <code>Score_Cuối_Cùng</code> lớn nhất sẽ đứng ở vị trí <strong>Top 1</strong>!
                    </p>
                </div>

            </div>
        </div>

        <!-- TAB 3: GALLERY -->
        <div id="tab-gallery" class="tab-content">
            <div class="gallery-grid">
                <div class="gallery-item">
                    <h4>1. Đồ Thị Tiên Quyết DAG & Nút Thắt (Bottleneck)</h4>
                    <img src="/artifacts/dag_prerequisite_tree.png" alt="DAG Tree">
                    <p>Các nốt màu đỏ mở khóa nhiều môn phía sau trong cây lộ trình.</p>
                </div>
                <div class="gallery-item">
                    <h4>2. Cây Quyết Định Tăng Cường (Tree 0)</h4>
                    <img src="/artifacts/tree_0_lambdamart.png" alt="Tree 0">
                    <p>Các lá màu xanh (+) tăng điểm, màu đỏ (-) giảm điểm xếp hạng.</p>
                </div>
                <div class="gallery-item">
                    <h4>3. Mức Độ Đóng Góp Của Các Đặc Trưng (SHAP Beeswarm)</h4>
                    <img src="/artifacts/shap_summary_beeswarm.png" alt="SHAP Summary">
                    <p>Độ phổ biến và nút thắt là 2 lực đẩy đưa môn lên Top 1 mạnh nhất.</p>
                </div>
                <div class="gallery-item">
                    <h4>4. Đối Sánh Hiệu Năng: LambdaMART vs Baseline</h4>
                    <img src="/artifacts/baseline_comparison.png" alt="Comparison">
                    <p>LambdaMART đạt NDCG@5 = 0.744, gấp 2.68 lần Baseline (0.278).</p>
                </div>
            </div>
        </div>

    </main>

    <!-- SHAP MODAL -->
    <div id="shapModal" class="modal-overlay" onclick="closeModalOnOverlay(event)">
        <div class="modal-box">
            <div class="modal-header">
                <div>
                    <div class="modal-title" id="modalCourseTitle">Phân Tích Đóng Góp SHAP & Persona</div>
                    <div class="modal-sub" id="modalCourseSub"></div>
                </div>
                <button class="close-x" onclick="closeModal()">&times;</button>
            </div>

            <div class="score-formula-box">
                <span>Điểm gốc Base Value: <strong id="modalBaseVal">0.0</strong></span>
                <span>Điểm tổng Rank Score: <strong id="modalFinalScore" style="color: var(--primary);">0.0</strong></span>
            </div>

            <div id="shapRows"></div>

            <button class="btn-close-modal" onclick="closeModal()">Đóng cửa sổ</button>
        </div>
    </div>

    <script>
        let currentStudentData = null;
        let selectedPersonaOverride = null;

        const PERSONA_LIST = [
            { id: 1, code: 'P1', name: 'Đúng hạn', full: 'P1 · Tốt nghiệp đúng hạn' },
            { id: 2, code: 'P2', name: 'Cải thiện', full: 'P2 · Cải thiện kết quả' },
            { id: 3, code: 'P3', name: 'Khám phá', full: 'P3 · Khám phá đa dạng' },
            { id: 4, code: 'P4', name: 'Chuyên sâu', full: 'P4 · Chuyên sâu ngành' },
            { id: 5, code: 'P5', name: 'Học vượt', full: 'P5 · Học vượt tiến độ' },
            { id: 6, code: 'P6', name: 'Nợ môn', full: 'P6 · Năm cuối nợ môn' }
        ];

        function switchTab(tabId) {
            document.querySelectorAll('.tab-item').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(p => p.classList.remove('active'));
            event.currentTarget.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        async function init() {
            const res = await fetch('/api/students');
            const students = await res.json();
            const select = document.getElementById('studentSelect');
            select.innerHTML = students.map(s => 
                `<option value="${s.student_id}">SV ${s.student_id} - ${s.persona_code} · ${s.persona_name} (GPA: ${s.gpa})</option>`
            ).join('');
            
            if (students.length > 0) {
                loadStudent(students[0].student_id);
            }
        }

        function onStudentChange(id) {
            selectedPersonaOverride = null; // Reset override khi chọn SV mới
            loadStudent(id);
        }

        function selectStudent(id) {
            document.getElementById('studentSelect').value = id;
            selectedPersonaOverride = null;
            loadStudent(id);
        }

        function changePersona(pId) {
            selectedPersonaOverride = pId;
            const currentStudentId = document.getElementById('studentSelect').value;
            loadStudent(currentStudentId, pId);
        }

        async function loadStudent(id, personaOverride = null) {
            let url = `/api/recommend?student_id=${id}`;
            if (personaOverride !== null) {
                url += `&persona_id=${personaOverride}`;
            } else if (selectedPersonaOverride !== null) {
                url += `&persona_id=${selectedPersonaOverride}`;
            }

            const res = await fetch(url);
            const data = await res.json();
            currentStudentData = data;

            document.getElementById('profId').innerText = data.student_id;
            document.getElementById('profGpa').innerText = data.gpa + " / 100";
            document.getElementById('profFailed').innerText = data.failed_count + " môn";
            document.getElementById('profCandidates').innerText = data.total_candidates + " môn";

            // Render Persona Buttons
            const detectedId = data.detected_persona_id;
            const activeId = data.active_persona_id;
            document.getElementById('detectedPersonaTag').innerText = `🤖 Nhận diện tự động: P${detectedId}`;
            document.getElementById('strategyText').innerText = data.persona_meta.strategy;
            document.getElementById('criteriaText').innerText = data.persona_meta.criteria;
            document.getElementById('activePersonaHeader').innerText = `${data.persona_meta.code} · ${data.persona_meta.name}`;

            const cardsHtml = PERSONA_LIST.map(p => `
                <div class="persona-btn ${p.id === activeId ? 'active' : ''}" onclick="changePersona(${p.id})">
                    ${p.id === detectedId ? '<div class="p-detected-dot" title="Chân dung nhận diện tự động từ dữ liệu"></div>' : ''}
                    <div class="p-code">${p.code}</div>
                    <div class="p-name">${p.name}</div>
                </div>
            `).join('');
            document.getElementById('personaCardsGrid').innerHTML = cardsHtml;

            // 1. Render Top 5 LambdaMART
            const lmHtml = data.lambdamart_top5.map(c => `
                <div class="course-card">
                    <div class="card-top">
                        <div class="rank-badge ${c.rank === 1 ? 'top1' : ''}">#${c.rank}</div>
                        <div class="course-title">${c.course_name}</div>
                        ${c.actually_registered ? '<span class="tag-registered">✓ Đã đăng ký</span>' : ''}
                    </div>
                    <div class="chips-grid">
                        <div class="chip">
                            <div class="chip-label">Mở khóa</div>
                            <div class="chip-val" style="color:#dc2626;">${c.bottleneck_score} môn</div>
                        </div>
                        <div class="chip">
                            <div class="chip-label">Tỷ lệ trượt</div>
                            <div class="chip-val">${c.fail_rate}%</div>
                        </div>
                        <div class="chip">
                            <div class="chip-label">Phổ biến</div>
                            <div class="chip-val">${(c.popularity * 100).toFixed(0)}%</div>
                        </div>
                        <div class="chip">
                            <div class="chip-label">Điểm Rank</div>
                            <div class="chip-val" style="color:#2563eb;">${c.score}</div>
                        </div>
                    </div>
                    <button class="btn-shap" onclick="showShap(${c.rank - 1})">
                        📊 Phân tích lực SHAP (Tại sao xếp hạng #${c.rank}?)
                    </button>
                </div>
            `).join('');
            document.getElementById('lambdamartList').innerHTML = lmHtml;

            // 2. Render Top 5 Baseline
            const baseHtml = data.baseline_top5.map(c => `
                <div class="course-card" style="opacity: 0.9;">
                    <div class="card-top">
                        <div class="rank-badge" style="background:#94a3b8;">#${c.rank}</div>
                        <div class="course-title" style="font-weight:600;">${c.course_name}</div>
                        ${c.actually_registered ? '<span class="tag-registered">✓ Đã đăng ký</span>' : ''}
                    </div>
                    <div class="chips-grid" style="grid-template-columns: 1fr;">
                        <div class="chip" style="text-align: left; padding-left: 8px;">
                            <span class="chip-label">Độ phổ biến toàn trường:</span>
                            <span class="chip-val">${(c.popularity * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                </div>
            `).join('');
            document.getElementById('baselineList').innerHTML = baseHtml;

            // 3. Tab Pipeline
            document.getElementById('historySummary').innerText = `Đã tích lũy: ${data.total_past_courses} môn | Số môn nợ: ${data.failed_count}`;
            const histBody = document.querySelector('#tableRawHistory tbody');
            histBody.innerHTML = data.history_sample.map(h => `
                <tr>
                    <td>Năm ${h.academic_year} K${h.semester}</td>
                    <td><strong>${h.course_name}</strong></td>
                    <td style="font-family: monospace; font-weight:700;">${h.score !== null ? h.score : 'Pass/Fail'}</td>
                    <td>${h.score !== null ? (h.score >= 60 ? '<span style="color:#16a34a; font-weight:700;">✓ Đạt</span>' : '<span style="color:#dc2626; font-weight:700;">✗ Trượt</span>') : '---'}</td>
                </tr>
            `).join('');

            const featBody = document.querySelector('#tableFeatureMatrix tbody');
            featBody.innerHTML = data.feature_matrix.slice(0, 7).map(f => `
                <tr>
                    <td><strong>${f.course_name}</strong></td>
                    <td style="font-weight:700; color:#dc2626; text-align:center;">${f.bottleneck_score}</td>
                    <td style="text-align:center;">${f.fail_rate}</td>
                    <td style="text-align:center;">${f.popularity}</td>
                    <td style="text-align:center;">${f.student_gpa}</td>
                    <td style="text-align:center;">${f.is_core ? 'Có' : 'Không'}</td>
                    <td style="text-align:center; font-weight:700;">P${f.persona}</td>
                    <td style="text-align:center;">${f.target === 1 ? '<span class="tag-registered">1 (Chọn)</span>' : '<span style="color:#94a3b8;">0</span>'}</td>
                </tr>
            `).join('');
        }

        function showShap(index) {
            const course = currentStudentData.lambdamart_top5[index];
            document.getElementById('modalCourseTitle').innerText = `#${course.rank} - ${course.course_name}`;
            document.getElementById('modalCourseSub').innerText = `Sinh viên: ${currentStudentData.student_id} | Điểm GPA: ${currentStudentData.gpa} | Chiến lược: ${currentStudentData.persona_meta.code}`;
            document.getElementById('modalBaseVal').innerText = course.base_value;
            document.getElementById('modalFinalScore').innerText = course.score;

            const featureLabels = {
                'bottleneck_score': 'Mở khóa môn sau',
                'fail_rate': 'Tỷ lệ trượt môn',
                'popularity': 'Độ phổ biến',
                'student_gpa': 'Điểm GPA sinh viên',
                'is_core_major': 'Môn thuộc ngành',
                'persona': 'Nhóm học vụ'
            };

            const rowsHtml = course.shap_breakdown.map(b => {
                const isPos = b.shap_val >= 0;
                const widthPercent = Math.min(Math.abs(b.shap_val) * 45, 100);
                const displayLabel = featureLabels[b.feature] || b.feature;
                return `
                    <div class="shap-row">
                        <div class="shap-feat" title="${displayLabel}">${displayLabel}</div>
                        <div class="shap-bar-bg">
                            <div class="shap-bar-val ${isPos ? 'pos' : 'neg'}" style="width: ${widthPercent}%;"></div>
                        </div>
                        <div class="shap-num" style="color: ${isPos ? '#10b981' : '#ef4444'}">
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

        function closeModalOnOverlay(e) {
            if (e.target.id === 'shapModal') {
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
            persona_param = query.get('persona_id', [None])[0]
            p_override = int(persona_param) if persona_param is not None and persona_param != '' else None
            
            result = get_recommendation_and_pipeline(s_id, active_persona_id=p_override)
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

def run_server(port=None):
    if port is None:
        port = int(os.environ.get('PORT', 5000))
    server = HTTPServer(('0.0.0.0', port), RequestHandler)
    url = f"http://localhost:{port}"
    print(f"\n" + "=" * 75)
    print(f"  [HOÀN TẤT] MÁY CHỦ WEB DEMO ĐANG CHẠY TẠI: {url}")
    print(f"  (Đã tích hợp Hệ thống Thích ứng 6 Personas P1 -> P6)")
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
    run_server()
