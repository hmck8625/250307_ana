import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import gspread
from google.oauth2.service_account import Credentials
from oauth2client.service_account import ServiceAccountCredentials
import openai
import os
import json

# ページ設定
st.set_page_config(
    page_title="広告パフォーマンス分析システム",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# サイドバーにタイトルを表示
st.sidebar.title("広告パフォーマンス分析システム")

# セッション状態の初期化
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None
if 'auto_analysis_result' not in st.session_state:
    st.session_state['auto_analysis_result'] = None

# [変更2] APIキーをサイドバーに移動
with st.sidebar.expander("API設定", expanded=False):
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        st.success("APIキーが設定されました")

# Googleスプレッドシートからデータを読み込む関数
def load_data_from_gsheet(url, sheet_name):
    try:
        st.sidebar.info("Google Spreadsheetからデータを読み込んでいます...")
        
        # Google APIの認証スコープ
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        
        # 認証情報の取得（実際のデプロイ時には、より安全な方法で認証情報を管理する必要があります）
        # Streamlit Secretsを使用する場合
        if 'gcp_service_account' in st.secrets:
            credentials_info = st.secrets['gcp_service_account']
            credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
        else:
            # ローカル開発用（サービスアカウントのJSONファイルが必要）
            # 注意: この方法は本番環境では使用しないでください
            st.sidebar.warning("サービスアカウントキーがセットアップされていません。credentials.jsonファイルが必要です。")
            credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        
        client = gspread.authorize(credentials)
        
        # URLからスプレッドシートを開く
        spreadsheet_id = url.split('/d/')[1].split('/')[0]
        sheet = client.open_by_key(spreadsheet_id).worksheet(sheet_name)
        
        # データをDataFrameに読み込む
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # データ型の変換
        numeric_cols = ['Impressions', 'Clicks', 'Cost', 'Conversions']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 日付の変換
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        # 欠損値のチェックと処理
        if df.isna().sum().sum() > 0:
            st.sidebar.warning(f"データに欠損値が {df.isna().sum().sum()} 件あります。")
            # 数値型の欠損値を0に置換
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        
        st.sidebar.success(f"データ読み込み完了: {len(df)} 行のデータを読み込みました")
        
        return df
    
    except Exception as e:
        st.sidebar.error(f"データ読み込み中にエラーが発生しました: {str(e)}")
        return None

# 派生指標の計算関数
def calculate_derived_metrics(df):
    # 0除算を避けるための関数
    def safe_divide(x, y):
        return np.where(y != 0, x / y, 0)
    
    # 派生指標の計算
    if 'Impressions' in df.columns and 'Clicks' in df.columns:
        df['CTR'] = safe_divide(df['Clicks'], df['Impressions']) * 100
    
    if 'Clicks' in df.columns and 'Conversions' in df.columns:
        df['CVR'] = safe_divide(df['Conversions'], df['Clicks']) * 100
    
    if 'Clicks' in df.columns and 'Cost' in df.columns:
        df['CPC'] = safe_divide(df['Cost'], df['Clicks'])
    
    if 'Conversions' in df.columns and 'Cost' in df.columns:
        df['CPA'] = safe_divide(df['Cost'], df['Conversions'])
    
    if 'Impressions' in df.columns and 'Cost' in df.columns:
        df['CPM'] = safe_divide(df['Cost'], df['Impressions']) * 1000
    
    return df

# データをフィルタリングする関数（期間指定）
def filter_data_by_date(df, start_date, end_date):
    if 'Date' not in df.columns:
        st.error("データに日付列がありません")
        return df
    
    # datetime.date オブジェクトを pandas の Timestamp に変換
    # 型チェックを単純化
    if hasattr(start_date, 'date') and not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    elif hasattr(start_date, 'year') and hasattr(start_date, 'month') and hasattr(start_date, 'day'):
        # dateオブジェクトの特性を持つオブジェクトかどうかをチェック
        start_date = pd.Timestamp(start_date)
        
    if hasattr(end_date, 'date') and not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)
    elif hasattr(end_date, 'year') and hasattr(end_date, 'month') and hasattr(end_date, 'day'):
        # dateオブジェクトの特性を持つオブジェクトかどうかをチェック
        end_date = pd.Timestamp(end_date)
    
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return filtered_df

# 期間ごとのデータを集計する関数
def aggregate_data_by_period(df, group_by_cols=['ServiceNameJA']):
    # 必須フィールドが存在するか確認
    required_cols = ['Impressions', 'Clicks', 'Cost', 'Conversions']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"必須フィールド {col} がデータにありません")
            return None
    
    # グループ化と集計
    agg_dict = {
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Cost': 'sum',
        'Conversions': 'sum'
    }
    
    # 集計
    agg_df = df.groupby(group_by_cols).agg(agg_dict).reset_index()
    
    # 派生指標の計算
    agg_df = calculate_derived_metrics(agg_df)
    
    return agg_df

# 期間比較のための分析関数
def compare_periods(current_df, previous_df, group_by_cols=['ServiceNameJA']):
    """
    二つの期間のデータを比較して分析結果を返す
    
    Parameters:
    current_df (DataFrame): 当期のデータ
    previous_df (DataFrame): 前期のデータ
    group_by_cols (list): グループ化するカラム
    
    Returns:
    dict: 分析結果を含む辞書
    """
    # グループ化と集計
    current_agg = aggregate_data_by_period(current_df, group_by_cols)
    previous_agg = aggregate_data_by_period(previous_df, group_by_cols)
    
    if current_agg is None or previous_agg is None:
        return None
    
    # 合計値の計算
    current_total = current_agg.sum(numeric_only=True).to_dict()
    previous_total = previous_agg.sum(numeric_only=True).to_dict()
    
    # 日数の計算（日平均値の計算用）
    if 'Date' in current_df.columns and 'Date' in previous_df.columns:
        current_days = (current_df['Date'].max() - current_df['Date'].min()).days + 1
        previous_days = (previous_df['Date'].max() - previous_df['Date'].min()).days + 1
    else:
        current_days = 30  # デフォルト値
        previous_days = 30  # デフォルト値
    
    # 1. CV増減の寄与度分析
    cv_contribution = analyze_cv_contribution(current_agg, previous_agg, group_by_cols)
    
    # 2. CPA変化要因分析
    cpa_change_factors = analyze_cpa_change_factors(current_agg, previous_agg, group_by_cols)
    
    # 3. 媒体グループ・パターン分析
    media_patterns = analyze_media_patterns(current_agg, previous_agg, group_by_cols)
    
    # 分析結果を辞書にまとめる
    result = {
        'current_agg': current_agg,
        'previous_agg': previous_agg,
        'current_total': current_total,
        'previous_total': previous_total,
        'current_days': current_days,
        'previous_days': previous_days,
        'cv_contribution': cv_contribution,
        'cpa_change_factors': cpa_change_factors,
        'media_patterns': media_patterns
    }
    
    return result

# 1. CV増減の寄与度分析
# 1. CV増減の寄与度分析（修正版）
def analyze_cv_contribution(current_agg, previous_agg, group_by_cols=['ServiceNameJA']):
    """
    CV増減の寄与度分析を行う
    
    Parameters:
    current_agg (DataFrame): 当期の集計データ
    previous_agg (DataFrame): 前期の集計データ
    group_by_cols (list): グループ化したカラム
    
    Returns:
    DataFrame: 寄与度分析結果
    """
    # 媒体名などのマッチング用にインデックスを設定
    current_df = current_agg.set_index(group_by_cols)
    previous_df = previous_agg.set_index(group_by_cols)
    
    # 共通のインデックスを取得
    common_indices = set(current_df.index) & set(previous_df.index)
    
    # データフレームの準備
    contribution_data = []
    
    # 全体のCV変化量を計算
    total_current_cv = current_df['Conversions'].sum()
    total_previous_cv = previous_df['Conversions'].sum()
    total_cv_change = total_current_cv - total_previous_cv
    
    # 各媒体のCV変化と寄与率を計算
    for idx in common_indices:
        current_cv = current_df.loc[idx, 'Conversions']
        previous_cv = previous_df.loc[idx, 'Conversions']
        cv_change = current_cv - previous_cv
        
        # 寄与率の計算 (全体のCV変化が0の場合は特別処理)
        if total_cv_change != 0:
            contribution_rate = (cv_change / total_cv_change) * 100
        else:
            contribution_rate = 0 if cv_change == 0 else float('inf') if cv_change > 0 else float('-inf')
        
        # 新規または終了した媒体の場合
        entry_status = "継続"
        if idx not in previous_df.index:
            entry_status = "新規"
            previous_cv = 0
        elif idx not in current_df.index:
            entry_status = "終了"
            current_cv = 0
        
        contribution_data.append({
            'index_value': idx,  # index列の名前を変更
            'previous_cv': previous_cv,
            'current_cv': current_cv,
            'cv_change': cv_change,
            'contribution_rate': contribution_rate,
            'entry_status': entry_status
        })
    
    # 新規追加された媒体の処理
    for idx in set(current_df.index) - set(previous_df.index):
        current_cv = current_df.loc[idx, 'Conversions']
        cv_change = current_cv
        
        # 寄与率の計算
        if total_cv_change != 0:
            contribution_rate = (cv_change / total_cv_change) * 100
        else:
            contribution_rate = float('inf') if cv_change > 0 else float('-inf')
        
        contribution_data.append({
            'index_value': idx,  # index列の名前を変更
            'previous_cv': 0,
            'current_cv': current_cv,
            'cv_change': cv_change,
            'contribution_rate': contribution_rate,
            'entry_status': "新規"
        })
    
    # 終了した媒体の処理
    for idx in set(previous_df.index) - set(current_df.index):
        previous_cv = previous_df.loc[idx, 'Conversions']
        cv_change = -previous_cv
        
        # 寄与率の計算
        if total_cv_change != 0:
            contribution_rate = (cv_change / total_cv_change) * 100
        else:
            contribution_rate = float('inf') if cv_change > 0 else float('-inf')
        
        contribution_data.append({
            'index_value': idx,  # index列の名前を変更
            'previous_cv': previous_cv,
            'current_cv': 0,
            'cv_change': cv_change,
            'contribution_rate': contribution_rate,
            'entry_status': "終了"
        })
    
    # データが空の場合、空のDataFrameを返す
    if not contribution_data:
        # 空のデータフレームを作成（必要な列を持つ）
        empty_df = pd.DataFrame(columns=group_by_cols + [
            'previous_cv', 'current_cv', 'cv_change', 'contribution_rate', 'entry_status'])
        return empty_df
    
    # DataFrameに変換
    contribution_df = pd.DataFrame(contribution_data)
    
    # マルチインデックスの場合の処理
    if len(group_by_cols) > 1:
        # インデックスの分解
        for i, col in enumerate(group_by_cols):
            contribution_df[col] = contribution_df['index_value'].apply(lambda x: x[i])
    else:
        # 単一インデックスの場合
        contribution_df[group_by_cols[0]] = contribution_df['index_value']
    
    # インデックス列を削除
    if 'index_value' in contribution_df.columns:
        contribution_df = contribution_df.drop(columns=['index_value'])
    
    # 寄与率の絶対値で降順ソート
    if 'contribution_rate' in contribution_df.columns:
        contribution_df['abs_contribution'] = contribution_df['contribution_rate'].abs()
        contribution_df = contribution_df.sort_values('abs_contribution', ascending=False)
        contribution_df = contribution_df.drop(columns=['abs_contribution'])
    
    return contribution_df
# 2. CPA変化要因分析
# 2. CPA変化要因分析（修正版）
def analyze_cpa_change_factors(current_agg, previous_agg, group_by_cols=['ServiceNameJA']):
    """
    CPA変化の要因分析を行う
    
    Parameters:
    current_agg (DataFrame): 当期の集計データ
    previous_agg (DataFrame): 前期の集計データ
    group_by_cols (list): グループ化したカラム
    
    Returns:
    DataFrame: CPA変化要因分析結果
    """
    # 媒体名などのマッチング用にインデックスを設定
    current_df = current_agg.set_index(group_by_cols)
    previous_df = previous_agg.set_index(group_by_cols)
    
    # 共通のインデックスを取得
    common_indices = set(current_df.index) & set(previous_df.index)
    
    # データフレームの準備
    factor_data = []
    
    # 各媒体のCPA変化要因を分析
    for idx in common_indices:
        # 前期・当期のCPA, CVR, CPCを取得
        try:
            current_cpa = current_df.loc[idx, 'CPA']
            previous_cpa = previous_df.loc[idx, 'CPA']
            
            current_cvr = current_df.loc[idx, 'CVR']
            previous_cvr = previous_df.loc[idx, 'CVR']
            
            current_cpc = current_df.loc[idx, 'CPC']
            previous_cpc = previous_df.loc[idx, 'CPC']
            
            current_cpm = current_df.loc[idx, 'CPM']
            previous_cpm = previous_df.loc[idx, 'CPM']
            
            current_ctr = current_df.loc[idx, 'CTR']
            previous_ctr = previous_df.loc[idx, 'CTR']
            
            # 変化率の計算
            cpa_change_rate = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else float('inf')
            cvr_change_rate = ((current_cvr - previous_cvr) / previous_cvr) * 100 if previous_cvr != 0 else float('inf')
            cpc_change_rate = ((current_cpc - previous_cpc) / previous_cpc) * 100 if previous_cpc != 0 else float('inf')
            cpm_change_rate = ((current_cpm - previous_cpm) / previous_cpm) * 100 if previous_cpm != 0 else float('inf')
            ctr_change_rate = ((current_ctr - previous_ctr) / previous_ctr) * 100 if previous_ctr != 0 else float('inf')
            
            # 主要因判定
            cvr_factor = abs(cvr_change_rate)
            cpc_factor = abs(cpc_change_rate)
            
            if cvr_factor > cpc_factor:
                main_factor = "CVR"
                secondary_factor = None
                description = f"CVRが{cvr_change_rate:.1f}%変化したことが主要因"
            else:
                main_factor = "CPC"
                # 副要因判定
                if abs(cpm_change_rate) > abs(ctr_change_rate):
                    secondary_factor = "CPM"
                    description = f"CPCが{cpc_change_rate:.1f}%変化し、CPM({cpm_change_rate:.1f}%)の影響が大きい"
                else:
                    secondary_factor = "CTR"
                    description = f"CPCが{cpc_change_rate:.1f}%変化し、CTR({ctr_change_rate:.1f}%)の影響が大きい"
            
            # パフォーマンスの変化の説明を生成
            if current_cpa < previous_cpa:
                performance_change = "改善"
            else:
                performance_change = "悪化"
            
            # 現在の各指標の値も追加
            factor_data.append({
                'index_value': idx,  # index列の名前を変更
                'previous_impressions': previous_df.loc[idx, 'Impressions'],
                'current_impressions': current_df.loc[idx, 'Impressions'],
                'previous_cpm': previous_cpm,
                'current_cpm': current_cpm,
                'previous_clicks': previous_df.loc[idx, 'Clicks'],
                'current_clicks': current_df.loc[idx, 'Clicks'],
                'previous_ctr': previous_ctr,
                'current_ctr': current_ctr,
                'previous_cpc': previous_cpc,
                'current_cpc': current_cpc,
                'previous_cost': previous_df.loc[idx, 'Cost'],
                'current_cost': current_df.loc[idx, 'Cost'],
                'previous_cv': previous_df.loc[idx, 'Conversions'],
                'current_cv': current_df.loc[idx, 'Conversions'],
                'previous_cpa': previous_cpa,
                'current_cpa': current_cpa,
                'previous_cvr': previous_cvr,
                'current_cvr': current_cvr,
                'cpa_change_rate': cpa_change_rate,
                'cvr_change_rate': cvr_change_rate,
                'cpc_change_rate': cpc_change_rate,
                'cpm_change_rate': cpm_change_rate,
                'ctr_change_rate': ctr_change_rate,
                'main_factor': main_factor,
                'secondary_factor': secondary_factor,
                'description': description,
                'performance_change': performance_change
            })
        
        except Exception as e:
            st.warning(f"CPA変化要因分析でエラーが発生しました（{idx}）: {str(e)}")
            continue
    
    # データが空の場合、空のDataFrameを返す
    if not factor_data:
        # 空のデータフレームを作成（必要な列を持つ）
        empty_df = pd.DataFrame(columns=group_by_cols + [
            'previous_cpa', 'current_cpa', 'cpa_change_rate', 'main_factor', 
            'secondary_factor', 'description', 'performance_change'])
        return empty_df
    
    # DataFrameに変換
    factor_df = pd.DataFrame(factor_data)
    
    # マルチインデックスの場合の処理
    if len(group_by_cols) > 1:
        # インデックスの分解
        for i, col in enumerate(group_by_cols):
            factor_df[col] = factor_df['index_value'].apply(lambda x: x[i])
    else:
        # 単一インデックスの場合
        factor_df[group_by_cols[0]] = factor_df['index_value']
    
    # インデックス列を削除
    if 'index_value' in factor_df.columns:
        factor_df = factor_df.drop(columns=['index_value'])
    
    # CPA変化率の絶対値で降順ソート
    if 'cpa_change_rate' in factor_df.columns:
        factor_df['abs_cpa_change'] = factor_df['cpa_change_rate'].abs()
        factor_df = factor_df.sort_values('abs_cpa_change', ascending=False)
        factor_df = factor_df.drop(columns=['abs_cpa_change'])
    
    return factor_df
# 3. 媒体グループ・パターン分析# 3. 媒体グループ・パターン分析（修正版）
def analyze_media_patterns(current_agg, previous_agg, group_by_cols=['ServiceNameJA']):
    """
    媒体のグループ・パターン分析を行う
    
    Parameters:
    current_agg (DataFrame): 当期の集計データ
    previous_agg (DataFrame): 前期の集計データ
    group_by_cols (list): グループ化したカラム
    
    Returns:
    dict: パターン別の媒体グループ
    """
    # 媒体名などのマッチング用にインデックスを設定
    current_df = current_agg.set_index(group_by_cols)
    previous_df = previous_agg.set_index(group_by_cols)
    
    # 共通のインデックスを取得
    common_indices = set(current_df.index) & set(previous_df.index)
    
    # パターン分類用の辞書
    patterns = {
        'success': [],  # CV増加かつCPA改善
        'growth': [],   # CV増加かつCPA悪化
        'efficiency': [], # CV減少かつCPA改善
        'issue': []     # CV減少かつCPA悪化
    }
    
    pattern_data = []
    
    # 各媒体のパターンを分析
    for idx in common_indices:
        try:
            current_cv = current_df.loc[idx, 'Conversions']
            previous_cv = previous_df.loc[idx, 'Conversions']
            
            current_cpa = current_df.loc[idx, 'CPA']
            previous_cpa = previous_df.loc[idx, 'CPA']
            
            # CV増減とCPA改善・悪化の判定
            cv_change = current_cv - previous_cv
            cpa_change = current_cpa - previous_cpa
            
            # 判定結果に基づいてパターン分類
            if cv_change >= 0 and cpa_change <= 0:
                pattern = 'success'  # CV増加かつCPA改善
            elif cv_change >= 0 and cpa_change > 0:
                pattern = 'growth'   # CV増加かつCPA悪化
            elif cv_change < 0 and cpa_change <= 0:
                pattern = 'efficiency' # CV減少かつCPA改善
            else:  # cv_change < 0 and cpa_change > 0
                pattern = 'issue'    # CV減少かつCPA悪化
            
            # パターンに追加
            patterns[pattern].append(idx)
            
            pattern_data.append({
                'index_value': idx,  # index列の名前を変更
                'previous_cv': previous_cv,
                'current_cv': current_cv,
                'cv_change': cv_change,
                'previous_cpa': previous_cpa,
                'current_cpa': current_cpa,
                'cpa_change': cpa_change,
                'pattern': pattern
            })
        
        except Exception as e:
            st.warning(f"媒体パターン分析でエラーが発生しました（{idx}）: {str(e)}")
            continue
    
    # データが空の場合、空のデータフレームを返す
    if not pattern_data:
        # 空のデータフレームを作成（必要な列を持つ）
        empty_df = pd.DataFrame(columns=group_by_cols + [
            'previous_cv', 'current_cv', 'cv_change', 'previous_cpa', 
            'current_cpa', 'cpa_change', 'pattern', 'pattern_name'])
        return {
            'pattern_groups': patterns,
            'pattern_df': empty_df
        }
    
    # DataFrameに変換
    pattern_df = pd.DataFrame(pattern_data)
    
    # マルチインデックスの場合の処理
    if len(group_by_cols) > 1:
        # インデックスの分解
        for i, col in enumerate(group_by_cols):
            pattern_df[col] = pattern_df['index_value'].apply(lambda x: x[i])
    else:
        # 単一インデックスの場合
        pattern_df[group_by_cols[0]] = pattern_df['index_value']
    
    # インデックス列を削除
    if 'index_value' in pattern_df.columns:
        pattern_df = pattern_df.drop(columns=['index_value'])
    
    # パターン名を日本語に変換
    pattern_names = {
        'success': '成功パターン（CV増加かつCPA改善）',
        'growth': '成長重視パターン（CV増加かつCPA悪化）',
        'efficiency': '効率重視パターン（CV減少かつCPA改善）',
        'issue': '課題パターン（CV減少かつCPA悪化）'
    }
    
    pattern_df['pattern_name'] = pattern_df['pattern'].map(pattern_names)
    
    return {
        'pattern_groups': patterns,
        'pattern_df': pattern_df
    }
# [変更5] 重要媒体の自動特定

def identify_important_media(analysis_result, cv_contribution_threshold=30, cpa_change_threshold=20):
    """
    重要な媒体を自動的に特定する
    
    Parameters:
    analysis_result (dict): 分析結果
    cv_contribution_threshold (float): CV寄与率の閾値 (%)
    cpa_change_threshold (float): CPA変化率の閾値 (%)
    
    Returns:
    dict: 重要媒体のリスト
    """
    important_media = {
        'high_cv_contribution': [],  # CV寄与率が高い媒体
        'high_cpa_change': [],       # CPA変化率が大きい媒体
        'success_pattern': [],       # 成功パターンの媒体
        'issue_pattern': []          # 課題パターンの媒体
    }
    
    # CV寄与率が高い媒体
    cv_contribution = analysis_result['cv_contribution']
    for _, row in cv_contribution.iterrows():
        if abs(row['contribution_rate']) >= cv_contribution_threshold:
            media_info = {
                'media_name': row['ServiceNameJA'] if 'ServiceNameJA' in row else 'Unknown',
                'contribution_rate': row['contribution_rate'],
                'previous_cv': row['previous_cv'],
                'current_cv': row['current_cv'],
                'cv_change': row['cv_change']
            }
            important_media['high_cv_contribution'].append(media_info)
    
    # CPA変化率が大きい媒体
    cpa_factors = analysis_result['cpa_change_factors']
    for _, row in cpa_factors.iterrows():
        if abs(row['cpa_change_rate']) >= cpa_change_threshold:
            media_info = {
                'media_name': row['ServiceNameJA'] if 'ServiceNameJA' in row else 'Unknown',
                'cpa_change_rate': row['cpa_change_rate'],
                'main_factor': row['main_factor'],
                'secondary_factor': row['secondary_factor'],
                'previous_cpa': row['previous_cpa'],
                'current_cpa': row['current_cpa'],
                'description': row['description']
            }
            important_media['high_cpa_change'].append(media_info)
    
    # 成功パターンの媒体
    patterns = analysis_result['media_patterns']['pattern_df']
    for _, row in patterns.iterrows():
        if row['pattern'] == 'success':
            media_info = {
                'media_name': row['ServiceNameJA'] if 'ServiceNameJA' in row else 'Unknown',
                'cv_change': row['cv_change'],
                'cpa_change': row['cpa_change'],
                'pattern': row['pattern'],
                'pattern_name': row['pattern_name']
            }
            important_media['success_pattern'].append(media_info)
        elif row['pattern'] == 'issue':
            media_info = {
                'media_name': row['ServiceNameJA'] if 'ServiceNameJA' in row else 'Unknown',
                'cv_change': row['cv_change'],
                'cpa_change': row['cpa_change'],
                'pattern': row['pattern'],
                'pattern_name': row['pattern_name']
            }
            important_media['issue_pattern'].append(media_info)
    
    return important_media

# [変更5] 自動分析機能
def auto_analyze(analysis_result, df, previous_df, current_df, group_by_cols=['ServiceNameJA']):
    """
    自動分析プロセスを実行する
    
    Parameters:
    analysis_result (dict): トップレベルの分析結果
    df (DataFrame): 全体データ
    previous_df (DataFrame): 前期データ
    current_df (DataFrame): 当期データ
    group_by_cols (list): グループ化したカラム
    
    Returns:
    dict: 自動分析結果
    """
    # 重要媒体の特定
    important_media = identify_important_media(analysis_result)
    
    # 媒体レベルからキャンペーンレベルへの分析
    campaign_analysis = {}
    
    # CV寄与率が高い媒体のキャンペーン分析
    for media_info in important_media['high_cv_contribution']:
        media_name = media_info['media_name']
        
        # 媒体フィルタリング
        media_previous_df = previous_df[previous_df['ServiceNameJA'] == media_name]
        media_current_df = current_df[current_df['ServiceNameJA'] == media_name]
        
        # キャンペーンレベル分析
        if 'CampaignName' in media_previous_df.columns and 'CampaignName' in media_current_df.columns:
            campaign_result = compare_periods(media_current_df, media_previous_df, ['CampaignName'])
            if campaign_result:
                campaign_analysis[media_name] = {
                    'type': 'cv_contribution',
                    'contribution_rate': media_info['contribution_rate'],
                    'analysis': campaign_result
                }
    
    # CPA変化率が大きい媒体のキャンペーン分析
    for media_info in important_media['high_cpa_change']:
        media_name = media_info['media_name']
        
        # 既に分析済みならスキップ
        if media_name in campaign_analysis:
            continue
        
        # 媒体フィルタリング
        media_previous_df = previous_df[previous_df['ServiceNameJA'] == media_name]
        media_current_df = current_df[current_df['ServiceNameJA'] == media_name]
        
        # キャンペーンレベル分析
        if 'CampaignName' in media_previous_df.columns and 'CampaignName' in media_current_df.columns:
            campaign_result = compare_periods(media_current_df, media_previous_df, ['CampaignName'])
            if campaign_result:
                campaign_analysis[media_name] = {
                    'type': 'cpa_change',
                    'cpa_change_rate': media_info['cpa_change_rate'],
                    'main_factor': media_info['main_factor'],
                    'secondary_factor': media_info['secondary_factor'],
                    'analysis': campaign_result
                }
    
    # 広告グループレベル分析（重要なキャンペーンに対して）
    adgroup_analysis = {}
    
    for media_name, campaign_data in campaign_analysis.items():
        campaign_result = campaign_data['analysis']
        
        # 重要なキャンペーンを特定
        important_campaigns = identify_important_campaigns(campaign_result)
        
        for campaign_info in important_campaigns:
            campaign_name = campaign_info['campaign_name']
            
            # 媒体・キャンペーンでフィルタリング
            campaign_previous_df = previous_df[
                (previous_df['ServiceNameJA'] == media_name) & 
                (previous_df['CampaignName'] == campaign_name)
            ]
            campaign_current_df = current_df[
                (current_df['ServiceNameJA'] == media_name) & 
                (current_df['CampaignName'] == campaign_name)
            ]
            
            # 広告グループレベル分析
            if 'AdgroupName' in campaign_previous_df.columns and 'AdgroupName' in campaign_current_df.columns:
                adgroup_result = compare_periods(campaign_current_df, campaign_previous_df, ['AdgroupName'])
                if adgroup_result:
                    key = f"{media_name}_{campaign_name}"
                    adgroup_analysis[key] = {
                        'media_name': media_name,
                        'campaign_name': campaign_name,
                        'type': campaign_info['type'],
                        'analysis': adgroup_result
                    }
    
    # 分析結果をまとめる
    auto_analysis_result = {
        'important_media': important_media,
        'campaign_analysis': campaign_analysis,
        'adgroup_analysis': adgroup_analysis
    }
    
    return auto_analysis_result

# 重要なキャンペーンを特定する関数
def identify_important_campaigns(campaign_result, cv_contribution_threshold=30, cpa_change_threshold=20):
    """
    重要なキャンペーンを特定する
    
    Parameters:
    campaign_result (dict): キャンペーンレベルの分析結果
    cv_contribution_threshold (float): CV寄与率の閾値 (%)
    cpa_change_threshold (float): CPA変化率の閾値 (%)
    
    Returns:
    list: 重要キャンペーンのリスト
    """
    important_campaigns = []
    
    # CV寄与率が高いキャンペーン
    cv_contribution = campaign_result['cv_contribution']
    for _, row in cv_contribution.iterrows():
        if abs(row['contribution_rate']) >= cv_contribution_threshold:
            campaign_info = {
                'campaign_name': row['CampaignName'] if 'CampaignName' in row else 'Unknown',
                'type': 'cv_contribution',
                'contribution_rate': row['contribution_rate'],
                'previous_cv': row['previous_cv'],
                'current_cv': row['current_cv'],
                'cv_change': row['cv_change']
            }
            important_campaigns.append(campaign_info)
    
    # CPA変化率が大きいキャンペーン
    cpa_factors = campaign_result['cpa_change_factors']
    for _, row in cpa_factors.iterrows():
        if abs(row['cpa_change_rate']) >= cpa_change_threshold:
            # 既に追加済みかチェック
            exists = False
            for campaign in important_campaigns:
                if campaign['campaign_name'] == row['CampaignName']:
                    exists = True
                    break
            
            if not exists:
                campaign_info = {
                    'campaign_name': row['CampaignName'] if 'CampaignName' in row else 'Unknown',
                    'type': 'cpa_change',
                    'cpa_change_rate': row['cpa_change_rate'],
                    'main_factor': row['main_factor'],
                    'secondary_factor': row['secondary_factor'],
                    'previous_cpa': row['previous_cpa'],
                    'current_cpa': row['current_cpa']
                }
                important_campaigns.append(campaign_info)
    
    return important_campaigns

# 分析結果をプロンプト用に整形する関数
def format_prompt_data(analysis_result):
    """
    分析結果をChatGPT用のプロンプトデータに整形する
    
    Parameters:
    analysis_result (dict): 分析結果
    
    Returns:
    dict: 整形されたデータ
    """
    # 期間の総括データ
    current_total = analysis_result['current_total']
    previous_total = analysis_result['previous_total']
    
    # 日付情報（あれば）
    current_days = analysis_result.get('current_days', 30)
    previous_days = analysis_result.get('previous_days', 30)
    
    # CV増減の寄与度分析
    cv_contribution = analysis_result['cv_contribution']
    
    # CPA変化要因分析
    cpa_change_factors = analysis_result['cpa_change_factors']
    
    # 媒体グループ・パターン分析
    media_patterns = analysis_result['media_patterns']['pattern_df']
    
    # 自動分析結果（あれば）
    auto_analysis = None
    if 'auto_analysis_result' in st.session_state and st.session_state['auto_analysis_result']:
        auto_analysis = st.session_state['auto_analysis_result']
    
    # データを整形
    formatted_data = {
        'summary': {
            'previous': {
                'impressions': previous_total.get('Impressions', 0),
                'clicks': previous_total.get('Clicks', 0),
                'cost': previous_total.get('Cost', 0),
                'conversions': previous_total.get('Conversions', 0),
                'days': previous_days
            },
            'current': {
                'impressions': current_total.get('Impressions', 0),
                'clicks': current_total.get('Clicks', 0),
                'cost': current_total.get('Cost', 0),
                'conversions': current_total.get('Conversions', 0),
                'days': current_days
            }
        },
        'cv_contribution': cv_contribution.to_dict('records'),
        'cpa_change_factors': cpa_change_factors.to_dict('records'),
        'media_patterns': media_patterns.to_dict('records'),
        'auto_analysis': auto_analysis
    }
    
    return formatted_data

# 分析プロンプトを作成する関数
def create_analysis_prompt(data):
    """
    分析用のプロンプトを作成する
    
    Parameters:
    data (dict): 整形された分析データ
    
    Returns:
    str: 分析プロンプト
    """
    # サマリーデータの取得
    summary = data['summary']
    previous = summary['previous']
    current = summary['current']
    
    # 変化率の計算
    imp_change = ((current['impressions'] - previous['impressions']) / previous['impressions']) * 100 if previous['impressions'] != 0 else float('inf')
    clicks_change = ((current['clicks'] - previous['clicks']) / previous['clicks']) * 100 if previous['clicks'] != 0 else float('inf')
    cost_change = ((current['cost'] - previous['cost']) / previous['cost']) * 100 if previous['cost'] != 0 else float('inf')
    cv_change = ((current['conversions'] - previous['conversions']) / previous['conversions']) * 100 if previous['conversions'] != 0 else float('inf')
    
    # インプレッションと1000回表示あたりのコスト（CPM）
    previous_cpm = (previous['cost'] / previous['impressions']) * 1000 if previous['impressions'] != 0 else 0
    current_cpm = (current['cost'] / current['impressions']) * 1000 if current['impressions'] != 0 else 0
    cpm_change = ((current_cpm - previous_cpm) / previous_cpm) * 100 if previous_cpm != 0 else float('inf')
    
    # クリック率（CTR）
    previous_ctr = (previous['clicks'] / previous['impressions']) * 100 if previous['impressions'] != 0 else 0
    current_ctr = (current['clicks'] / current['impressions']) * 100 if current['impressions'] != 0 else 0
    ctr_change = ((current_ctr - previous_ctr) / previous_ctr) * 100 if previous_ctr != 0 else float('inf')
    
    # クリックあたりのコスト（CPC）
    previous_cpc = previous['cost'] / previous['clicks'] if previous['clicks'] != 0 else 0
    current_cpc = current['cost'] / current['clicks'] if current['clicks'] != 0 else 0
    cpc_change = ((current_cpc - previous_cpc) / previous_cpc) * 100 if previous_cpc != 0 else float('inf')
    
    # コンバージョン率（CVR）
    previous_cvr = (previous['conversions'] / previous['clicks']) * 100 if previous['clicks'] != 0 else 0
    current_cvr = (current['conversions'] / current['clicks']) * 100 if current['clicks'] != 0 else 0
    cvr_change = ((current_cvr - previous_cvr) / previous_cvr) * 100 if previous_cvr != 0 else float('inf')
    
    # コンバージョンあたりのコスト（CPA）
    previous_cpa = previous['cost'] / previous['conversions'] if previous['conversions'] != 0 else 0
    current_cpa = current['cost'] / current['conversions'] if current['conversions'] != 0 else 0
    cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else float('inf')
    
    # 日平均値の計算
    previous_daily_cv = previous['conversions'] / previous['days']
    current_daily_cv = current['conversions'] / current['days']
    daily_cv_change = ((current_daily_cv - previous_daily_cv) / previous_daily_cv) * 100 if previous_daily_cv != 0 else float('inf')
    
    # [変更3] 指標の表示順序の変更、[変更4] 数値フォーマットの調整
    # サマリーテーブル
    summary_table = f"""
| 指標 | 前期 | 当期 | 差分 | 変化率 |
|------|-----|-----|------|--------|
| インプレッション数 | {previous['impressions']:,.0f} | {current['impressions']:,.0f} | {current['impressions'] - previous['impressions']:,.0f} | {imp_change:.1f}% |
| CPM | {previous_cpm:.0f}円 | {current_cpm:.0f}円 | {current_cpm - previous_cpm:.0f}円 | {cpm_change:.1f}% |
| クリック数 | {previous['clicks']:,.0f} | {current['clicks']:,.0f} | {current['clicks'] - previous['clicks']:,.0f} | {clicks_change:.1f}% |
| CTR | {previous_ctr:.1f}% | {current_ctr:.1f}% | {current_ctr - previous_ctr:.1f}% | {ctr_change:.1f}% |
| CPC | {previous_cpc:.0f}円 | {current_cpc:.0f}円 | {current_cpc - previous_cpc:.0f}円 | {cpc_change:.1f}% |
| コスト | {previous['cost']:,.0f}円 | {current['cost']:,.0f}円 | {current['cost'] - previous['cost']:,.0f}円 | {cost_change:.1f}% |
| コンバージョン数 | {previous['conversions']:.1f} | {current['conversions']:.1f} | {current['conversions'] - previous['conversions']:.1f} | {cv_change:.1f}% |
| CPA | {previous_cpa:.0f}円 | {current_cpa:.0f}円 | {current_cpa - previous_cpa:.0f}円 | {cpa_change:.1f}% |
| CVR | {previous_cvr:.1f}% | {current_cvr:.1f}% | {current_cvr - previous_cvr:.1f}% | {cvr_change:.1f}% |
| 日数 | {previous['days']} | {current['days']} | {current['days'] - previous['days']} | - |
| 日平均CV数 | {previous_daily_cv:.1f} | {current_daily_cv:.1f} | {current_daily_cv - previous_daily_cv:.1f} | {daily_cv_change:.1f}% |
"""
    
    # CV増減の寄与度ランキング
    cv_contributions = data['cv_contribution'][:5]  # 上位5件
    
    cv_table = "| 順位 | 媒体名 | 前期CV数 | 当期CV数 | CV数変化 | 寄与率(%) |\n|------|------|------|------|------|------|\n"
    
    for i, item in enumerate(cv_contributions, 1):
        media_name = item.get('ServiceNameJA', 'Unknown')
        previous_cv = item.get('previous_cv', 0)
        current_cv = item.get('current_cv', 0)
        cv_change = item.get('cv_change', 0)
        contribution_rate = item.get('contribution_rate', 0)
        
        cv_table += f"| {i} | {media_name} | {previous_cv:.1f} | {current_cv:.1f} | {cv_change:.1f} | {contribution_rate:.1f}% |\n"
    
    # CPA変化要因分析
    cpa_factors = data['cpa_change_factors'][:5]  # 上位5件
    
    cpa_table = "| 媒体名 | 前期CPA | 当期CPA | CPA変化率 | 主要因 | 変化の詳細説明 |\n|------|------|------|------|------|------|\n"
    
    for item in cpa_factors:
        media_name = item.get('ServiceNameJA', 'Unknown')
        previous_cpa = item.get('previous_cpa', 0)
        current_cpa = item.get('current_cpa', 0)
        cpa_change_rate = item.get('cpa_change_rate', 0)
        main_factor = item.get('main_factor', 'Unknown')
        description = item.get('description', '-')
        
        cpa_table += f"| {media_name} | {previous_cpa:.0f}円 | {current_cpa:.0f}円 | {cpa_change_rate:.1f}% | {main_factor} | {description} |\n"
    
    # 媒体パターン分析
    pattern_counts = {}
    for item in data['media_patterns']:
        pattern = item.get('pattern', 'unknown')
        if pattern in pattern_counts:
            pattern_counts[pattern] += 1
        else:
            pattern_counts[pattern] = 1
    
    pattern_summary = "媒体パターン分類:\n"
    pattern_summary += f"- 成功パターン（CV増加かつCPA改善）: {pattern_counts.get('success', 0)}媒体\n"
    pattern_summary += f"- 成長重視パターン（CV増加かつCPA悪化）: {pattern_counts.get('growth', 0)}媒体\n"
    pattern_summary += f"- 効率重視パターン（CV減少かつCPA改善）: {pattern_counts.get('efficiency', 0)}媒体\n"
    pattern_summary += f"- 課題パターン（CV減少かつCPA悪化）: {pattern_counts.get('issue', 0)}媒体\n"
    
    # 自動分析結果（あれば）
    auto_analysis_summary = ""
    if data['auto_analysis'] is not None:
        auto_analysis = data['auto_analysis']
        important_media = auto_analysis['important_media']
        
        auto_analysis_summary += "\n## 自動分析結果\n\n"
        
        # CV寄与率が高い媒体
        if important_media['high_cv_contribution']:
            auto_analysis_summary += "### CV寄与率が高い媒体\n\n"
            for media in important_media['high_cv_contribution'][:3]:  # 上位3件に制限
                auto_analysis_summary += f"- **{media['media_name']}**: CV寄与率 {media['contribution_rate']:.1f}%, CV変化 {media['cv_change']:.1f}件\n"
        
        # CPA変化率が大きい媒体
        if important_media['high_cpa_change']:
            auto_analysis_summary += "\n### CPA変化率が大きい媒体\n\n"
            for media in important_media['high_cpa_change'][:3]:  # 上位3件に制限
                auto_analysis_summary += f"- **{media['media_name']}**: CPA変化率 {media['cpa_change_rate']:.1f}%, 主要因: {media['main_factor']}\n"
        
        # 成功パターンの媒体
        if important_media['success_pattern']:
            auto_analysis_summary += "\n### 成功パターンの媒体\n\n"
            for media in important_media['success_pattern'][:3]:  # 上位3件に制限
                auto_analysis_summary += f"- **{media['media_name']}**: CV変化 {media['cv_change']:.1f}件, CPA変化 {media['cpa_change']:.0f}円\n"
        
        # 課題パターンの媒体
        if important_media['issue_pattern']:
            auto_analysis_summary += "\n### 課題パターンの媒体\n\n"
            for media in important_media['issue_pattern'][:3]:  # 上位3件に制限
                auto_analysis_summary += f"- **{media['media_name']}**: CV変化 {media['cv_change']:.1f}件, CPA変化 {media['cpa_change']:.0f}円\n"
    
    # プロンプトの作成
    prompt = f"""# 広告パフォーマンス分析

## 全体サマリー
{summary_table}

## CV増減の寄与度ランキング（上位5媒体）
{cv_table}

## CPA変化要因分析（上位5媒体）
{cpa_table}

## 媒体グループ・パターン分析
{pattern_summary}
{auto_analysis_summary}
---

上記の広告パフォーマンスデータを分析し、以下の内容を含むレポートを生成してください：

1. 全体サマリー
   - 主要指標の変化状況
   - 日数差を考慮した場合の評価

2. CV増減の寄与度分析
   - CV増減に大きく影響した媒体の特定
   - 寄与率の高い媒体の動向分析

3. CPA変化要因分析
   - 主なCPA変化の要因（CVRまたはCPC）
   - CPCが主要因の場合はCPMとCTRの影響も説明
   - 変化の詳細説明

4. 戦略的変化の解釈
   - 媒体タイプ間の予算シフト分析
   - 効率と規模のバランス変化
   - 新規導入または縮小された媒体の評価

5. 重点的に見るべき問題点と機会
   - 優先的に対応すべき3つの課題
   - 活用すべき3つの好機
   - 各項目に対する推奨アクション

以下の注意点を考慮してください：
- 単純な数値比較だけでなく、背景にある戦略的意図を考慮
- 日数の違いがある場合は、日平均値での比較も検討
- CV数が極端に少ない媒体（5件未満等）はCPA等の変動が大きくなるため解釈に注意
- 新規追加や停止された媒体については、特別に言及
- 季節性や市場環境変化など、外部要因の可能性も考慮
"""
    
    return prompt

# ChatGPTを使用した分析結果の解釈
def interpret_analysis_with_chatgpt(analysis_result, api_key, model="gpt-3.5-turbo-16k"):
    """
    分析結果をChatGPT APIを使用して解釈する
    
    Parameters:
    analysis_result (dict): 分析結果
    api_key (str): OpenAI API Key
    model (str): 使用するモデル
    
    Returns:
    dict: ChatGPTの解釈結果
    """
    if not api_key:
        st.warning("OpenAI API Keyが設定されていません。分析結果の解釈を行うにはAPI Keyを設定してください。")
        return None
    
    try:
        # OpenAI APIの設定
        openai.api_key = api_key
        
        # 分析結果の整形
        prompt_data = format_prompt_data(analysis_result)
        
        # プロンプトの作成
        prompt = create_analysis_prompt(prompt_data)
        
        # ChatGPT APIにリクエスト
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは広告データ分析の専門家です。以下の指示に従って広告パフォーマンスデータを分析し、洞察と推奨事項を提供してください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=4000
        )
        
        # レスポンスの取得
        interpretation = response.choices[0].message.content
        
        return {
            'interpretation': interpretation,
            'prompt': prompt
        }
    
    except Exception as e:
        st.error(f"ChatGPT APIでエラーが発生しました: {str(e)}")
        return None

# [変更4] 数値フォーマット関数
def format_metrics(df, integer_cols=['Impressions', 'CPM', 'Clicks', 'Cost', 'CPC', 'CPA'],
                  decimal_cols=['Conversions', 'CTR', 'CVR']):
    """
    データフレームの数値を指定されたフォーマットに整形する
    
    Parameters:
    df (DataFrame): フォーマットするデータフレーム
    integer_cols (list): 整数表示する列
    decimal_cols (list): 小数第一位まで表示する列
    
    Returns:
    DataFrame: フォーマット済みのデータフレーム
    """
    df_formatted = df.copy()
    
    for col in integer_cols:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
    
    for col in decimal_cols:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "-")
    
    return df_formatted

# メイン機能の実装
def main():
    # アプリケーションタイトル
    st.title("広告パフォーマンス分析システム")
    
    # サイドバーの設定
    st.sidebar.title("分析設定")
    
    # Googleスプレッドシートの接続情報
    default_url = "https://docs.google.com/spreadsheets/d/1dKjwuk5kOL1bK2KUwZuPMF4sG_aGZt1x4tK6_ooYZv0/edit?gid=1161532243#gid=1161532243"
    default_sheet = "基本データ"
    
    # データ読み込み設定
    with st.sidebar.expander("データソース設定", expanded=True):
        spreadsheet_url = st.text_input("スプレッドシートURL", value=default_url)
        sheet_name = st.text_input("シート名", value=default_sheet)
        
        if st.button("データを読み込む"):
            df = load_data_from_gsheet(spreadsheet_url, sheet_name)
            if df is not None:
                # 派生指標の計算
                df = calculate_derived_metrics(df)
                st.session_state['data'] = df
                st.success("データを読み込みました")
            else:
                st.error("データの読み込みに失敗しました")
    
    # [変更5] 自動分析設定
    with st.sidebar.expander("自動分析設定", expanded=False):
        auto_analysis_mode = st.checkbox("自動分析を有効にする", value=True)
        analysis_depth = st.select_slider(
            "分析の深さ",
            options=["媒体レベル", "キャンペーンレベル", "広告グループレベル"],
            value="キャンペーンレベル"
        )
        cv_threshold = st.slider("CV寄与率閾値 (%)", min_value=10, max_value=50, value=30, step=5)
        cpa_threshold = st.slider("CPA変化率閾値 (%)", min_value=10, max_value=50, value=20, step=5)
    
    # データが読み込まれているか確認
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.info("サイドバーからデータを読み込んでください")
        return
    
    # データの概要を表示
    df = st.session_state['data']
    
    # タブの設定
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["データ概要", "期間比較分析", "自動分析", "レポート出力", "分析手法の説明"])
    
    # タブ1: データ概要
    with tab1:
        st.header("データ概要")
        
        # データの基本情報
        st.subheader("基本情報")
        st.write(f"行数: {len(df)}, 列数: {len(df.columns)}")
        
        # 日付範囲の表示
        if 'Date' in df.columns:
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            st.write(f"データ期間: {min_date.strftime('%Y-%m-%d')} から {max_date.strftime('%Y-%m-%d')} ({(max_date - min_date).days + 1} 日間)")
        
        # 媒体数の表示
        if 'ServiceNameJA' in df.columns:
            media_count = df['ServiceNameJA'].nunique()
            st.write(f"媒体数: {media_count}")
        
        # サンプルデータの表示
        st.subheader("サンプルデータ")
        
        # [変更3] 指標の表示順序を変更
        column_order = []
        
        # 識別子列（媒体名、キャンペーン名、など）
        id_columns = ['Date', 'ServiceNameJA', 'CampaignName', 'AdgroupName']
        for col in id_columns:
            if col in df.columns:
                column_order.append(col)
        
        # 指標列（指定された順序）
        metrics_order = ['Impressions', 'CPM', 'Clicks', 'CTR', 'CPC', 'Cost', 'Conversions', 'CPA', 'CVR']
        for col in metrics_order:
            if col in df.columns:
                column_order.append(col)
        
        # その他の列
        for col in df.columns:
            if col not in column_order:
                column_order.append(col)
        
        # 並べ替えたデータフレームを表示
        sample_df = df[column_order].head(10)
        
        # [変更4] 数値フォーマットの調整
        formatted_sample_df = format_metrics(sample_df)
        st.dataframe(formatted_sample_df)
        
        # 日次データの表示（折れ線グラフ）
        if 'Date' in df.columns:
            st.subheader("日次推移")
            
            # グラフ選択
            metric_option = st.selectbox(
                "指標選択",
                ["Impressions", "CPM", "Clicks", "CTR", "CPC", "Cost", "Conversions", "CPA", "CVR"],
                index=6  # デフォルトはConversions
            )
            
            # 日次集計
            daily_df = df.groupby('Date')[metric_option].sum().reset_index()
            
            # グラフ作成
            fig = px.line(
                daily_df,
                x='Date',
                y=metric_option,
                title=f"{metric_option}の日次推移",
                labels={'Date': '日付', metric_option: metric_option}
            )
            
            # Yラベルのフォーマット調整
            if metric_option in ['CTR', 'CVR']:
                fig.update_layout(yaxis_ticksuffix='%')
            elif metric_option in ['Cost', 'CPC', 'CPA', 'CPM']:
                fig.update_layout(yaxis_ticksuffix='円')
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 媒体別データの表示（円グラフ）
        if 'ServiceNameJA' in df.columns:
            st.subheader("媒体別データ")
            
            # グラフ選択
            media_metric = st.selectbox(
                "指標選択（媒体別）",
                ["Impressions", "CPM", "Clicks", "CTR", "CPC", "Cost", "Conversions", "CPA", "CVR"],
                index=6  # デフォルトはConversions
            )
            
            # 媒体別集計
            if media_metric in ['CTR', 'CVR', 'CPC', 'CPA', 'CPM']:
                # 平均値を計算する必要がある指標
                if media_metric == 'CTR':
                    media_df = df.groupby('ServiceNameJA').apply(
                        lambda x: (x['Clicks'].sum() / x['Impressions'].sum()) * 100 if x['Impressions'].sum() > 0 else 0
                    ).reset_index(name=media_metric)
                elif media_metric == 'CVR':
                    media_df = df.groupby('ServiceNameJA').apply(
                        lambda x: (x['Conversions'].sum() / x['Clicks'].sum()) * 100 if x['Clicks'].sum() > 0 else 0
                    ).reset_index(name=media_metric)
                elif media_metric == 'CPC':
                    media_df = df.groupby('ServiceNameJA').apply(
                        lambda x: x['Cost'].sum() / x['Clicks'].sum() if x['Clicks'].sum() > 0 else 0
                    ).reset_index(name=media_metric)
                elif media_metric == 'CPA':
                    media_df = df.groupby('ServiceNameJA').apply(
                        lambda x: x['Cost'].sum() / x['Conversions'].sum() if x['Conversions'].sum() > 0 else 0
                    ).reset_index(name=media_metric)
                elif media_metric == 'CPM':
                    media_df = df.groupby('ServiceNameJA').apply(
                        lambda x: (x['Cost'].sum() / x['Impressions'].sum()) * 1000 if x['Impressions'].sum() > 0 else 0
                    ).reset_index(name=media_metric)
            else:
                # 合計値を使用する指標
                media_df = df.groupby('ServiceNameJA')[media_metric].sum().reset_index()
            
            # 上位10媒体に絞る
            media_df = media_df.sort_values(media_metric, ascending=False).head(10)
            
            # グラフ作成
            fig = px.pie(
                media_df,
                values=media_metric,
                names='ServiceNameJA',
                title=f"媒体別 {media_metric} 構成比（上位10媒体）"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
    
    # タブ2: 期間比較分析
    with tab2:
        st.header("期間比較分析")
        
        # 期間設定
        st.subheader("期間設定")
        
        # 分析単位の選択
        analysis_unit = st.radio(
            "分析単位",
            ["月次", "週次", "カスタム"],
            horizontal=True
        )
        
        # 日付範囲の取得
        if 'Date' in df.columns:
            min_date = df['Date'].min()
            max_date = df['Date'].max()
        else:
            min_date = datetime.now() - timedelta(days=60)
            max_date = datetime.now()
        
        # 期間設定UI
        if analysis_unit == "月次":
            # 月次分析用の期間設定
            months = []
            current_date = min_date
            while current_date <= max_date:
                months.append(current_date.strftime("%Y-%m"))
                # 次の月の初日に移動
                if current_date.month == 12:
                    current_date = datetime(current_date.year + 1, 1, 1)
                else:
                    current_date = datetime(current_date.year, current_date.month + 1, 1)
            
            # 重複を削除し、ソート
            months = sorted(list(set(months)))
            
            if len(months) < 2:
                st.warning("月次分析を行うには、少なくとも2つの異なる月のデータが必要です")
                return
            
            # 月の選択
            col1, col2 = st.columns(2)
            with col1:
                previous_month = st.selectbox("前期（月）", months[:-1], index=len(months)-2)
            with col2:
                current_month = st.selectbox("当期（月）", months[1:], index=len(months)-2)
            
            # 選択された月の開始日と終了日を計算
            previous_year, previous_month_num = map(int, previous_month.split('-'))
            previous_start = datetime(previous_year, previous_month_num, 1)
            if previous_month_num == 12:
                previous_end = datetime(previous_year + 1, 1, 1) - timedelta(days=1)
            else:
                previous_end = datetime(previous_year, previous_month_num + 1, 1) - timedelta(days=1)
            
            current_year, current_month_num = map(int, current_month.split('-'))
            current_start = datetime(current_year, current_month_num, 1)
            if current_month_num == 12:
                current_end = datetime(current_year + 1, 1, 1) - timedelta(days=1)
            else:
                current_end = datetime(current_year, current_month_num + 1, 1) - timedelta(days=1)
        
        elif analysis_unit == "週次":
            # 週次分析用の期間設定
            # 週の開始日を月曜日とする
            start_of_week = min_date - timedelta(days=min_date.weekday())
            
            weeks = []
            current_date = start_of_week
            while current_date <= max_date:
                week_end = current_date + timedelta(days=6)
                weeks.append((current_date, week_end))
                current_date = current_date + timedelta(days=7)
            
            # 週の選択肢を作成
            week_options = [f"{week[0].strftime('%m/%d')}～{week[1].strftime('%m/%d')}" for week in weeks]
            
            if len(week_options) < 2:
                st.warning("週次分析を行うには、少なくとも2週間のデータが必要です")
                return
            
            # 週の選択
            col1, col2 = st.columns(2)
            with col1:
                previous_week_idx = st.selectbox("前週", week_options[:-1], index=len(week_options)-2)
            with col2:
                current_week_idx = st.selectbox("当週", week_options[1:], index=len(week_options)-2)
            
            # 選択された週のインデックスを取得
            previous_week_idx = week_options.index(previous_week_idx)
            current_week_idx = week_options.index(current_week_idx)
            
            # 日付範囲を設定
            previous_start, previous_end = weeks[previous_week_idx]
            current_start, current_end = weeks[current_week_idx]
        
        else:  # カスタム
            # カスタム期間設定
            col1, col2 = st.columns(2)
            with col1:
                st.write("前期")
                previous_start = st.date_input(
                    "開始日",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                previous_end = st.date_input(
                    "終了日",
                    value=min_date + timedelta(days=6),
                    min_value=previous_start,
                    max_value=max_date
                )
            
            with col2:
                st.write("当期")
                current_start = st.date_input(
                    "開始日",
                    value=previous_end + timedelta(days=1),
                    min_value=min_date,
                    max_value=max_date
                )
                current_end = st.date_input(
                    "終了日",
                    value=min(current_start + timedelta(days=6), max_date.date()),
                    min_value=current_start,
                    max_value=max_date.date()
                )
        
        # 選択された期間を表示
        st.write(f"前期: {previous_start.strftime('%Y-%m-%d')} から {previous_end.strftime('%Y-%m-%d')} ({(previous_end - previous_start).days + 1} 日間)")
        st.write(f"当期: {current_start.strftime('%Y-%m-%d')} から {current_end.strftime('%Y-%m-%d')} ({(current_end - current_start).days + 1} 日間)")
        
        # 分析粒度の選択
        st.subheader("分析粒度")
        
        # 利用可能なグループ化カラムの確認
        group_columns = ['ServiceNameJA']
        if 'CampaignName' in df.columns:
            group_columns.append('CampaignName')
        if 'AdgroupName' in df.columns and df['AdgroupName'].notna().any():
            group_columns.append('AdgroupName')
        
        analysis_granularity = st.selectbox(
            "分析粒度",
            group_columns,
            index=0
        )
        
        # 分析実行ボタン
        if st.button("分析実行", key="manual_analysis"):
            with st.spinner("分析を実行中..."):
                # 前期・当期のデータを抽出
                previous_df = filter_data_by_date(df, previous_start, previous_end)
                current_df = filter_data_by_date(df, current_start, current_end)
                
                # 分析粒度に基づいてグループ化カラムを設定
                if analysis_granularity == 'ServiceNameJA':
                    group_by_cols = ['ServiceNameJA']
                elif analysis_granularity == 'CampaignName':
                    group_by_cols = ['ServiceNameJA', 'CampaignName']
                elif analysis_granularity == 'AdgroupName':
                    group_by_cols = ['ServiceNameJA', 'CampaignName', 'AdgroupName']
                
                # 期間比較分析を実行
                analysis_result = compare_periods(current_df, previous_df, group_by_cols)
                
                if analysis_result:
                    st.session_state['analysis_result'] = analysis_result
                    st.session_state['previous_df'] = previous_df
                    st.session_state['current_df'] = current_df
                    st.session_state['group_by_cols'] = group_by_cols
                    
                    # 自動分析モードが有効な場合
                    if auto_analysis_mode:
                        with st.spinner("自動分析を実行中..."):
                            auto_analysis_result = auto_analyze(
                                analysis_result,
                                df,
                                previous_df,
                                current_df,
                                group_by_cols
                            )
                            st.session_state['auto_analysis_result'] = auto_analysis_result
                    
                    # ChatGPTによる分析結果の解釈（API Keyが設定されている場合）
                    if openai_api_key:
                        with st.spinner("ChatGPTによる分析結果の解釈中..."):
                            try:
                                interpretation = interpret_analysis_with_chatgpt(analysis_result, openai_api_key)
                                if interpretation:
                                    st.session_state['interpretation'] = interpretation
                                    st.success("分析完了！「レポート出力」タブで結果を確認できます")
                                else:
                                    st.warning("ChatGPTによる分析結果の解釈に失敗しました")
                            except Exception as e:
                                st.error(f"ChatGPT APIとの通信中にエラーが発生しました: {str(e)}")
                    else:
                        st.warning("OpenAI API Keyが設定されていないため、レポート生成はスキップされます")
                        st.success("分析完了！「レポート出力」タブで結果を確認できます")
                else:
                    st.error("分析に失敗しました")
        
        # 分析結果があれば表示
        if 'analysis_result' in st.session_state and st.session_state['analysis_result']:
            result = st.session_state['analysis_result']
            
            st.subheader("分析結果プレビュー")
            
            # 1. 全体サマリー
            st.write("#### 全体サマリー")
            col1, col2, col3 = st.columns(3)
            
            # 前期・当期の合計値
            current_total = result['current_total']
            previous_total = result['previous_total']
            
            # 各指標の変化率を計算
            cv_change = ((current_total['Conversions'] - previous_total['Conversions']) / previous_total['Conversions']) * 100 if previous_total['Conversions'] != 0 else 0
            cost_change = ((current_total['Cost'] - previous_total['Cost']) / previous_total['Cost']) * 100 if previous_total['Cost'] != 0 else 0
            
            # CPAの計算
            previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] != 0 else 0
            current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] != 0 else 0
            cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else 0
            
            with col1:
                st.metric(
                    "コンバージョン",
                    f"{current_total['Conversions']:.1f}",
                    f"{cv_change:.1f}%"
                )
            
            with col2:
                st.metric(
                    "コスト",
                    f"{current_total['Cost']:,.0f}円",
                    f"{cost_change:.1f}%"
                )
            
            with col3:
                st.metric(
                    "CPA",
                    f"{current_cpa:.0f}円",
                    f"{cpa_change:.1f}%",
                    delta_color="inverse" # CPAは下がる方がプラス表示
                )
            
            # 2. CV増減の寄与度ランキング
            st.write("#### CV増減の寄与度ランキング")
            
            cv_contribution = result['cv_contribution'].head(5)
            
            # [変更4] 数値フォーマットの調整
            cv_contribution_formatted = format_metrics(
                cv_contribution,
                integer_cols=[],
                decimal_cols=['previous_cv', 'current_cv', 'cv_change', 'contribution_rate']
            )
            
            st.dataframe(cv_contribution_formatted)
            
            # 3. CPA変化要因分析
            st.write("#### CPA変化要因分析")
            
            cpa_factors = result['cpa_change_factors'].head(5)
            
            # [変更1] CPA変化要因の詳細情報と、[変更4] 数値フォーマットの調整
            cpa_details = cpa_factors[['ServiceNameJA', 'previous_cpa', 'current_cpa', 'cpa_change_rate', 'main_factor', 'secondary_factor', 'description']]
            cpa_details_formatted = format_metrics(
                cpa_details,
                integer_cols=['previous_cpa', 'current_cpa'],
                decimal_cols=['cpa_change_rate']
            )
            
            st.dataframe(cpa_details_formatted)
            
            # 4. 媒体グループ・パターン分析
            st.write("#### 媒体グループ・パターン分析")
            
            patterns = result['media_patterns']['pattern_df']
            pattern_counts = patterns['pattern'].value_counts()
            
            # パターン分布の円グラフ
            fig = px.pie(
                pattern_counts,
                values=pattern_counts.values,
                names=pattern_counts.index,
                title="媒体パターン分布",
                labels={
                    'index': 'パターン',
                    'value': '媒体数'
                }
            )
            
            # パターン名を日本語に変換
            pattern_names = {
                'success': '成功パターン（CV増加かつCPA改善）',
                'growth': '成長重視パターン（CV増加かつCPA悪化）',
                'efficiency': '効率重視パターン（CV減少かつCPA改善）',
                'issue': '課題パターン（CV減少かつCPA悪化）'
            }
            
            fig.update_traces(
                labels=[pattern_names.get(p, p) for p in pattern_counts.index],
                textinfo='percent+label'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # [変更5] タブ3: 自動分析
    with tab3:
        st.header("自動分析")
        
        if 'auto_analysis_result' not in st.session_state or not st.session_state['auto_analysis_result']:
            st.info("「期間比較分析」タブで自動分析を有効にして分析を実行してください")
            return
        
        auto_result = st.session_state['auto_analysis_result']
        important_media = auto_result['important_media']
        
        # 重要媒体の表示
        st.subheader("重要な媒体")
        
        # CV寄与率が高い媒体
        if important_media['high_cv_contribution']:
            st.write("##### CV寄与率が高い媒体")
            
            cv_media_data = []
            for media in important_media['high_cv_contribution']:
                cv_media_data.append({
                    '媒体名': media['media_name'],
                    '前期CV': media['previous_cv'],
                    '当期CV': media['current_cv'],
                    'CV変化': media['cv_change'],
                    '寄与率(%)': media['contribution_rate']
                })
            
            cv_media_df = pd.DataFrame(cv_media_data)
            
            # [変更4] 数値フォーマットの調整
            cv_media_df_formatted = format_metrics(
                cv_media_df,
                integer_cols=[],
                decimal_cols=['前期CV', '当期CV', 'CV変化', '寄与率(%)']
            )
            
            st.dataframe(cv_media_df_formatted)
        
        # CPA変化率が大きい媒体
        if important_media['high_cpa_change']:
            st.write("##### CPA変化率が大きい媒体")
            
            cpa_media_data = []
            for media in important_media['high_cpa_change']:
                cpa_media_data.append({
                    '媒体名': media['media_name'],
                    '前期CPA': media['previous_cpa'],
                    '当期CPA': media['current_cpa'],
                    'CPA変化率(%)': media['cpa_change_rate'],
                    '主要因': media['main_factor'],
                    '副要因': media['secondary_factor'] if media['secondary_factor'] else '-',
                    '説明': media['description']
                })
            
            cpa_media_df = pd.DataFrame(cpa_media_data)
            
            # [変更4] 数値フォーマットの調整
            cpa_media_df_formatted = format_metrics(
                cpa_media_df,
                integer_cols=['前期CPA', '当期CPA'],
                decimal_cols=['CPA変化率(%)']
            )
            
            st.dataframe(cpa_media_df_formatted)
        
        # パターン別の媒体
        st.write("##### パターン別の重要媒体")
        
        pattern_tabs = st.tabs(["成功パターン", "課題パターン", "成長重視パターン", "効率重視パターン"])
        
        with pattern_tabs[0]:
            if important_media['success_pattern']:
                success_data = []
                for media in important_media['success_pattern']:
                    success_data.append({
                        '媒体名': media['media_name'],
                        'CV変化': media['cv_change'],
                        'CPA変化': media['cpa_change']
                    })
                
                success_df = pd.DataFrame(success_data)
                
                # [変更4] 数値フォーマットの調整
                success_df_formatted = format_metrics(
                    success_df,
                    integer_cols=['CPA変化'],
                    decimal_cols=['CV変化']
                )
                
                st.dataframe(success_df_formatted)
            else:
                st.info("成功パターンの媒体はありません")
        
        with pattern_tabs[1]:
            if important_media['issue_pattern']:
                issue_data = []
                for media in important_media['issue_pattern']:
                    issue_data.append({
                        '媒体名': media['media_name'],
                        'CV変化': media['cv_change'],
                        'CPA変化': media['cpa_change']
                    })
                
                issue_df = pd.DataFrame(issue_data)
                
                # [変更4] 数値フォーマットの調整
                issue_df_formatted = format_metrics(
                    issue_df,
                    integer_cols=['CPA変化'],
                    decimal_cols=['CV変化']
                )
                
                st.dataframe(issue_df_formatted)
            else:
                st.info("課題パターンの媒体はありません")
        
        # キャンペーンレベル分析と広告グループレベル分析
        if auto_result['campaign_analysis']:
            st.subheader("キャンペーンレベル分析")
            
            media_selection = st.selectbox(
                "媒体選択",
                list(auto_result['campaign_analysis'].keys())
            )
            
            if media_selection:
                campaign_data = auto_result['campaign_analysis'][media_selection]
                campaign_result = campaign_data['analysis']
                
                st.write(f"##### {media_selection}のキャンペーン分析")
                
                if campaign_data['type'] == 'cv_contribution':
                    st.write(f"CV寄与率: {campaign_data['contribution_rate']:.1f}%")
                elif campaign_data['type'] == 'cpa_change':
                    st.write(f"CPA変化率: {campaign_data['cpa_change_rate']:.1f}%")
                    st.write(f"主要因: {campaign_data['main_factor']}, 副要因: {campaign_data['secondary_factor'] if campaign_data['secondary_factor'] else '-'}")
                
                # キャンペーンレベルのCV寄与度
                st.write("**キャンペーンレベルのCV寄与度**")
                campaign_cv = campaign_result['cv_contribution'].head(5)
                
                # [変更4] 数値フォーマットの調整
                campaign_cv_formatted = format_metrics(
                    campaign_cv[['CampaignName', 'previous_cv', 'current_cv', 'cv_change', 'contribution_rate']],
                    integer_cols=[],
                    decimal_cols=['previous_cv', 'current_cv', 'cv_change', 'contribution_rate']
                )
                
                st.dataframe(campaign_cv_formatted)
                
                # キャンペーンレベルのCPA変化要因
                st.write("**キャンペーンレベルのCPA変化要因**")
                campaign_cpa = campaign_result['cpa_change_factors'].head(5)
                
                # [変更1] CPA変化要因の詳細情報と、[変更4] 数値フォーマットの調整
                campaign_cpa_details = campaign_cpa[['CampaignName', 'previous_cpa', 'current_cpa', 'cpa_change_rate', 'main_factor', 'secondary_factor', 'description']]
                campaign_cpa_formatted = format_metrics(
                    campaign_cpa_details,
                    integer_cols=['previous_cpa', 'current_cpa'],
                    decimal_cols=['cpa_change_rate']
                )
                
                st.dataframe(campaign_cpa_formatted)
        
        # 広告グループレベル分析
        if auto_result['adgroup_analysis']:
            st.subheader("広告グループレベル分析")
            
            campaign_keys = list(auto_result['adgroup_analysis'].keys())
            campaign_selection = st.selectbox(
                "キャンペーン選択",
                campaign_keys
            )
            
            if campaign_selection:
                adgroup_data = auto_result['adgroup_analysis'][campaign_selection]
                adgroup_result = adgroup_data['analysis']
                
                media_name = adgroup_data['media_name']
                campaign_name = adgroup_data['campaign_name']
                
                st.write(f"##### {media_name} / {campaign_name} の広告グループ分析")
                
                # 広告グループレベルのCV寄与度
                st.write("**広告グループレベルのCV寄与度**")
                adgroup_cv = adgroup_result['cv_contribution'].head(5)
                
                # [変更4] 数値フォーマットの調整
                adgroup_cv_formatted = format_metrics(
                    adgroup_cv[['AdgroupName', 'previous_cv', 'current_cv', 'cv_change', 'contribution_rate']],
                    integer_cols=[],
                    decimal_cols=['previous_cv', 'current_cv', 'cv_change', 'contribution_rate']
                )
                
                st.dataframe(adgroup_cv_formatted)
                
                # 広告グループレベルのCPA変化要因
                st.write("**広告グループレベルのCPA変化要因**")
                adgroup_cpa = adgroup_result['cpa_change_factors'].head(5)
                
                # [変更1] CPA変化要因の詳細情報と、[変更4] 数値フォーマットの調整
                adgroup_cpa_details = adgroup_cpa[['AdgroupName', 'previous_cpa', 'current_cpa', 'cpa_change_rate', 'main_factor', 'secondary_factor', 'description']]
                adgroup_cpa_formatted = format_metrics(
                    adgroup_cpa_details,
                    integer_cols=['previous_cpa', 'current_cpa'],
                    decimal_cols=['cpa_change_rate']
                )
                
                st.dataframe(adgroup_cpa_formatted)
    
    # タブ4: レポート出力
    with tab4:
        st.header("レポート出力")
        
        # 分析結果があるか確認
        if 'analysis_result' not in st.session_state or not st.session_state['analysis_result']:
            st.info("「期間比較分析」タブで分析を実行してください")
            return
        
        # ChatGPTによる解釈結果があるか確認
        if 'interpretation' in st.session_state and st.session_state['interpretation']:
            interpretation = st.session_state['interpretation']
            
            # レポート表示
            st.subheader("広告パフォーマンス分析レポート")
            
            # マークダウンでレポートを表示
            st.markdown(interpretation['interpretation'])
            
            # プロンプトを表示するオプション
            with st.expander("分析プロンプト（開発者用）"):
                st.code(interpretation['prompt'], language="markdown")
            
            # レポートダウンロードボタン
            report_text = interpretation['interpretation']
            
            # レポートのテキストを生成
            report_download = f"""# 広告パフォーマンス分析レポート

{report_text}
            """
            
            # ダウンロードボタン
            st.download_button(
                label="レポートをダウンロード",
                data=report_download,
                file_name="広告パフォーマンス分析レポート.md",
                mime="text/markdown"
            )
        else:
            st.warning("ChatGPTによる分析レポートがありません。OpenAI API Keyを設定して分析を再実行してください。")
            
            # 手動でレポートを生成するためのプロンプト表示
            st.subheader("手動レポート生成用プロンプト")
            if 'analysis_result' in st.session_state:
                prompt_data = format_prompt_data(st.session_state['analysis_result'])
                prompt = create_analysis_prompt(prompt_data)
                
                st.code(prompt, language="markdown")
                
                st.info("上記のプロンプトをChatGPTに入力することで、手動でレポートを生成できます。")
    
    # タブ5: 分析手法の説明
    with tab5:
        st.header("分析手法の説明")
        
        st.subheader("分析対象データ")
        st.markdown("""
        * 広告媒体ごとの月次パフォーマンスデータ
        * 主要指標：インプレッション数、クリック数、コスト、CV（コンバージョン）数
        * 比較期間：前期と当期
        """)
        
        st.subheader("分析手順")
        
        # 1. データの基本処理
        with st.expander("1. データの基本処理", expanded=False):
            st.markdown("""
            1. 数値データのクリーニング（カンマや通貨記号の除去）
            2. 以下の指標を計算:
               * CTR (Click-Through Rate) = クリック数 ÷ インプレッション数 × 100%
               * CVR (Conversion Rate) = CV数 ÷ クリック数 × 100%
               * CPC (Cost Per Click) = コスト ÷ クリック数
               * CPA (Cost Per Acquisition) = コスト ÷ CV数
               * CPM (Cost Per Mille) = コスト ÷ インプレッション数 × 1000
            """)
        
        # 2. CV増減の寄与度分析
        with st.expander("2. CV増減の寄与度分析", expanded=False):
            st.markdown("""
            1. 全体のCV変化量を算出:
            ```
            全体CV変化 = 当期合計CV - 前期合計CV
            ```
            
            2. 各媒体のCV変化量を算出:
            ```
            媒体CV変化 = 当期媒体CV - 前期媒体CV
            ```
            
            3. 寄与率（影響度）の計算:
            ```
            寄与率 = 媒体CV変化 ÷ 全体CV変化 × 100%
            ```
            注: 全体CV変化がプラスの場合、正の寄与率は増加に貢献、負の寄与率は相殺する方向に影響していることを意味します。
            
            4. 寄与率の絶対値で降順ソートし、影響の大きい順にランキングします。
            """)
        
        # 3. CPA変化要因分析
        with st.expander("3. CPA変化要因分析", expanded=False):
            st.markdown("""
            各媒体において以下の変化率を計算し、CPA変化の主要因を特定します:
            
            1. CPA変化率 = (当期CPA - 前期CPA) / 前期CPA × 100%
            2. CVR変化率 = (当期CVR - 前期CVR) / 前期CVR × 100%
            3. CPC変化率 = (当期CPC - 前期CPC) / 前期CPC × 100%
            4. CPC内訳:
               * CPM変化率 = (当期CPM - 前期CPM) / 前期CPM × 100%
               * CTR変化率 = (当期CTR - 前期CTR) / 前期CTR × 100%
            5. 主要因判定:
               * |CVR変化率| > |CPC変化率| であれば、CVRが主要因
               * |CPC変化率| > |CVR変化率| であれば、CPCが主要因
               * CPCが主要因の場合、|CPM変化率| > |CTR変化率| であれば、CPMが副要因
               * CPCが主要因の場合、|CTR変化率| > |CPM変化率| であれば、CTRが副要因
            """)
        
        # 4. 媒体グループ・パターン分析
        with st.expander("4. 媒体グループ・パターン分析", expanded=False):
            st.markdown("""
            以下のグループに媒体を分類します:
            
            1. CV増加かつCPA改善の媒体（成功パターン）
            2. CV増加かつCPA悪化の媒体（成長重視パターン）
            3. CV減少かつCPA改善の媒体（効率重視パターン）
            4. CV減少かつCPA悪化の媒体（課題パターン）
            """)
        
        # 5. 自動分析
        with st.expander("5. 自動分析機能", expanded=False):
            st.markdown("""
            1. **重要媒体の自動特定**
               * CV寄与率が閾値以上の媒体
               * CPA変化率が閾値以上の媒体
               * 成功パターンおよび課題パターンの媒体

            2. **階層的分析**
               * トップレベル(媒体)分析
               * 重要媒体のキャンペーンレベル分析
               * 重要キャンペーンの広告グループレベル分析

            3. **分析の深さ設定**
               * 媒体レベルのみ
               * キャンペーンレベルまで
               * 広告グループレベルまで
            """)
        
        st.subheader("アウトプット要件")
        
        # 1. 全体サマリー
        with st.expander("1. 全体サマリー", expanded=False):
            st.markdown("""
            数値は以下のフォーマットで出力されます:
            * インプレッション数、CPM、クリック数、コスト、CPC、CPA: 整数表示
            * コンバージョン数、CTR、CVR: 小数第1位まで表示
            
            表形式で主要指標の比較を示します:
            * 期間（前期/当期）
            * インプレッション数
            * CPM
            * クリック数
            * CTR (%)
            * CPC
            * コスト
            * コンバージョン数
            * CPA
            * CVR (%)
            """)
        
        # 2. CV増減の寄与度ランキング
        with st.expander("2. CV増減の寄与度ランキング", expanded=False):
            st.markdown("""
            表形式で上位5媒体（プラス・マイナス含む）を出力します:
            * 順位
            * 媒体名
            * 前期CV数
            * 当期CV数
            * CV数変化
            * 寄与率（%）
            """)
        
        # 3. CPA変化要因分析
        with st.expander("3. CPA変化要因分析", expanded=False):
            st.markdown("""
            表形式で以下を出力します:
            * 媒体名（CPA変化率降順）
            * 前期CPA
            * 当期CPA
            * CPA変化率（%）
            * 主要因（CVRまたはCPC）
            * 副要因（CPMまたはCTR、CPCが主要因の場合のみ）
            * 変化の詳細説明
            """)
        
        # 4. 戦略的変化の解釈
        with st.expander("4. 戦略的変化の解釈", expanded=False):
            st.markdown("""
            以下の観点から媒体間の予算配分や戦略変更を分析します:
            * 媒体タイプ間の予算シフト（例: リターゲティングへの注力）
            * 効率と規模のバランス変化
            * 新規導入または縮小された媒体の評価
            """)
        
        # 5. 重点的に見るべき問題点と機会
        with st.expander("5. 重点的に見るべき問題点と機会", expanded=False):
            st.markdown("""
            最優先で対応すべき3つの課題と3つの好機を列挙し、各項目に:
            * 問題/機会の簡潔な説明
            * データに基づく根拠
            * 推奨される次のアクション
            """)
        
        st.subheader("分析の注意点")
        st.markdown("""
        1. 単純な数値比較だけでなく、背景にある戦略的意図を考慮します
        2. 日数の違いがある場合は、日平均値での比較も検討します
        3. CV数が極端に少ない媒体（5件未満等）はCPA等の変動が大きくなるため解釈に注意します
        4. 新規追加や停止された媒体については、特別に言及します
        5. 季節性や市場環境変化など、外部要因の可能性も考慮します
        """)
        
        st.subheader("サンプルレポート")
        st.markdown("""
        ```
        # 広告パフォーマンス分析レポート: 2024年6月 vs 7月
        
        ## 全体サマリー
        | 指標 | 6月 | 7月 | 差分 | 変化率 |
        |------|-----|-----|------|--------|
        | インプレッション数 | 5,230,500 | 5,430,200 | +199,700 | +3.8% |
        | CPM | 2,810円 | 2,846円 | +36円 | +1.3% |
        | クリック数 | 142,518 | 148,562 | +6,044 | +4.2% |
        | CTR | 2.7% | 2.7% | +0.0% | +0.4% |
        | CPC | 103円 | 104円 | +1円 | +1.0% |
        | コスト | 14,694,182円 | 15,453,042円 | +758,860円 | +5.2% |
        | コンバージョン数 | 2,066.1 | 2,111.3 | +45.2 | +2.2% |
        | CPA | 7,112円 | 7,320円 | +208円 | +2.9% |
        | CVR | 1.4% | 1.4% | -0.0% | -1.9% |
        
        ## 主要な発見
        1. Yahoo!では非リタゲからリタゲへの明確な予算シフトが実施され、大きな成果
        2. CV数が増加した媒体の多くはCPA悪化を伴うが、効率と規模のバランスが必要
        3. CPA悪化の主因は多くの媒体でCVR低下であり、コンバージョン率の改善が課題
        ```
        """)

# アプリケーションの実行
if __name__ == "__main__":
    main()
