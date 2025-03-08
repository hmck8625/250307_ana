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
import io

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
# filter_data_by_date 関数の修正部分
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


# 指標変化の寄与度計算アルゴリズムの強化

def analyze_cpa_change_attribution(current_df, previous_df, group_by_cols=['ServiceNameJA']):
    """
    CPAの変化要因をより詳細に分解して寄与度を計算する
    
    Parameters:
    current_df (DataFrame): 当期のデータ
    previous_df (DataFrame): 前期のデータ
    group_by_cols (list): グループ化するカラム
    
    Returns:
    DataFrame: 詳細なCPA変化要因分析結果
    """
    # 媒体名などのマッチング用にインデックスを設定
    current_df = current_df.copy().set_index(group_by_cols)
    previous_df = previous_df.copy().set_index(group_by_cols)
    
    # 共通のインデックスを取得
    common_indices = set(current_df.index) & set(previous_df.index)
    
    # データフレームの準備
    factor_data = []
    
    # 各媒体のCPA変化要因を詳細に分析
    for idx in common_indices:
        try:
            # 基本指標の取得
            current_imp = current_df.loc[idx, 'Impressions']
            previous_imp = previous_df.loc[idx, 'Impressions']
            
            current_clicks = current_df.loc[idx, 'Clicks']
            previous_clicks = previous_df.loc[idx, 'Clicks']
            
            current_conversions = current_df.loc[idx, 'Conversions']
            previous_conversions = previous_df.loc[idx, 'Conversions']
            
            current_cost = current_df.loc[idx, 'Cost']
            previous_cost = previous_df.loc[idx, 'Cost']
            
            # CPA, CVR, CPC, CPM, CTRの計算
            current_cpa = current_cost / current_conversions if current_conversions > 0 else float('inf')
            previous_cpa = previous_cost / previous_conversions if previous_conversions > 0 else float('inf')
            
            current_cvr = (current_conversions / current_clicks) * 100 if current_clicks > 0 else 0
            previous_cvr = (previous_conversions / previous_clicks) * 100 if previous_clicks > 0 else 0
            
            current_cpc = current_cost / current_clicks if current_clicks > 0 else 0
            previous_cpc = previous_cost / previous_clicks if previous_clicks > 0 else 0
            
            current_cpm = (current_cost / current_imp) * 1000 if current_imp > 0 else 0
            previous_cpm = (previous_cost / previous_imp) * 1000 if previous_imp > 0 else 0
            
            current_ctr = (current_clicks / current_imp) * 100 if current_imp > 0 else 0
            previous_ctr = (previous_clicks / previous_imp) * 100 if previous_imp > 0 else 0
            
            # 変化量と変化率の計算
            if previous_cpa > 0 and previous_cpa != float('inf'):
                cpa_change = current_cpa - previous_cpa
                cpa_change_rate = (cpa_change / previous_cpa) * 100
            else:
                cpa_change = 0
                cpa_change_rate = 0
                
            if previous_cvr > 0:
                cvr_change = current_cvr - previous_cvr
                cvr_change_rate = (cvr_change / previous_cvr) * 100
            else:
                cvr_change = 0
                cvr_change_rate = 0
                
            if previous_cpc > 0:
                cpc_change = current_cpc - previous_cpc
                cpc_change_rate = (cpc_change / previous_cpc) * 100
            else:
                cpc_change = 0
                cpc_change_rate = 0
                
            if previous_cpm > 0:
                cpm_change = current_cpm - previous_cpm
                cpm_change_rate = (cpm_change / previous_cpm) * 100
            else:
                cpm_change = 0
                cpm_change_rate = 0
                
            if previous_ctr > 0:
                ctr_change = current_ctr - previous_ctr
                ctr_change_rate = (ctr_change / previous_ctr) * 100
            else:
                ctr_change = 0
                ctr_change_rate = 0
            
            # 寄与度の計算 (CPA = Cost / Conversions = CPC / CVR)
            # CVRとCPCが独立変数とすると、それぞれの変化がCPAに与える影響の寄与度を計算できる
            
            # 理論的には: 
            # ln(CPA_current/CPA_previous) = ln(CPC_current/CPC_previous) - ln(CVR_current/CVR_previous)
            # これを線形近似して寄与度を計算
            
            # 数値的な安定性のため、変化率で近似計算
            cpa_cvr_contribution = -cvr_change_rate  # CVRが増加するとCPAは減少するため負の寄与
            cpa_cpc_contribution = cpc_change_rate   # CPCが増加するとCPAも増加するため正の寄与
            
            # CPCの内訳寄与度計算 (CPC = CPM / CTR * 100)
            # ln(CPC_current/CPC_previous) = ln(CPM_current/CPM_previous) - ln(CTR_current/CTR_previous)
            
            # 数値的な安定性のため、変化率で近似計算
            cpc_cpm_contribution = cpm_change_rate   # CPMが増加するとCPCも増加
            cpc_ctr_contribution = -ctr_change_rate  # CTRが増加するとCPCは減少
            
            # 総合寄与度の計算と標準化
            total_attribution = abs(cpa_cvr_contribution) + abs(cpa_cpc_contribution)
            if total_attribution > 0:
                cvr_normalized_contribution = (cpa_cvr_contribution / total_attribution) * 100
                cpc_normalized_contribution = (cpa_cpc_contribution / total_attribution) * 100
            else:
                cvr_normalized_contribution = 0
                cpc_normalized_contribution = 0
            
            # CPC内訳の寄与度計算と標準化
            total_cpc_attribution = abs(cpc_cpm_contribution) + abs(cpc_ctr_contribution)
            if total_cpc_attribution > 0:
                cpm_normalized_contribution = (cpc_cpm_contribution / total_cpc_attribution) * 100
                ctr_normalized_contribution = (cpc_ctr_contribution / total_cpc_attribution) * 100
            else:
                cpm_normalized_contribution = 0
                ctr_normalized_contribution = 0
            
            # 最終的な寄与度の計算
            # CPAの変動に対する各要素の寄与度
            cpa_total_change_rate = cpa_change_rate  # CPAの総変動率
            
            # CPAの変動に対するCVRとCPCの寄与度
            if abs(cpa_total_change_rate) > 0:
                cvr_contribution_to_cpa = (cpa_cvr_contribution / abs(cpa_total_change_rate)) * 100
                cpc_contribution_to_cpa = (cpa_cpc_contribution / abs(cpa_total_change_rate)) * 100
            else:
                cvr_contribution_to_cpa = 0
                cpc_contribution_to_cpa = 0
            
            # CPC変動に対するCPMとCTRの寄与度
            if abs(cpc_change_rate) > 0:
                cpm_contribution_to_cpc = (cpc_cpm_contribution / abs(cpc_change_rate)) * 100
                ctr_contribution_to_cpc = (cpc_ctr_contribution / abs(cpc_change_rate)) * 100
            else:
                cpm_contribution_to_cpc = 0
                ctr_contribution_to_cpc = 0
            
            # 主要因と副要因の判定
            if abs(cvr_contribution_to_cpa) > abs(cpc_contribution_to_cpa):
                main_factor = "CVR"
                if cvr_change_rate > 0:
                    factor_direction = "悪化" if cpa_change_rate > 0 else "改善"
                else:
                    factor_direction = "改善" if cpa_change_rate < 0 else "悪化"
                secondary_factor = None
                description = f"CVRが{cvr_change_rate:.1f}%変化({factor_direction})したことが主要因(寄与度:{abs(cvr_contribution_to_cpa):.1f}%)"
            else:
                main_factor = "CPC"
                if cpc_change_rate > 0:
                    factor_direction = "悪化" if cpa_change_rate > 0 else "改善"
                else:
                    factor_direction = "改善" if cpa_change_rate < 0 else "悪化"
                
                # CPC内訳の副要因判定
                if abs(cpm_contribution_to_cpc) > abs(ctr_contribution_to_cpc):
                    secondary_factor = "CPM"
                    if cpm_change_rate > 0:
                        sub_direction = "上昇"
                    else:
                        sub_direction = "低下"
                    description = f"CPCが{cpc_change_rate:.1f}%変化({factor_direction})し、CPMの{sub_direction}({cpm_change_rate:.1f}%)の影響が大きい(寄与度:{abs(cpm_contribution_to_cpc):.1f}%)"
                else:
                    secondary_factor = "CTR"
                    if ctr_change_rate > 0:
                        sub_direction = "上昇"
                    else:
                        sub_direction = "低下"
                    description = f"CPCが{cpc_change_rate:.1f}%変化({factor_direction})し、CTRの{sub_direction}({ctr_change_rate:.1f}%)の影響が大きい(寄与度:{abs(ctr_contribution_to_cpc):.1f}%)"
            
            # パフォーマンスの変化の説明を生成
            if current_cpa < previous_cpa:
                performance_change = "改善"
            else:
                performance_change = "悪化"
            
            # 詳細な分析結果を辞書に保存
            factor_data.append({
                'index_value': idx,  # index列の名前を変更
                # 基本指標
                'previous_impressions': previous_imp,
                'current_impressions': current_imp,
                'impressions_change_rate': ((current_imp - previous_imp) / previous_imp * 100) if previous_imp > 0 else 0,
                
                'previous_clicks': previous_clicks,
                'current_clicks': current_clicks,
                'clicks_change_rate': ((current_clicks - previous_clicks) / previous_clicks * 100) if previous_clicks > 0 else 0,
                
                'previous_cost': previous_cost,
                'current_cost': current_cost,
                'cost_change_rate': ((current_cost - previous_cost) / previous_cost * 100) if previous_cost > 0 else 0,
                
                'previous_cv': previous_conversions,
                'current_cv': current_conversions,
                'cv_change_rate': ((current_conversions - previous_conversions) / previous_conversions * 100) if previous_conversions > 0 else 0,
                
                # 派生指標
                'previous_cpm': previous_cpm,
                'current_cpm': current_cpm,
                'cpm_change_rate': cpm_change_rate,
                
                'previous_ctr': previous_ctr,
                'current_ctr': current_ctr,
                'ctr_change_rate': ctr_change_rate,
                
                'previous_cpc': previous_cpc,
                'current_cpc': current_cpc,
                'cpc_change_rate': cpc_change_rate,
                
                'previous_cvr': previous_cvr,
                'current_cvr': current_cvr,
                'cvr_change_rate': cvr_change_rate,
                
                'previous_cpa': previous_cpa,
                'current_cpa': current_cpa,
                'cpa_change': cpa_change,
                'cpa_change_rate': cpa_change_rate,
                
                # 寄与度分析
                'cvr_contribution_to_cpa': cvr_contribution_to_cpa,
                'cpc_contribution_to_cpa': cpc_contribution_to_cpa,
                'cpm_contribution_to_cpc': cpm_contribution_to_cpc,
                'ctr_contribution_to_cpc': ctr_contribution_to_cpc,
                
                # 主要因判定
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


# 構造変化分析関数の実装
def analyze_structure_change(current_df, previous_df, group_by_cols=['ServiceNameJA']):
    """
    構造変化（コスト配分など）の分析を行う
    
    Parameters:
    current_df (DataFrame): 当期のデータ
    previous_df (DataFrame): 前期のデータ
    group_by_cols (list): グループ化するカラム
    
    Returns:
    dict: 構造変化分析結果
    """
    # 媒体別の集計
    current_agg = aggregate_data_by_period(current_df, group_by_cols)
    previous_agg = aggregate_data_by_period(previous_df, group_by_cols)
    
    if current_agg is None or previous_agg is None:
        return None
    
    # 全体の合計値を計算
    current_total = current_agg['Cost'].sum()
    previous_total = previous_agg['Cost'].sum()
    current_cv_total = current_agg['Conversions'].sum()
    previous_cv_total = previous_agg['Conversions'].sum()
    
    # 共通の媒体を抽出
    current_set = set(current_agg[group_by_cols[0]].values)
    previous_set = set(previous_agg[group_by_cols[0]].values)
    common_media = current_set & previous_set
    
    # 構造変化データの準備
    structure_data = []
    
    # 変数初期化
    cost_shift_impact_on_cpa = 0
    performance_change_impact_on_cpa = 0
    
    # コスト配分変化とパフォーマンス変化の分析
    for media in common_media:
        # 現在と以前のデータを取得
        current_row = current_agg[current_agg[group_by_cols[0]] == media].iloc[0]
        previous_row = previous_agg[previous_agg[group_by_cols[0]] == media].iloc[0]
        
        # コスト配分比率の計算
        current_cost_ratio = current_row['Cost'] / current_total if current_total > 0 else 0
        previous_cost_ratio = previous_row['Cost'] / previous_total if previous_total > 0 else 0
        cost_ratio_change = current_cost_ratio - previous_cost_ratio
        
        # CPA値の取得
        current_cpa = current_row['CPA'] if 'CPA' in current_row else current_row['Cost'] / current_row['Conversions'] if current_row['Conversions'] > 0 else float('inf')
        previous_cpa = previous_row['CPA'] if 'CPA' in previous_row else previous_row['Cost'] / previous_row['Conversions'] if previous_row['Conversions'] > 0 else float('inf')
        
        # CPA変化率
        cpa_change_rate = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa > 0 and previous_cpa != float('inf') else 0
        
        # CV比率の計算
        current_cv_ratio = current_row['Conversions'] / current_cv_total if current_cv_total > 0 else 0
        previous_cv_ratio = previous_row['Conversions'] / previous_cv_total if previous_cv_total > 0 else 0
        cv_ratio_change = current_cv_ratio - previous_cv_ratio
        
        # コスト配分変化がCPAに与える理論的影響の計算
        # コスト配分が高CPAの媒体に増えると、全体CPAは悪化する
        if previous_cpa != float('inf'):
            if previous_cpa > 0:
                # 配分の変化によるCPA変化への影響
                cost_shift_impact = cost_ratio_change * (previous_cpa / (previous_total / previous_cv_total))
                # 媒体自体のCPA変化影響
                performance_impact = current_cost_ratio * (current_cpa - previous_cpa) / (previous_total / previous_cv_total)
            else:
                cost_shift_impact = 0
                performance_impact = 0
        else:
            cost_shift_impact = 0
            performance_impact = 0
        
        # 集計値に追加
        cost_shift_impact_on_cpa += cost_shift_impact
        performance_change_impact_on_cpa += performance_impact
        
        # データ保存
        structure_data.append({
            'media': media,
            'previous_cost': previous_row['Cost'],
            'current_cost': current_row['Cost'],
            'previous_cost_ratio': previous_cost_ratio * 100,  # パーセント表示
            'current_cost_ratio': current_cost_ratio * 100,    # パーセント表示
            'cost_ratio_change': cost_ratio_change * 100,      # パーセントポイント
            'previous_cv': previous_row['Conversions'],
            'current_cv': current_row['Conversions'],
            'previous_cv_ratio': previous_cv_ratio * 100,      # パーセント表示
            'current_cv_ratio': current_cv_ratio * 100,        # パーセント表示
            'cv_ratio_change': cv_ratio_change * 100,          # パーセントポイント
            'previous_cpa': previous_cpa,
            'current_cpa': current_cpa,
            'cpa_change_rate': cpa_change_rate,
            'cost_shift_impact': cost_shift_impact,
            'performance_impact': performance_impact
        })
    
    # 新規と削除された媒体の分析
    new_media = current_set - previous_set
    deleted_media = previous_set - current_set
    
    for media in new_media:
        current_row = current_agg[current_agg[group_by_cols[0]] == media].iloc[0]
        current_cost_ratio = current_row['Cost'] / current_total if current_total > 0 else 0
        current_cpa = current_row['CPA'] if 'CPA' in current_row else current_row['Cost'] / current_row['Conversions'] if current_row['Conversions'] > 0 else float('inf')
        current_cv_ratio = current_row['Conversions'] / current_cv_total if current_cv_total > 0 else 0
        
        # 新規媒体の影響
        new_media_impact = current_cost_ratio * (current_cpa / (current_total / current_cv_total))
        
        structure_data.append({
            'media': media,
            'previous_cost': 0,
            'current_cost': current_row['Cost'],
            'previous_cost_ratio': 0,
            'current_cost_ratio': current_cost_ratio * 100,
            'cost_ratio_change': current_cost_ratio * 100,
            'previous_cv': 0,
            'current_cv': current_row['Conversions'],
            'previous_cv_ratio': 0,
            'current_cv_ratio': current_cv_ratio * 100,
            'cv_ratio_change': current_cv_ratio * 100,
            'previous_cpa': float('inf'),
            'current_cpa': current_cpa,
            'cpa_change_rate': None,  # 前期データがないので変化率は計算不能
            'cost_shift_impact': new_media_impact,
            'performance_impact': 0,  # 新規のため、パフォーマンス変化の影響はなし
            'status': 'new'
        })
    
    for media in deleted_media:
        previous_row = previous_agg[previous_agg[group_by_cols[0]] == media].iloc[0]
        previous_cost_ratio = previous_row['Cost'] / previous_total if previous_total > 0 else 0
        previous_cpa = previous_row['CPA'] if 'CPA' in previous_row else previous_row['Cost'] / previous_row['Conversions'] if previous_row['Conversions'] > 0 else float('inf')
        previous_cv_ratio = previous_row['Conversions'] / previous_cv_total if previous_cv_total > 0 else 0
        
        # 削除媒体の影響
        deleted_media_impact = -previous_cost_ratio * (previous_cpa / (previous_total / previous_cv_total))
        
        structure_data.append({
            'media': media,
            'previous_cost': previous_row['Cost'],
            'current_cost': 0,
            'previous_cost_ratio': previous_cost_ratio * 100,
            'current_cost_ratio': 0,
            'cost_ratio_change': -previous_cost_ratio * 100,
            'previous_cv': previous_row['Conversions'],
            'current_cv': 0,
            'previous_cv_ratio': previous_cv_ratio * 100,
            'current_cv_ratio': 0,
            'cv_ratio_change': -previous_cv_ratio * 100,
            'previous_cpa': previous_cpa,
            'current_cpa': float('inf'),
            'cpa_change_rate': None,  # 当期データがないので変化率は計算不能
            'cost_shift_impact': deleted_media_impact,
            'performance_impact': 0,  # 削除のため、パフォーマンス変化の影響はなし
            'status': 'deleted'
        })
    
    # データフレームに変換
    structure_df = pd.DataFrame(structure_data)
    
    # コスト変化率でソート
    if 'cost_ratio_change' in structure_df.columns:
        structure_df = structure_df.sort_values('cost_ratio_change', ascending=False)
    
    # 全体サマリーの計算
    previous_overall_cpa = previous_total / previous_cv_total if previous_cv_total > 0 else float('inf')
    current_overall_cpa = current_total / current_cv_total if current_cv_total > 0 else float('inf')
    overall_cpa_change = (current_overall_cpa - previous_overall_cpa) / previous_overall_cpa * 100 if previous_overall_cpa > 0 and previous_overall_cpa != float('inf') else 0
    
    # 構造変化の影響の割合を計算
    total_impact = abs(cost_shift_impact_on_cpa) + abs(performance_change_impact_on_cpa)
    if total_impact > 0:
        cost_shift_percentage = (cost_shift_impact_on_cpa / total_impact) * 100
        performance_change_percentage = (performance_change_impact_on_cpa / total_impact) * 100
    else:
        cost_shift_percentage = 0
        performance_change_percentage = 0
    
    # 結果をまとめる
    result = {
        'structure_df': structure_df,
        'summary': {
            'previous_overall_cpa': previous_overall_cpa,
            'current_overall_cpa': current_overall_cpa,
            'overall_cpa_change': overall_cpa_change,
            'cost_shift_impact': cost_shift_impact_on_cpa,
            'performance_change_impact': performance_change_impact_on_cpa,
            'cost_shift_percentage': cost_shift_percentage,
            'performance_change_percentage': performance_change_percentage
        }
    }
    
    return result


# 階層的変化点特定関数
def identify_hierarchical_change_points(current_df, previous_df, group_by_cols=['ServiceNameJA']):
    """
    階層的に変化点を特定する
    
    Parameters:
    current_df (DataFrame): 当期のデータ
    previous_df (DataFrame): 前期のデータ
    group_by_cols (list): グループ化するカラム
    
    Returns:
    dict: 階層別の変化点
    """
    # 主要指標の設定
    key_metrics = ['Impressions', 'Clicks', 'Cost', 'Conversions', 'CTR', 'CVR', 'CPC', 'CPA']
    
    # 階層レベルの定義
    hierarchy_levels = []
    
    if 'ServiceNameJA' in current_df.columns:
        hierarchy_levels.append(['ServiceNameJA'])
    
    if 'ServiceNameJA' in current_df.columns and 'CampaignName' in current_df.columns:
        hierarchy_levels.append(['ServiceNameJA', 'CampaignName'])
    
    if 'ServiceNameJA' in current_df.columns and 'CampaignName' in current_df.columns and 'AdgroupName' in current_df.columns:
        hierarchy_levels.append(['ServiceNameJA', 'CampaignName', 'AdgroupName'])
    
    # 各階層レベルでの変化点を保存する辞書
    change_points = {}
    
    # 階層ごとの分析
    for level in hierarchy_levels:
        level_name = "_".join(level)
        
        # 当期・前期のデータを集計
        current_agg = aggregate_data_by_period(current_df, level)
        previous_agg = aggregate_data_by_period(previous_df, level)
        
        if current_agg is None or previous_agg is None:
            continue
        
        # マージして比較
        merged = pd.merge(previous_agg, current_agg, on=level, how='outer', suffixes=('_prev', '_curr'))
        
        # 欠損値を埋める
        for metric in key_metrics:
            prev_col = f"{metric}_prev"
            curr_col = f"{metric}_curr"
            
            if prev_col in merged.columns and curr_col in merged.columns:
                merged[prev_col] = merged[prev_col].fillna(0)
                merged[curr_col] = merged[curr_col].fillna(0)
                
                # 変化率の計算
                change_col = f"{metric}_change_rate"
                merged[change_col] = merged.apply(
                    lambda row: ((row[curr_col] - row[prev_col]) / row[prev_col] * 100) if row[prev_col] > 0 else float('inf') if row[curr_col] > 0 else 0, 
                    axis=1
                )
                
                # 極端な変化を検出
                threshold = 20  # 20%以上の変化を重要とする
                merged[f"{metric}_significant_change"] = merged[change_col].abs() > threshold
        
        # 重要な変化点を抽出
        significant_changes = merged[merged[[f"{metric}_significant_change" for metric in key_metrics if f"{metric}_significant_change" in merged.columns]].any(axis=1)]
        
        # CV変化とCPA変化に注目
        if 'Conversions_significant_change' in significant_changes.columns and 'CPA_significant_change' in significant_changes.columns:
            cv_changes = significant_changes[significant_changes['Conversions_significant_change']]
            cpa_changes = significant_changes[significant_changes['CPA_significant_change']]
            
            # 変化点情報を整理
            change_points[level_name] = {
                'all_changes': significant_changes,
                'cv_changes': cv_changes,
                'cpa_changes': cpa_changes,
                'level_columns': level
            }
    
    # 階層間の関連性分析
    if len(hierarchy_levels) > 1:
        # 上位レベルの変化点と下位レベルの変化点の関連付け
        for i in range(len(hierarchy_levels) - 1):
            upper_level = "_".join(hierarchy_levels[i])
            lower_level = "_".join(hierarchy_levels[i+1])
            
            if upper_level in change_points and lower_level in change_points:
                upper_changes = change_points[upper_level]['all_changes']
                lower_changes = change_points[lower_level]['all_changes']
                
                # 上位レベルの各変化点について、関連する下位レベルの変化点を特定
                for idx, upper_row in upper_changes.iterrows():
                    # 上位レベルの識別子を取得
                    upper_id = tuple(upper_row[col] for col in hierarchy_levels[i])
                    
                    # 下位レベルのデータをフィルタリング
                    related_lower_changes = lower_changes[lower_changes.apply(
                        lambda row: tuple(row[col] for col in hierarchy_levels[i]) == upper_id, 
                        axis=1
                    )]
                    
                    # 関連性情報を保存
                    if not related_lower_changes.empty:
                        if 'related_lower_changes' not in change_points[upper_level]:
                            change_points[upper_level]['related_lower_changes'] = {}
                        
                        change_points[upper_level]['related_lower_changes'][upper_id] = related_lower_changes
    
    return change_points


# 変化要因の深掘り分析機能
def deep_dive_analysis(current_df, previous_df, group_by_cols=['ServiceNameJA'], target_entity=None):
    """
    特定の媒体やキャンペーンに対する深掘り分析を行う
    
    Parameters:
    current_df (DataFrame): 当期のデータ
    previous_df (DataFrame): 前期のデータ
    group_by_cols (list): グループ化するカラム
    target_entity (str or tuple): 分析対象の媒体名やキャンペーン名
    
    Returns:
    dict: 深掘り分析結果
    """
    if target_entity is None:
        return None
    
    # 分析対象のデータをフィルタリング
    if isinstance(target_entity, str):
        current_filtered = current_df[current_df[group_by_cols[0]] == target_entity]
        previous_filtered = previous_df[previous_df[group_by_cols[0]] == target_entity]
    elif isinstance(target_entity, tuple):
        # 複数のカラムでのフィルタリング
        if len(target_entity) != len(group_by_cols):
            return None
        
        current_filtered = current_df.copy()
        previous_filtered = previous_df.copy()
        
        for i, col in enumerate(group_by_cols):
            current_filtered = current_filtered[current_filtered[col] == target_entity[i]]
            previous_filtered = previous_filtered[previous_filtered[col] == target_entity[i]]
    else:
        return None
    
    # より詳細なレベルで分析
    next_level = None
    
    if group_by_cols == ['ServiceNameJA'] and 'CampaignName' in current_filtered.columns:
        next_level = ['ServiceNameJA', 'CampaignName']
    elif group_by_cols == ['ServiceNameJA', 'CampaignName'] and 'AdgroupName' in current_filtered.columns:
        next_level = ['ServiceNameJA', 'CampaignName', 'AdgroupName']
    
    if next_level:
        # 次のレベルでの分析
        detailed_analysis = compare_periods(current_filtered, previous_filtered, next_level)
        structure_analysis = analyze_structure_change(current_filtered, previous_filtered, next_level[1:])
        
        # 時系列分析
        time_series_analysis = None
        if 'Date' in current_filtered.columns and 'Date' in previous_filtered.columns:
            # 日ごとの推移を分析
            current_daily = current_filtered.groupby('Date').agg({
                'Impressions': 'sum', 
                'Clicks': 'sum', 
                'Cost': 'sum', 
                'Conversions': 'sum'
            }).reset_index()
            
            previous_daily = previous_filtered.groupby('Date').agg({
                'Impressions': 'sum', 
                'Clicks': 'sum', 
                'Cost': 'sum', 
                'Conversions': 'sum'
            }).reset_index()
            
            # 派生指標の計算
            current_daily = calculate_derived_metrics(current_daily)
            previous_daily = calculate_derived_metrics(previous_daily)
            
            # 両方のデータセットに共通するFace列を作成 (月日のみの文字列)
            if len(current_daily) > 0 and len(previous_daily) > 0:
                current_daily['face_date'] = current_daily['Date'].dt.strftime('%m-%d')
                previous_daily['face_date'] = previous_daily['Date'].dt.strftime('%m-%d')
                
                # time_series_analysis に結果を保存
                time_series_analysis = {
                    'current_daily': current_daily,
                    'previous_daily': previous_daily
                }
        
        return {
            'detailed_analysis': detailed_analysis,
            'structure_analysis': structure_analysis,
            'time_series_analysis': time_series_analysis,
            'target_entity': target_entity,
            'next_level': next_level
        }
    
    return None


# エグゼクティブサマリー生成関数
def generate_executive_summary(analysis_result, structure_analysis, group_by_cols=['ServiceNameJA'], deep_dive_results=None):
    """
    全体パフォーマンス変化の簡潔なエグゼクティブサマリーを生成する
    
    Parameters:
    analysis_result (dict): 基本分析結果
    structure_analysis (dict): 構造変化分析結果
    group_by_cols (list): グループ化するカラム
    deep_dive_results (dict): 深掘り分析結果
    
    Returns:
    dict: エグゼクティブサマリー
    """
    # 基本指標の変化を取得
    current_total = analysis_result['current_total']
    previous_total = analysis_result['previous_total']
    
    # CV変化率
    cv_change = ((current_total['Conversions'] - previous_total['Conversions']) / previous_total['Conversions']) * 100 if previous_total['Conversions'] > 0 else 0
    
    # CPA変化率
    previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] > 0 else 0
    current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] > 0 else 0
    cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa > 0 else 0
    
    # CV寄与度ランキングから主要な媒体を特定
    cv_contributors = analysis_result['cv_contribution'].head(3)
    
    # CPA変化要因から主要な要因を特定
    cpa_factors = analysis_result['cpa_change_factors'].head(3)
    
    # 構造変化の要約を取得
    structure_summary = None
    if structure_analysis:
        structure_summary = structure_analysis['summary']
    
    # エグゼクティブサマリーテキストの生成
    if cv_change >= 0 and cpa_change <= 0:
        summary_text = f"好調: CV {cv_change:.1f}%増加、CPA {abs(cpa_change):.1f}%改善。"
    elif cv_change >= 0 and cpa_change > 0:
        summary_text = f"規模拡大: CV {cv_change:.1f}%増加、CPA {cpa_change:.1f}%悪化。"
    elif cv_change < 0 and cpa_change <= 0:
        summary_text = f"効率化: CV {abs(cv_change):.1f}%減少、CPA {abs(cpa_change):.1f}%改善。"
    else:
        summary_text = f"課題あり: CV {abs(cv_change):.1f}%減少、CPA {cpa_change:.1f}%悪化。"
    
    # 主要な影響媒体の特定
    if not cv_contributors.empty:
        top_cv_contributor = cv_contributors.iloc[0]
        media_col = group_by_cols[0] if group_by_cols[0] in top_cv_contributor else top_cv_contributor.index.names[0] if hasattr(top_cv_contributor, 'index') and hasattr(top_cv_contributor.index, 'names') else cv_contributors.columns[0]
        media_name = top_cv_contributor[media_col]
        contribution = top_cv_contributor['contribution_rate']
        if contribution > 0:
            summary_text += f" {media_name}が{contribution:.1f}%の増加に貢献。"
        else:
            summary_text += f" {media_name}が{abs(contribution):.1f}%の減少に影響。"
    
    # 主要なCPA変化要因の追加
    if not cpa_factors.empty:
        top_cpa_factor = cpa_factors.iloc[0]
        summary_text += f" CPA変化の主要因は{top_cpa_factor['main_factor']}。"
    
    # 構造変化の影響の追加
    if structure_summary:
        if abs(structure_summary['cost_shift_percentage']) > abs(structure_summary['performance_change_percentage']):
            summary_text += f" 予算配分変更の影響が{abs(structure_summary['cost_shift_percentage']):.1f}%と大きい。"
        else:
            summary_text += f" 媒体自体のパフォーマンス変化の影響が{abs(structure_summary['performance_change_percentage']):.1f}%と大きい。"
    
    # 主要変化点の特定と説明
    key_change_points = []
    
    # CV変化への影響が大きい媒体
    for i, row in cv_contributors.iterrows():
        if i >= 3:  # 上位3媒体のみ考慮
            break
        
        media_col = group_by_cols[0] if group_by_cols[0] in row else row.index.names[0] if hasattr(row, 'index') and hasattr(row.index, 'names') else cv_contributors.columns[0]
        media_name = row[media_col]
        cv_change = row['cv_change']
        contribution = row['contribution_rate']
        
        if cv_change > 0:
            point = f"{media_name}のCV {cv_change:.1f}件増加（寄与率{contribution:.1f}%）"
        else:
            point = f"{media_name}のCV {abs(cv_change):.1f}件減少（寄与率{contribution:.1f}%）"
        
        key_change_points.append(point)
    
    # CPA変化への影響が大きい媒体
    for i, row in cpa_factors.iterrows():
        if i >= 3:  # 上位3媒体のみ考慮
            break
        
        media_col = group_by_cols[0] if group_by_cols[0] in row else row.index.names[0] if hasattr(row, 'index') and hasattr(row.index, 'names') else cpa_factors.columns[0]
        media_name = row[media_col]
        cpa_change_rate = row['cpa_change_rate']
        main_factor = row['main_factor']
        
        if cpa_change_rate < 0:
            point = f"{media_name}のCPA {abs(cpa_change_rate):.1f}%改善（主因:{main_factor}）"
        else:
            point = f"{media_name}のCPA {cpa_change_rate:.1f}%悪化（主因:{main_factor}）"
        
        key_change_points.append(point)
    
    # 構造変化の影響
    if structure_summary:
        # コスト配分の変化が大きい媒体を特定
        if not structure_analysis['structure_df'].empty:
            structure_df = structure_analysis['structure_df']
            top_shift = structure_df.iloc[0]
            media_name = top_shift['media']
            cost_ratio_change = top_shift['cost_ratio_change']
            
            if cost_ratio_change > 0:
                point = f"{media_name}への予算配分を{cost_ratio_change:.1f}%pt増加"
            else:
                point = f"{media_name}からの予算配分を{abs(cost_ratio_change):.1f}%pt減少"
            
            key_change_points.append(point)
    
    # 深掘り分析結果の追加
    detailed_insights = []
    
    if deep_dive_results:
        for target, result in deep_dive_results.items():
            if result and 'detailed_analysis' in result:
                detailed = result['detailed_analysis']
                if detailed and 'cv_contribution' in detailed:
                    cv_detail = detailed['cv_contribution'].head(1)
                    if not cv_detail.empty:
                        next_level_col = result['next_level'][-1] if 'next_level' in result and result['next_level'] else cv_detail.columns[0]
                        entity_name = cv_detail.iloc[0][next_level_col]
                        cv_change = cv_detail.iloc[0]['cv_change']
                        contribution = cv_detail.iloc[0]['contribution_rate']
                        
                        insight = f"{target}の中で{entity_name}がCV {cv_change:.1f}件変化（寄与率{contribution:.1f}%）"
                        detailed_insights.append(insight)
    
    return {
        'summary_text': summary_text,
        'cv_change': cv_change,
        'cpa_change': cpa_change,
        'key_change_points': key_change_points,
        'detailed_insights': detailed_insights
    }

# 比較期間の分析関数を強化
def compare_periods_enhanced(current_df, previous_df, group_by_cols=['ServiceNameJA']):
    """
    二つの期間のデータを比較して強化された分析結果を返す
    
    Parameters:
    current_df (DataFrame): 当期のデータ
    previous_df (DataFrame): 前期のデータ
    group_by_cols (list): グループ化するカラム
    
    Returns:
    dict: 強化された分析結果を含む辞書
    """
    # 基本分析を実行
    basic_analysis = compare_periods(current_df, previous_df, group_by_cols)
    
    if basic_analysis is None:
        return None
    
    # 詳細なCPA変化要因分析を実行
    cpa_attribution = analyze_cpa_change_attribution(
        basic_analysis['current_agg'], 
        basic_analysis['previous_agg'], 
        group_by_cols
    )
    
    # 構造変化分析を実行
    structure_analysis = analyze_structure_change(current_df, previous_df, group_by_cols)
    
    # 階層的変化点特定を実行
    change_points = identify_hierarchical_change_points(current_df, previous_df, group_by_cols)
    
    # 重要な媒体の深掘り分析
    deep_dive_results = {}
    
    # CV寄与度が高い上位3媒体を深掘り
    top_cv_contributors = basic_analysis['cv_contribution'].head(3)
    for _, row in top_cv_contributors.iterrows():
        media_name = row[group_by_cols[0]]
        deep_dive = deep_dive_analysis(current_df, previous_df, group_by_cols, media_name)
        if deep_dive:
            deep_dive_results[media_name] = deep_dive
    
    # CPA変化が大きい上位3媒体を深掘り
    top_cpa_changers = basic_analysis['cpa_change_factors'].head(3)
    for _, row in top_cpa_changers.iterrows():
        media_name = row[group_by_cols[0]]
        # 既に分析済みならスキップ
        if media_name in deep_dive_results:
            continue
        
        deep_dive = deep_dive_analysis(current_df, previous_df, group_by_cols, media_name)
        if deep_dive:
            deep_dive_results[media_name] = deep_dive
    
    # エグゼクティブサマリーの生成
    executive_summary = generate_executive_summary(
        basic_analysis, 
        structure_analysis, 
        deep_dive_results
    )
    
    # 結果をまとめる
    enhanced_result = {
        **basic_analysis,
        'cpa_attribution': cpa_attribution,
        'structure_analysis': structure_analysis,
        'change_points': change_points,
        'deep_dive_results': deep_dive_results,
        'executive_summary': executive_summary
    }
    
    return enhanced_result


# ビジュアライゼーション関数 - 指標変化のヒートマップ
def create_metrics_heatmap(analysis_result):
    """
    媒体×指標のマトリクスで変化率をヒートマップとして表示
    
    Parameters:
    analysis_result (dict): 分析結果
    
    Returns:
    plotly.graph_objects.Figure: ヒートマップ図
    """
    if 'cpa_attribution' not in analysis_result:
        return None
    
    # データの準備
    attribution = analysis_result['cpa_attribution'].copy()
    
    # 表示する指標を選択
    metrics = [
        'impressions_change_rate', 
        'cpm_change_rate', 
        'clicks_change_rate', 
        'ctr_change_rate', 
        'cpc_change_rate', 
        'cost_change_rate', 
        'cv_change_rate', 
        'cvr_change_rate', 
        'cpa_change_rate'
    ]
    
    # 指標名の表示用マッピング
    metric_names = {
        'impressions_change_rate': 'インプレッション変化率(%)',
        'cpm_change_rate': 'CPM変化率(%)',
        'clicks_change_rate': 'クリック変化率(%)',
        'ctr_change_rate': 'CTR変化率(%)',
        'cpc_change_rate': 'CPC変化率(%)',
        'cost_change_rate': 'コスト変化率(%)',
        'cv_change_rate': 'CV変化率(%)',
        'cvr_change_rate': 'CVR変化率(%)',
        'cpa_change_rate': 'CPA変化率(%)'
    }
    
    # 媒体名の列を特定
    media_col = 'ServiceNameJA' if 'ServiceNameJA' in attribution.columns else attribution.columns[0]
    
    # 上位15媒体に絞る
    top_media = attribution.sort_values('abs_cpa_change', ascending=False).head(15) if 'abs_cpa_change' in attribution.columns else attribution.head(15)
    
    # ヒートマップ用のデータを整形
    heatmap_data = []
    
    for media in top_media[media_col].values:
        media_row = attribution[attribution[media_col] == media].iloc[0]
        row_data = {}
        row_data[media_col] = media
        
        for metric in metrics:
            if metric in media_row:
                # 無限大や異常値の処理
                value = media_row[metric]
                if pd.isna(value) or abs(value) > 1000:
                    row_data[metric_names[metric]] = 0
                else:
                    row_data[metric_names[metric]] = value
            else:
                row_data[metric_names[metric]] = 0
        
        heatmap_data.append(row_data)
    
    # DataFrameに変換
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # ヒートマップのデータ準備
    z_data = []
    y_labels = heatmap_df[media_col].tolist()
    x_labels = [metric_names[m] for m in metrics]
    
    for _, row in heatmap_df.iterrows():
        z_row = [row[metric_names[m]] for m in metrics]
        z_data.append(z_row)
    
    # カラースケールの設定（青: 改善、赤: 悪化）
    # ただし、インプレッション、クリック、CV数は増加が青、減少が赤
    # CPA, CPC, CPM, は減少が青、増加が赤
    # CTR, CVRは増加が青、減少が赤
    
    increase_good = ['インプレッション変化率(%)', 'クリック変化率(%)', 'CV変化率(%)', 'CTR変化率(%)', 'CVR変化率(%)']
    decrease_good = ['CPA変化率(%)', 'CPC変化率(%)', 'CPM変化率(%)']
    
    # カスタムカラースケール用のテキスト配列
    customdata = []
    for i, media in enumerate(y_labels):
        customdata_row = []
        for j, metric in enumerate(x_labels):
            value = z_data[i][j]
            
            # 改善/悪化の判定
            if metric in increase_good:
                status = "改善" if value > 0 else "悪化"
            elif metric in decrease_good:
                status = "改善" if value < 0 else "悪化"
            else:
                status = ""
            
            customdata_row.append(f"{metric}: {value:.1f}% ({status})")
        
        customdata.append(customdata_row)
    
    # ヒートマップ作成
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu_r',  # 赤青反転スケール
        zmid=0,  # 0を中心に色を変える
        text=customdata,
        hoverinfo='text',
        colorbar=dict(
            title="変化率 (%)",
            titleside="right"
        )
    ))
    
    # レイアウト設定
    fig.update_layout(
        title="指標変化率ヒートマップ",
        xaxis=dict(
            title="指標",
            tickangle=-45
        ),
        yaxis=dict(
            title="媒体",
            autorange="reversed"  # 上から順に表示
        ),
        height=600,
        margin=dict(l=100, r=20, t=70, b=100),
    )
    
    return fig


# ビジュアライゼーション関数 - 変化要因の寄与度チャート
def create_contribution_waterfall(analysis_result):
    """
    CPA変化への要因寄与度をウォーターフォールチャートで表示
    
    Parameters:
    analysis_result (dict): 分析結果
    
    Returns:
    plotly.graph_objects.Figure: ウォーターフォールチャート
    """
    if 'cpa_attribution' not in analysis_result:
        return None
    
    # 全体のCPA変化率
    current_total = analysis_result['current_total']
    previous_total = analysis_result['previous_total']
    
    previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] > 0 else 0
    current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] > 0 else 0
    total_cpa_change = current_cpa - previous_cpa
    total_cpa_change_rate = (total_cpa_change / previous_cpa) * 100 if previous_cpa > 0 else 0
    
    # 構造変化分析からの情報取得
    if 'structure_analysis' in analysis_result and analysis_result['structure_analysis']:
        structure_summary = analysis_result['structure_analysis']['summary']
        cost_shift_impact = structure_summary['cost_shift_impact']
        performance_impact = structure_summary['performance_change_impact']
    else:
        cost_shift_impact = 0
        performance_impact = 0
    
    # 詳細な寄与度分析
    attribution = analysis_result['cpa_attribution']
    
    # 主要な媒体のCVR, CPCの影響を計算
    cvr_impacts = []
    cpc_impacts = []
    
    # 総コストの取得
    current_total_cost = current_total['Cost']
    previous_total_cost = previous_total['Cost']
    
    for _, row in attribution.iterrows():
        media_name = row['ServiceNameJA'] if 'ServiceNameJA' in row else 'Unknown'
        
        # コスト比率の計算
        current_cost_ratio = row['current_cost'] / current_total_cost if current_total_cost > 0 else 0
        
        # CVRの影響
        cvr_contribution = row['cvr_contribution_to_cpa'] * current_cost_ratio
        cvr_impacts.append((media_name, cvr_contribution))
        
        # CPCの影響
        cpc_contribution = row['cpc_contribution_to_cpa'] * current_cost_ratio
        cpc_impacts.append((media_name, cpc_contribution))
    
    # 影響の大きい順にソート
    cvr_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    cpc_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # 上位3媒体と「その他」に集約
    top_cvr = cvr_impacts[:3]
    other_cvr = sum(x[1] for x in cvr_impacts[3:])
    
    top_cpc = cpc_impacts[:3]
    other_cpc = sum(x[1] for x in cpc_impacts[3:])
    
    # ウォーターフォールチャート用のデータ準備
    measures = ['relative'] * (len(top_cvr) + len(top_cpc) + 4) + ['total']
    
    # x値（ラベル）の作成
    x_values = ['前期CPA']
    for media, _ in top_cvr:
        x_values.append(f"{media} CVR")
    if other_cvr != 0:
        x_values.append("その他 CVR")
    
    for media, _ in top_cpc:
        x_values.append(f"{media} CPC")
    if other_cpc != 0:
        x_values.append("その他 CPC")
    
    x_values.append("予算配分変更")
    x_values.append("当期CPA")
    
    # y値（影響額）の作成
    y_values = [0]  # 最初の相対値は0
    
    # CVRの影響
    for _, impact in top_cvr:
        y_values.append(impact * previous_cpa / 100)  # パーセント→金額に変換
    
    if other_cvr != 0:
        y_values.append(other_cvr * previous_cpa / 100)
    
    # CPCの影響
    for _, impact in top_cpc:
        y_values.append(impact * previous_cpa / 100)
    
    if other_cpc != 0:
        y_values.append(other_cpc * previous_cpa / 100)
    
    # 予算配分の影響
    y_values.append(cost_shift_impact)
    
    # 最後に当期CPA値
    y_values.append(current_cpa)
    
    # 色の設定
    colors = ['rgba(204, 204, 204, 0.7)']  # 最初のマーカーは灰色
    
    # CVRの色設定
    for _, impact in top_cvr:
        if impact < 0:
            colors.append('rgba(44, 160, 44, 0.7)')  # 緑（改善）
        else:
            colors.append('rgba(214, 39, 40, 0.7)')  # 赤（悪化）
    
    if other_cvr != 0:
        if other_cvr < 0:
            colors.append('rgba(44, 160, 44, 0.7)')
        else:
            colors.append('rgba(214, 39, 40, 0.7)')
    
    # CPCの色設定
    for _, impact in top_cpc:
        if impact < 0:
            colors.append('rgba(31, 119, 180, 0.7)')  # 青（改善）
        else:
            colors.append('rgba(255, 127, 14, 0.7)')  # オレンジ（悪化）
    
    if other_cpc != 0:
        if other_cpc < 0:
            colors.append('rgba(31, 119, 180, 0.7)')
        else:
            colors.append('rgba(255, 127, 14, 0.7)')
    
    # 予算配分の色設定
    if cost_shift_impact < 0:
        colors.append('rgba(148, 103, 189, 0.7)')  # 紫（改善）
    else:
        colors.append('rgba(140, 86, 75, 0.7)')    # 茶（悪化）
    
    # 最終マーカーは灰色
    colors.append('rgba(204, 204, 204, 0.7)')
    
    # ウォーターフォールチャートの作成
    fig = go.Figure(go.Waterfall(
        name="CPA変化要因分析",
        orientation="v",
        measure=measures,
        x=x_values,
        textposition="outside",
        text=[f"{previous_cpa:.0f}円"] + [f"{val:.0f}円" for val in y_values[1:-1]] + [f"{current_cpa:.0f}円"],
        y=[previous_cpa] + y_values[1:],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "rgba(214, 39, 40, 0.7)"}},
        decreasing={"marker": {"color": "rgba(44, 160, 44, 0.7)"}},
        marker={"color": colors}
    ))
    
    # レイアウト設定
    fig.update_layout(
        title=f"CPA変化要因分析 ({previous_cpa:.0f}円 → {current_cpa:.0f}円, {total_cpa_change_rate:.1f}%)",
        showlegend=False,
        xaxis_title="変化要因",
        yaxis_title="CPA (円)",
        height=600
    )
    
    return fig


# ビジュアライゼーション関数 - 構造変化の可視化
def create_structure_change_sankey(analysis_result):
    """
    媒体・キャンペーン間の予算配分変化をサンキーダイアグラムで表示
    
    Parameters:
    analysis_result (dict): 分析結果
    
    Returns:
    plotly.graph_objects.Figure: サンキーダイアグラム
    """
    if 'structure_analysis' not in analysis_result or not analysis_result['structure_analysis']:
        return None
    
    structure_analysis = analysis_result['structure_analysis']
    structure_df = structure_analysis['structure_df']
    
    # 媒体名とコスト情報を抽出
    media_costs = structure_df[['media', 'previous_cost', 'current_cost']].copy()
    
    # 「その他」にまとめる閾値（全体の1%未満）
    current_total = media_costs['current_cost'].sum()
    previous_total = media_costs['previous_cost'].sum()
    current_threshold = current_total * 0.01
    previous_threshold = previous_total * 0.01
    
    # 小さい媒体を「その他」にまとめる
    media_costs['current_group'] = media_costs.apply(
        lambda row: row['media'] if row['current_cost'] >= current_threshold else "その他 (当期)", 
        axis=1
    )
    
    media_costs['previous_group'] = media_costs.apply(
        lambda row: row['media'] if row['previous_cost'] >= previous_threshold else "その他 (前期)", 
        axis=1
    )
    
    # 「その他」グループを集計
    current_others = media_costs[media_costs['current_group'] == "その他 (当期)"]
    current_others_total = current_others['current_cost'].sum()
    
    previous_others = media_costs[media_costs['previous_group'] == "その他 (前期)"]
    previous_others_total = previous_others['previous_cost'].sum()
    
    # 十分大きい媒体のみを抽出
    significant_media = set(media_costs[media_costs['current_cost'] >= current_threshold]['media']) | \
                         set(media_costs[media_costs['previous_cost'] >= previous_threshold]['media'])
    
    # サンキーダイアグラム用のデータ準備
    nodes = []
    node_dict = {}  # mediaとnodeインデックスのマッピング
    
    # ノードの作成
    # 期間ラベル
    nodes.append({"name": "前期"})
    nodes.append({"name": "当期"})
    node_dict["前期"] = 0
    node_dict["当期"] = 1
    
    # 媒体ノード
    for i, media in enumerate(significant_media, 2):
        nodes.append({"name": media})
        node_dict[media] = i
    
    # その他ノード（あれば）
    if previous_others_total > 0:
        nodes.append({"name": "その他 (前期)"})
        node_dict["その他 (前期)"] = len(nodes) - 1
    
    if current_others_total > 0:
        nodes.append({"name": "その他 (当期)"})
        node_dict["その他 (当期)"] = len(nodes) - 1
    
    # リンクの作成
    links = []
    
    # 前期から各媒体へのリンク
    for media in significant_media:
        media_row = media_costs[media_costs['media'] == media].iloc[0]
        if media_row['previous_cost'] > 0:
            links.append({
                "source": node_dict["前期"],
                "target": node_dict[media],
                "value": media_row['previous_cost'],
                "label": f"{media_row['previous_cost']:,.0f}円"
            })
    
    # その他（前期）のリンク
    if previous_others_total > 0:
        links.append({
            "source": node_dict["前期"],
            "target": node_dict["その他 (前期)"],
            "value": previous_others_total,
            "label": f"{previous_others_total:,.0f}円"
        })
    
    # 各媒体から当期へのリンク
    for media in significant_media:
        media_row = media_costs[media_costs['media'] == media].iloc[0]
        if media_row['current_cost'] > 0:
            links.append({
                "source": node_dict[media],
                "target": node_dict["当期"],
                "value": media_row['current_cost'],
                "label": f"{media_row['current_cost']:,.0f}円"
            })
    
    # その他（当期）のリンク
    if current_others_total > 0:
        links.append({
            "source": node_dict["その他 (当期)"],
            "target": node_dict["当期"],
            "value": current_others_total,
            "label": f"{current_others_total:,.0f}円"
        })
    
    # サンキーダイアグラムの作成
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[node["name"] for node in nodes],
            color="blue"
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            value=[link["value"] for link in links],
            label=[link.get("label", "") for link in links]
        )
    )])
    
    # レイアウト設定
    fig.update_layout(
        title_text="媒体間の予算配分変化",
        font_size=12,
        height=600
    )
    
    return fig


# レポート生成関数 - 強化版
def generate_enhanced_report(analysis_result, openai_api_key, model="gpt-3.5-turbo-16k"):
    """
    分析結果から詳細なレポートを生成する強化版関数
    
    Parameters:
    analysis_result (dict): 強化された分析結果
    openai_api_key (str): OpenAI API Key
    model (str): 使用するモデル
    
    Returns:
    dict: ChatGPTの解釈結果
    """
    if not openai_api_key:
        st.warning("OpenAI API Keyが設定されていません。分析結果の解釈を行うにはAPI Keyを設定してください。")
        return None
    
    try:
        # OpenAI APIの設定
        openai.api_key = openai_api_key
        
        # エグゼクティブサマリーの取得
        if 'executive_summary' in analysis_result:
            executive_summary = analysis_result['executive_summary']
            summary_text = executive_summary['summary_text']
            key_points = executive_summary.get('key_change_points', [])
            detailed_insights = executive_summary.get('detailed_insights', [])
        else:
            summary_text = "CPA・CV変化の分析結果"
            key_points = []
            detailed_insights = []
        
        # 基本情報の取得
        current_total = analysis_result['current_total']
        previous_total = analysis_result['previous_total']
        
        # 日付情報（あれば）
        current_days = analysis_result.get('current_days', 30)
        previous_days = analysis_result.get('previous_days', 30)
        
        # 主要指標の変化率計算
        imp_change = ((current_total['Impressions'] - previous_total['Impressions']) / previous_total['Impressions']) * 100 if previous_total['Impressions'] != 0 else float('inf')
        clicks_change = ((current_total['Clicks'] - previous_total['Clicks']) / previous_total['Clicks']) * 100 if previous_total['Clicks'] != 0 else float('inf')
        cost_change = ((current_total['Cost'] - previous_total['Cost']) / previous_total['Cost']) * 100 if previous_total['Cost'] != 0 else float('inf')
        cv_change = ((current_total['Conversions'] - previous_total['Conversions']) / previous_total['Conversions']) * 100 if previous_total['Conversions'] != 0 else float('inf')
        
        # CPA計算
        previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] > 0 else 0
        current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] > 0 else 0
        cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa > 0 else float('inf')
        
        # CV寄与度分析
        cv_contributions = analysis_result['cv_contribution'][:5].to_dict('records')
        
        # 詳細なCPA変化要因分析
        if 'cpa_attribution' in analysis_result:
            cpa_factors = analysis_result['cpa_attribution'][:5].to_dict('records')
        else:
            cpa_factors = analysis_result['cpa_change_factors'][:5].to_dict('records')
        
        # 構造変化分析
        structure_summary = None
        if 'structure_analysis' in analysis_result and analysis_result['structure_analysis']:
            structure_summary = analysis_result['structure_analysis']['summary']
            structure_data = analysis_result['structure_analysis']['structure_df'].to_dict('records')
        else:
            structure_data = []
        
        # 階層的変化点特定
        change_points = {}
        if 'change_points' in analysis_result:
            change_points = analysis_result['change_points']
        
        # 深掘り分析
        deep_dive_results = {}
        if 'deep_dive_results' in analysis_result:
            deep_dive_results = analysis_result['deep_dive_results']
        
        # プロンプトの作成
        prompt = f"""# 広告パフォーマンス詳細分析レポート

## エグゼクティブサマリー
{summary_text}

## 主要な変化点
{chr(10).join([f"- {point}" for point in key_points])}

## 全体指標の変化
| 指標 | 前期 | 当期 | 変化率 |
|------|-----|-----|--------|
| インプレッション数 | {previous_total['Impressions']:,.0f} | {current_total['Impressions']:,.0f} | {imp_change:.1f}% |
| クリック数 | {previous_total['Clicks']:,.0f} | {current_total['Clicks']:,.0f} | {clicks_change:.1f}% |
| コスト | {previous_total['Cost']:,.0f}円 | {current_total['Cost']:,.0f}円 | {cost_change:.1f}% |
| コンバージョン数 | {previous_total['Conversions']:.1f} | {current_total['Conversions']:.1f} | {cv_change:.1f}% |
| CPA | {previous_cpa:.0f}円 | {current_cpa:.0f}円 | {cpa_change:.1f}% |
| 日数 | {previous_days} | {current_days} | - |

## 構造変化の分析結果
"""

        if structure_summary:
            prompt += f"""
配分変更の影響: {structure_summary['cost_shift_percentage']:.1f}%
パフォーマンス変化の影響: {structure_summary['performance_change_percentage']:.1f}%

コスト配分変化の大きい媒体:
"""
            for i, item in enumerate(sorted(structure_data, key=lambda x: abs(x.get('cost_ratio_change', 0)), reverse=True)[:3]):
                prompt += f"- {item['media']}: {item['previous_cost_ratio']:.1f}% → {item['current_cost_ratio']:.1f}% ({item['cost_ratio_change']:.1f}%pt)\n"
        else:
            prompt += "構造変化の分析データはありません。\n"
        
        # CV寄与度分析
        prompt += """
## CV増減の寄与度分析
"""
        for item in cv_contributions:
            media_name = item.get('ServiceNameJA', 'Unknown')
            contribution = item.get('contribution_rate', 0)
            cv_change = item.get('cv_change', 0)
            prompt += f"- {media_name}: CV {cv_change:.1f}件変化、寄与率 {contribution:.1f}%\n"
        
        # CPA変化要因分析
        prompt += """
## CPA変化要因分析
"""
        for item in cpa_factors:
            media_name = item.get('ServiceNameJA', 'Unknown')
            cpa_change_rate = item.get('cpa_change_rate', 0)
            main_factor = item.get('main_factor', 'Unknown')
            description = item.get('description', '-')
            prompt += f"- {media_name}: CPA {cpa_change_rate:.1f}%変化、主要因: {main_factor}、{description}\n"
        
        # 深掘り分析
        if deep_dive_results:
            prompt += """
## 重要媒体の深掘り分析
"""
            for media_name, result in deep_dive_results.items():
                if result and 'detailed_analysis' in result:
                    detailed = result['detailed_analysis']
                    if detailed and 'cv_contribution' in detailed and not detailed['cv_contribution'].empty:
                        prompt += f"\n### {media_name}の詳細分析\n"
                        
                        # キャンペーンレベルのCV寄与度
                        campaign_cv = detailed['cv_contribution'].head(3)
                        for _, row in campaign_cv.iterrows():
                            campaign_name = row.get('CampaignName', 'Unknown')
                            cv_change = row.get('cv_change', 0)
                            contribution = row.get('contribution_rate', 0)
                            prompt += f"- {campaign_name}: CV {cv_change:.1f}件変化、寄与率 {contribution:.1f}%\n"
        
        # 詳細洞察
        if detailed_insights:
            prompt += """
## 詳細な洞察
"""
            for insight in detailed_insights:
                prompt += f"- {insight}\n"
        
        # レポート生成指示
        prompt += """
---

以上のデータと分析結果に基づいて、以下の内容を含む包括的な広告パフォーマンス分析レポートを作成してください：

1. **エグゼクティブサマリー（3行程度）**
   - 全体パフォーマンス変化の簡潔で明確な要約
   - 最も重要な変化点の強調

2. **CPA・CV変化の構造分析**
   - CPA・CV変化の発生構造（予算配分変更 vs 媒体自体のパフォーマンス変化）
   - 主要な変化要因の特定と根拠の説明

3. **媒体別の詳細分析**
   - 影響の大きい上位3-5媒体の詳細分析
   - 各媒体の変化が全体に与えた影響の定量化
   - 指標変化の連鎖関係（CTR→CVRなど）の説明

4. **階層的変化点の特定**
   - 媒体→キャンペーン→広告グループの流れで変化の発生源を特定
   - 特に注目すべき変化点の具体的な分析

5. **実践的対応策の提案**
   - 課題解決のための具体的かつ実行可能な提案（3-5項目）
   - 好機活用のための行動提案（3-5項目）
   - 優先順位とその根拠

以下の点に留意してください：
- 数値だけでなく、なぜその変化が起きたのかの解釈を提供
- 実行可能な具体的なアクションに落とし込む
- 重要度に応じて強弱をつけて説明する
- 専門的でありながらも理解しやすい表現を使用する
"""
        
        # ChatGPT APIにリクエスト
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "あなたは広告データ分析の専門家です。徹底的に定量分析を行い、複雑なデータから示唆に富んだ洞察と行動可能な提案を提供します。"},
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


# メイン関数の変更
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
    
    # API設定
    with st.sidebar.expander("API設定", expanded=False):
        openai_api_key = st.text_input("OpenAI API Key", type="password", key="unique_openai_api_key")
        if openai_api_key:
            st.success("APIキーが設定されました")
    
    # 自動分析設定
    with st.sidebar.expander("分析設定", expanded=True):
        analysis_depth = st.select_slider(
            "分析の深さ",
            options=["基本分析", "詳細分析", "高度分析"],
            value="詳細分析"
        )
        show_visualizations = st.checkbox("ビジュアライゼーションを表示", value=True)
        generate_detailed_report = st.checkbox("詳細レポートを生成", value=True)
    
    # データが読み込まれているか確認
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.info("サイドバーからデータを読み込んでください")
        return
    
    # データの概要を表示
    df = st.session_state['data']
    
    # タブの設定
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["データ概要", "期間比較分析", "詳細分析", "レポート出力", "分析手法の説明"])
    
    # タブ1: データ概要 (既存のコードを活用)
    
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
            
        else:  
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
        if st.button("分析実行", key="enhanced_analysis"):
            with st.spinner("高度な分析を実行中..."):
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
                
                # セッション状態に保存
                st.session_state['previous_df'] = previous_df
                st.session_state['current_df'] = current_df
                st.session_state['group_by_cols'] = group_by_cols
                
                # 分析深度に応じた処理
                if analysis_depth == "基本分析":
                    # 基本的な期間比較分析
                    analysis_result = compare_periods(current_df, previous_df, group_by_cols)
                    if analysis_result:
                        st.session_state['analysis_result'] = analysis_result
                elif analysis_depth == "詳細分析" or analysis_depth == "高度分析":
                    # 詳細な期間比較分析
                    analysis_result = compare_periods_enhanced(current_df, previous_df, group_by_cols)
                    if analysis_result:
                        st.session_state['analysis_result'] = analysis_result
                        
                        # ビジュアライゼーションの生成（設定されている場合）
                        if show_visualizations:
                            st.session_state['heatmap_fig'] = create_metrics_heatmap(analysis_result)
                            st.session_state['waterfall_fig'] = create_contribution_waterfall(analysis_result)
                            st.session_state['sankey_fig'] = create_structure_change_sankey(analysis_result)
                
                # ChatGPTによる分析結果の解釈（API Keyが設定されている場合）
                if openai_api_key and generate_detailed_report:
                    with st.spinner("ChatGPTによる詳細分析レポートを生成中..."):
                        try:
                            if analysis_depth == "基本分析":
                                interpretation = interpret_analysis_with_chatgpt(analysis_result, openai_api_key)
                            else:
                                interpretation = generate_enhanced_report(analysis_result, openai_api_key)
                                
                            if interpretation:
                                st.session_state['interpretation'] = interpretation
                                st.success("分析完了！「レポート出力」タブで詳細な結果を確認できます")
                            else:
                                st.warning("ChatGPTによる分析結果の解釈に失敗しました")
                        except Exception as e:
                            st.error(f"ChatGPT APIとの通信中にエラーが発生しました: {str(e)}")
                else:
                    st.success("分析完了！「詳細分析」タブで結果を確認できます")
                    if not openai_api_key and generate_detailed_report:
                        st.warning("OpenAI API Keyが設定されていないため、詳細レポート生成はスキップされました")
        
        # 分析結果があれば基本的な情報を表示
        if 'analysis_result' in st.session_state and st.session_state['analysis_result']:
            result = st.session_state['analysis_result']
            
            st.subheader("分析結果サマリー")
            
            # エグゼクティブサマリーの表示（強化版分析結果の場合）
            if 'executive_summary' in result:
                exec_summary = result['executive_summary']
                st.info(exec_summary['summary_text'])
                
                # 主要指標の変化
                col1, col2, col3 = st.columns(3)
                
                # 前期・当期の合計値
                current_total = result['current_total']
                previous_total = result['previous_total']
                
                # 各指標の変化率を計算
                cv_change = exec_summary['cv_change']
                cpa_change = exec_summary['cpa_change']
                
                # CPAの計算
                previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] != 0 else 0
                current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] != 0 else 0
                
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
                        f"{((current_total['Cost'] - previous_total['Cost']) / previous_total['Cost'] * 100):.1f}%" if previous_total['Cost'] > 0 else "N/A"
                    )
                
                with col3:
                    st.metric(
                        "CPA",
                        f"{current_cpa:.0f}円",
                        f"{cpa_change:.1f}%",
                        delta_color="inverse" # CPAは下がる方がプラス表示
                    )
                
                # 主要変化点の表示
                if 'key_change_points' in exec_summary and exec_summary['key_change_points']:
                    st.subheader("主要変化点")
                    for point in exec_summary['key_change_points']:
                        st.markdown(f"- {point}")
            else:
                # 従来の表示方法
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
            
            # 詳細分析が利用可能であることを通知
            st.info("「詳細分析」タブでより詳細な分析結果を確認できます。また「レポート出力」タブで包括的なレポートを生成できます。")
    
    # タブ3: 詳細分析
    with tab3:
        st.header("詳細分析")
        
        if 'analysis_result' not in st.session_state or st.session_state['analysis_result'] is None:
            st.info("「期間比較分析」タブで分析を実行してください")
            return
        
        analysis_result = st.session_state['analysis_result']
        
        # 詳細分析の表示
        if 'executive_summary' in analysis_result:
            # 強化版分析結果の表示
            
            # エグゼクティブサマリー
            st.subheader("エグゼクティブサマリー")
            exec_summary = analysis_result['executive_summary']
            st.info(exec_summary['summary_text'])
            
            # 重要変化点
            if 'key_change_points' in exec_summary and exec_summary['key_change_points']:
                with st.expander("重要変化点", expanded=True):
                    for point in exec_summary['key_change_points']:
                        st.markdown(f"- {point}")
            
            # 詳細な洞察
            if 'detailed_insights' in exec_summary and exec_summary['detailed_insights']:
                with st.expander("詳細な洞察", expanded=False):
                    for insight in exec_summary['detailed_insights']:
                        st.markdown(f"- {insight}")
            
            # 視覚化セクション
            st.subheader("データ可視化")
            
            # タブで視覚化を分ける
            viz_tabs = st.tabs(["指標変化ヒートマップ", "CPA変化要因", "構造変化"])
            
            with viz_tabs[0]:
                if 'heatmap_fig' in st.session_state and st.session_state['heatmap_fig'] is not None:
                    st.plotly_chart(st.session_state['heatmap_fig'], use_container_width=True)
                else:
                    st.info("指標変化ヒートマップは利用できません。「期間比較分析」タブでビジュアライゼーションを有効にして分析を再実行してください。")
            
            with viz_tabs[1]:
                if 'waterfall_fig' in st.session_state and st.session_state['waterfall_fig'] is not None:
                    st.plotly_chart(st.session_state['waterfall_fig'], use_container_width=True)
                else:
                    st.info("CPA変化要因チャートは利用できません。「期間比較分析」タブでビジュアライゼーションを有効にして分析を再実行してください。")
            
            with viz_tabs[2]:
                if 'sankey_fig' in st.session_state and st.session_state['sankey_fig'] is not None:
                    st.plotly_chart(st.session_state['sankey_fig'], use_container_width=True)
                else:
                    st.info("構造変化ダイアグラムは利用できません。「期間比較分析」タブでビジュアライゼーションを有効にして分析を再実行してください。")
            
            # 詳細な分析データ表示
            st.subheader("詳細分析データ")
            
            analysis_data_tabs = st.tabs(["CV寄与度", "CPA変化要因", "構造変化", "深掘り分析"])
            
            with analysis_data_tabs[0]:
                st.write("#### CV寄与度分析")
                if 'cv_contribution' in analysis_result:
                    cv_contribution = analysis_result['cv_contribution'].head(10)
                    st.dataframe(format_metrics(cv_contribution))
            
            with analysis_data_tabs[1]:
                st.write("#### CPA変化要因分析")
                if 'cpa_attribution' in analysis_result:
                    # 強化版CPA変化要因分析の表示
                    cpa_factors = analysis_result['cpa_attribution'].head(10)
                    # 主要列の選択
                    display_cols = ['ServiceNameJA', 'previous_cpa', 'current_cpa', 'cpa_change_rate', 
                                    'cvr_change_rate', 'cpc_change_rate', 'cvr_contribution_to_cpa', 
                                    'cpc_contribution_to_cpa', 'main_factor', 'secondary_factor', 'description']
                    valid_cols = [col for col in display_cols if col in cpa_factors.columns]
                    st.dataframe(format_metrics(cpa_factors[valid_cols]))
                elif 'cpa_change_factors' in analysis_result:
                    # 基本版CPA変化要因分析の表示
                    cpa_factors = analysis_result['cpa_change_factors'].head(10)
                    st.dataframe(format_metrics(cpa_factors))
            
            with analysis_data_tabs[2]:
                st.write("#### 構造変化分析")
                if 'structure_analysis' in analysis_result and analysis_result['structure_analysis']:
                    structure_summary = analysis_result['structure_analysis']['summary']
                    structure_df = analysis_result['structure_analysis']['structure_df']
                    
                    # サマリー情報
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "配分変更の影響",
                            f"{structure_summary['cost_shift_percentage']:.1f}%"
                        )
                    with col2:
                        st.metric(
                            "パフォーマンス変化の影響",
                            f"{structure_summary['performance_change_percentage']:.1f}%"
                        )
                    
                    # 構造データを表示
                    st.dataframe(format_metrics(structure_df.head(10)))
                else:
                    st.info("構造変化分析データはありません。")
            
            with analysis_data_tabs[3]:
                st.write("#### 媒体深掘り分析")
                if 'deep_dive_results' in analysis_result and analysis_result['deep_dive_results']:
                    # 分析対象の媒体選択
                    media_options = list(analysis_result['deep_dive_results'].keys())
                    selected_media = st.selectbox("媒体選択", media_options)
                    
                    if selected_media and selected_media in analysis_result['deep_dive_results']:
                        deep_dive = analysis_result['deep_dive_results'][selected_media]
                        
                        if 'detailed_analysis' in deep_dive and deep_dive['detailed_analysis']:
                            detailed = deep_dive['detailed_analysis']
                            
                            # CV寄与度
                            if 'cv_contribution' in detailed:
                                st.write("##### キャンペーンレベルのCV寄与度")
                                campaign_cv = detailed['cv_contribution'].head(5)
                                st.dataframe(format_metrics(campaign_cv))
                            
                            # CPA変化要因
                            if 'cpa_change_factors' in detailed:
                                st.write("##### キャンペーンレベルのCPA変化要因")
                                campaign_cpa = detailed['cpa_change_factors'].head(5)
                                st.dataframe(format_metrics(campaign_cpa))
                        
                        # 構造変化分析
                        if 'structure_analysis' in deep_dive and deep_dive['structure_analysis']:
                            st.write("##### キャンペーン間の構造変化")
                            campaign_structure = deep_dive['structure_analysis']['structure_df']
                            st.dataframe(format_metrics(campaign_structure.head(5)))
                        
                        # 時系列分析
                        if 'time_series_analysis' in deep_dive and deep_dive['time_series_analysis']:
                            st.write("##### 時系列分析")
                            time_series = deep_dive['time_series_analysis']
                            
                            # 指標選択
                            metric_options = ["Impressions", "Clicks", "Cost", "Conversions", "CTR", "CVR", "CPC", "CPA"]
                            selected_metric = st.selectbox("表示する指標", metric_options)
                            
                            if selected_metric in metric_options:
                                # 時系列チャートの作成
                                fig = go.Figure()
                                
                                current_daily = time_series['current_daily']
                                previous_daily = time_series['previous_daily']
                                
                                # 前期データ
                                if selected_metric in previous_daily.columns:
                                    fig.add_trace(go.Scatter(
                                        x=previous_daily['Date'],
                                        y=previous_daily[selected_metric],
                                        mode='lines',
                                        name='前期',
                                        line=dict(color='blue', dash='dash')
                                    ))
                                
                                # 当期データ
                                if selected_metric in current_daily.columns:
                                    fig.add_trace(go.Scatter(
                                        x=current_daily['Date'],
                                        y=current_daily[selected_metric],
                                        mode='lines',
                                        name='当期',
                                        line=dict(color='red')
                                    ))
                                
                                # レイアウト設定
                                fig.update_layout(
                                    title=f"{selected_media}の{selected_metric}時系列推移",
                                    xaxis_title="日付",
                                    yaxis_title=selected_metric,
                                    legend=dict(x=0, y=1, traceorder='normal'),
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("媒体深掘り分析データはありません。")
        else:
            # 従来の分析結果表示
            st.subheader("基本分析結果")
            
            # 1. 全体サマリー
            st.write("#### 全体サマリー")
            
            # 全体指標の表示（既存コードを活用）
            current_total = analysis_result['current_total']
            previous_total = analysis_result['previous_total']
            
            # インプレッションと1000回表示あたりのコスト（CPM）
            previous_cpm = (previous_total['Cost'] / previous_total['Impressions']) * 1000 if previous_total['Impressions'] != 0 else 0
            current_cpm = (current_total['Cost'] / current_total['Impressions']) * 1000 if current_total['Impressions'] != 0 else 0
            
            # クリック率（CTR）
            previous_ctr = (previous_total['Clicks'] / previous_total['Impressions']) * 100 if previous_total['Impressions'] != 0 else 0
            current_ctr = (current_total['Clicks'] / current_total['Impressions']) * 100 if current_total['Impressions'] != 0 else 0
            
            # クリックあたりのコスト（CPC）
            previous_cpc = previous_total['Cost'] / previous_total['Clicks'] if previous_total['Clicks'] != 0 else 0
            current_cpc = current_total['Cost'] / current_total['Clicks'] if current_total['Clicks'] != 0 else 0
            
            # コンバージョン率（CVR）
            previous_cvr = (previous_total['Conversions'] / previous_total['Clicks']) * 100 if previous_total['Clicks'] != 0 else 0
            current_cvr = (current_total['Conversions'] / current_total['Clicks']) * 100 if current_total['Clicks'] != 0 else 0
            
            # コンバージョンあたりのコスト（CPA）
            previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] != 0 else 0
            current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] != 0 else 0
            
            # 変化率の計算
            imp_change = ((current_total['Impressions'] - previous_total['Impressions']) / previous_total['Impressions']) * 100 if previous_total['Impressions'] != 0 else float('inf')
            cpm_change = ((current_cpm - previous_cpm) / previous_cpm) * 100 if previous_cpm != 0 else float('inf')
            clicks_change = ((current_total['Clicks'] - previous_total['Clicks']) / previous_total['Clicks']) * 100 if previous_total['Clicks'] != 0 else float('inf')
            ctr_change = ((current_ctr - previous_ctr) / previous_ctr) * 100 if previous_ctr != 0 else float('inf')
            cpc_change = ((current_cpc - previous_cpc) / previous_cpc) * 100 if previous_cpc != 0 else float('inf')
            cost_change = ((current_total['Cost'] - previous_total['Cost']) / previous_total['Cost']) * 100 if previous_total['Cost'] != 0 else float('inf')
            cv_change = ((current_total['Conversions'] - previous_total['Conversions']) / previous_total['Conversions']) * 100 if previous_total['Conversions'] != 0 else float('inf')
            cvr_change = ((current_cvr - previous_cvr) / previous_cvr) * 100 if previous_cvr != 0 else float('inf')
            cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else float('inf')
            
            # 日平均値の計算
            current_days = analysis_result.get('current_days', 30)
            previous_days = analysis_result.get('previous_days', 30)
            previous_daily_cv = previous_total['Conversions'] / previous_days if previous_days > 0 else 0
            current_daily_cv = current_total['Conversions'] / current_days if current_days > 0 else 0
            daily_cv_change = ((current_daily_cv - previous_daily_cv) / previous_daily_cv) * 100 if previous_daily_cv != 0 else float('inf')
            
            # サマリーテーブルをデータフレームとして作成
            summary_data = {
                '指標': ['インプレッション数', 'CPM', 'クリック数', 'CTR', 'CPC', 'コスト', 'コンバージョン数', 'CPA', 'CVR', '日数', '日平均CV数'],
                '前期': [
                    f"{previous_total['Impressions']:,.0f}", f"{previous_cpm:.0f}円", f"{previous_total['Clicks']:,.0f}", 
                    f"{previous_ctr:.1f}%", f"{previous_cpc:.0f}円", f"{previous_total['Cost']:,.0f}円", 
                    f"{previous_total['Conversions']:.1f}", f"{previous_cpa:.0f}円", f"{previous_cvr:.1f}%", 
                    f"{previous_days}", f"{previous_daily_cv:.1f}"
                ],
                '当期': [
                    f"{current_total['Impressions']:,.0f}", f"{current_cpm:.0f}円", f"{current_total['Clicks']:,.0f}", 
                    f"{current_ctr:.1f}%", f"{current_cpc:.0f}円", f"{current_total['Cost']:,.0f}円", 
                    f"{current_total['Conversions']:.1f}", f"{current_cpa:.0f}円", f"{current_cvr:.1f}%", 
                    f"{current_days}", f"{current_daily_cv:.1f}"
                ],
                '差分': [
                    f"{current_total['Impressions'] - previous_total['Impressions']:,.0f}", 
                    f"{current_cpm - previous_cpm:.0f}円", 
                    f"{current_total['Clicks'] - previous_total['Clicks']:,.0f}", 
                    f"{current_ctr - previous_ctr:.1f}%", 
                    f"{current_cpc - previous_cpc:.0f}円", 
                    f"{current_total['Cost'] - previous_total['Cost']:,.0f}円", 
                    f"{current_total['Conversions'] - previous_total['Conversions']:.1f}", 
                    f"{current_cpa - previous_cpa:.0f}円", 
                    f"{current_cvr - previous_cvr:.1f}%", 
                    f"{current_days - previous_days}", 
                    f"{current_daily_cv - previous_daily_cv:.1f}"
                ],
                '変化率': [
                    f"{imp_change:.1f}%", f"{cpm_change:.1f}%", f"{clicks_change:.1f}%", 
                    f"{ctr_change:.1f}%", f"{cpc_change:.1f}%", f"{cost_change:.1f}%", 
                    f"{cv_change:.1f}%", f"{cpa_change:.1f}%", f"{cvr_change:.1f}%", 
                    "-", f"{daily_cv_change:.1f}%"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True)
            
            # 2. CV増減の寄与度ランキング
            st.write("#### CV増減の寄与度ランキング")
            
            cv_contribution = analysis_result['cv_contribution'].head(10)
            st.dataframe(format_metrics(cv_contribution))
            
            # 3. CPA変化要因分析
            st.write("#### CPA変化要因分析")
            
            cpa_factors = analysis_result['cpa_change_factors'].head(10)
            st.dataframe(format_metrics(cpa_factors))
            
            # 4. 媒体グループ・パターン分析
            st.write("#### 媒体グループ・パターン分析")
            
            patterns = analysis_result['media_patterns']['pattern_df']
            st.dataframe(format_metrics(patterns.head(10)))
    
    # タブ4: レポート出力
    with tab4:
        st.header("レポート出力")
        
        # 分析結果があるか確認
        if 'analysis_result' not in st.session_state or st.session_state['analysis_result'] is None:
            st.info("「期間比較分析」タブで分析を実行してください")
            return
        
        # ビジュアライゼーションがあれば表示
        if show_visualizations:
            st.subheader("データ可視化")
            
            viz_cols = st.columns(2)
            
            with viz_cols[0]:
                if 'heatmap_fig' in st.session_state and st.session_state['heatmap_fig'] is not None:
                    st.plotly_chart(st.session_state['heatmap_fig'], use_container_width=True)
            
            with viz_cols[1]:
                if 'waterfall_fig' in st.session_state and st.session_state['waterfall_fig'] is not None:
                    st.plotly_chart(st.session_state['waterfall_fig'], use_container_width=True)
            
            if 'sankey_fig' in st.session_state and st.session_state['sankey_fig'] is not None:
                st.plotly_chart(st.session_state['sankey_fig'], use_container_width=True)
        
        # ChatGPTによる解釈結果があるか確認
        if 'interpretation' in st.session_state and st.session_state['interpretation']:
            interpretation = st.session_state['interpretation']
            
            # レポート表示
            st.subheader("広告パフォーマンス分析レポート")
            
            # フォーマット切り替えオプション
            report_format = st.radio(
                "レポート形式",
                ["詳細レポート", "要約レポート", "プレゼンテーション形式"],
                horizontal=True
            )
            
            if report_format == "詳細レポート":
                # 詳細レポートを表示
                st.markdown(interpretation['interpretation'])
            
            elif report_format == "要約レポート":
                # 要約レポートを生成（ChatGPTを使用）
                if openai_api_key:
                    with st.spinner("要約レポートを生成中..."):
                        try:
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "あなたは広告データ分析の専門家です。詳細レポートを簡潔な要約に変換してください。"},
                                    {"role": "user", "content": f"以下の詳細レポートを、経営層向けの簡潔な要約（500字程度）に変換してください。重要なポイントと具体的な数値、そして最も重要なアクションアイテムを含めてください。\n\n{interpretation['interpretation']}"}
                                ],
                                temperature=0.5,
                                max_tokens=1000
                            )
                            
                            summary = response.choices[0].message.content
                            st.markdown(summary)
                        except Exception as e:
                            st.error(f"要約レポート生成中にエラーが発生しました: {str(e)}")
                            st.markdown(interpretation['interpretation'])
                else:
                    st.warning("要約レポートを生成するにはOpenAI API Keyが必要です。")
                    st.markdown(interpretation['interpretation'])
            
            else:  # プレゼンテーション形式
                # プレゼンテーション形式でレポートを生成（ChatGPTを使用）
                if openai_api_key:
                    with st.spinner("プレゼンテーション形式のレポートを生成中..."):
                        try:
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "あなたは広告データ分析の専門家です。詳細レポートをプレゼンテーション形式に変換してください。"},
                                    {"role": "user", "content": f"以下の詳細レポートを、プレゼンテーション用のスライド形式に変換してください。各スライドは「# スライドタイトル」で始まり、箇条書きの重要ポイントを含めてください。合計5-7スライドにまとめ、最初のスライドは全体サマリー、最後のスライドはアクションアイテムとしてください。\n\n{interpretation['interpretation']}"}
                                ],
                                temperature=0.5,
                                max_tokens=1500
                            )
                            
                            presentation = response.choices[0].message.content
                            st.markdown(presentation)
                        except Exception as e:
                            st.error(f"プレゼンテーション形式レポート生成中にエラーが発生しました: {str(e)}")
                            st.markdown(interpretation['interpretation'])
                else:
                    st.warning("プレゼンテーション形式レポートを生成するにはOpenAI API Keyが必要です。")
                    st.markdown(interpretation['interpretation'])
            
            # プロンプトを表示するオプション
            with st.expander("分析プロンプト（開発者用）", expanded=False):
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
            
            # エクセルレポートのダウンロードオプション
            if 'analysis_result' in st.session_state:
                analysis_result = st.session_state['analysis_result']
                
                if st.button("エクセルレポートを生成"):
                    with st.spinner("エクセルレポートを生成中..."):
                        try:
                            # エクセルレポート生成用の関数を呼び出し
                            excel_file = generate_excel_report(analysis_result)
                            
                            if excel_file:
                                st.download_button(
                                    label="エクセルレポートをダウンロード",
                                    data=excel_file,
                                    file_name="広告パフォーマンス分析レポート.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        except Exception as e:
                            st.error(f"エクセルレポート生成中にエラーが発生しました: {str(e)}")
        else:
            st.warning("ChatGPTによる分析レポートがありません。OpenAI API Keyを設定して分析を再実行するか、「詳細分析」タブで結果を確認してください。")
            
            # 手動でレポートを生成するためのプロンプト表示
            st.subheader("手動レポート生成用プロンプト")
            if 'analysis_result' in st.session_state:
                analysis_result = st.session_state['analysis_result']
                
                if 'executive_summary' in analysis_result:
                    # 強化版分析結果の場合
                    prompt_data = {
                        'executive_summary': analysis_result['executive_summary'],
                        'cv_contribution': analysis_result['cv_contribution'].to_dict('records')[:5],
                        'cpa_factors': analysis_result['cpa_attribution'].to_dict('records')[:5] if 'cpa_attribution' in analysis_result else analysis_result['cpa_change_factors'].to_dict('records')[:5],
                        'structure_analysis': analysis_result['structure_analysis'] if 'structure_analysis' in analysis_result else None
                    }
                    
                    prompt = f"""# 広告パフォーマンス分析

## エグゼクティブサマリー
{prompt_data['executive_summary']['summary_text']}

## 主要な変化点
{chr(10).join([f"- {point}" for point in prompt_data['executive_summary'].get('key_change_points', [])])}

## CV増減の寄与度分析
{''.join([f"- {item.get('ServiceNameJA', 'Unknown')}: CV {item.get('cv_change', 0):.1f}件変化、寄与率 {item.get('contribution_rate', 0):.1f}%" for item in prompt_data['cv_contribution']])}

## CPA変化要因分析
{''.join([f"- {item.get('ServiceNameJA', 'Unknown')}: CPA {item.get('cpa_change_rate', 0):.1f}%変化、主要因: {item.get('main_factor', 'Unknown')}" for item in prompt_data['cpa_factors']])}

上記のデータに基づいて、広告パフォーマンスの詳細分析と具体的な改善提案を含むレポートを作成してください。
"""
                else:
                    # 基本分析結果の場合
                    prompt_data = format_prompt_data(analysis_result)
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

# エクセルレポート生成関数
def generate_excel_report(analysis_result):
    """
    分析結果からエクセルレポートを生成する
    
    Parameters:
    analysis_result (dict): 分析結果
    
    Returns:
    bytes: エクセルファイルのバイナリデータ
    """
    try:
        # Excelファイルの作成
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        
        # ワークブックとワークシートの取得
        workbook = writer.book
        
        # 全体サマリーシート
        summary_sheet = workbook.add_worksheet('全体サマリー')
        
        # 書式設定
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
        cell_format = workbook.add_format({'border': 1})
        percent_format = workbook.add_format({'num_format': '0.0%', 'border': 1})
        currency_format = workbook.add_format({'num_format': '¥#,##0', 'border': 1})
        
        # 基本情報の取得
        current_total = analysis_result['current_total']
        previous_total = analysis_result['previous_total']
        
        # 日付情報（あれば）
        current_days = analysis_result.get('current_days', 30)
        previous_days = analysis_result.get('previous_days', 30)
        
        # インプレッションと1000回表示あたりのコスト（CPM）
        previous_cpm = (previous_total['Cost'] / previous_total['Impressions']) * 1000 if previous_total['Impressions'] != 0 else 0
        current_cpm = (current_total['Cost'] / current_total['Impressions']) * 1000 if current_total['Impressions'] != 0 else 0
        
        # クリック率（CTR）
        previous_ctr = (previous_total['Clicks'] / previous_total['Impressions']) * 100 if previous_total['Impressions'] != 0 else 0
        current_ctr = (current_total['Clicks'] / current_total['Impressions']) * 100 if current_total['Impressions'] != 0 else 0
        
        # クリックあたりのコスト（CPC）
        previous_cpc = previous_total['Cost'] / previous_total['Clicks'] if previous_total['Clicks'] != 0 else 0
        current_cpc = current_total['Cost'] / current_total['Clicks'] if current_total['Clicks'] != 0 else 0
        
        # コンバージョン率（CVR）
        previous_cvr = (previous_total['Conversions'] / previous_total['Clicks']) * 100 if previous_total['Clicks'] != 0 else 0
        current_cvr = (current_total['Conversions'] / current_total['Clicks']) * 100 if current_total['Clicks'] != 0 else 0
        
        # コンバージョンあたりのコスト（CPA）
        previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] != 0 else 0
        current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] != 0 else 0
        
        # タイトル
        summary_sheet.write(0, 0, '広告パフォーマンス分析レポート', workbook.add_format({'bold': True, 'font_size': 14}))
        
        # エグゼクティブサマリー
        if 'executive_summary' in analysis_result:
            summary_sheet.write(2, 0, 'エグゼクティブサマリー:', workbook.add_format({'bold': True}))
            summary_sheet.write(3, 0, analysis_result['executive_summary']['summary_text'])
            
            # 主要変化点
            row = 5
            summary_sheet.write(row, 0, '主要変化点:', workbook.add_format({'bold': True}))
            row += 1
            
            for point in analysis_result['executive_summary'].get('key_change_points', []):
                summary_sheet.write(row, 0, f"• {point}")
                row += 1
        
        # 全体指標の変化率計算
        imp_change = ((current_total['Impressions'] - previous_total['Impressions']) / previous_total['Impressions']) if previous_total['Impressions'] != 0 else 0
        clicks_change = ((current_total['Clicks'] - previous_total['Clicks']) / previous_total['Clicks']) if previous_total['Clicks'] != 0 else 0
        cost_change = ((current_total['Cost'] - previous_total['Cost']) / previous_total['Cost']) if previous_total['Cost'] != 0 else 0
        cv_change = ((current_total['Conversions'] - previous_total['Conversions']) / previous_total['Conversions']) if previous_total['Conversions'] != 0 else 0
        cpa_change = ((current_cpa - previous_cpa) / previous_cpa) if previous_cpa != 0 else 0
        
        # 全体サマリーテーブル
        row = 10
        summary_sheet.write(row, 0, '全体指標サマリー:', workbook.add_format({'bold': True}))
        row += 1
        
        # ヘッダー
        headers = ['指標', '前期', '当期', '変化量', '変化率']
        for col, header in enumerate(headers):
            summary_sheet.write(row, col, header, header_format)
        
        # データ行
        metrics = [
            ('インプレッション数', previous_total['Impressions'], current_total['Impressions'], imp_change),
            ('CPM (円)', previous_cpm, current_cpm, (current_cpm - previous_cpm) / previous_cpm if previous_cpm != 0 else 0),
            ('クリック数', previous_total['Clicks'], current_total['Clicks'], clicks_change),
            ('CTR (%)', previous_ctr / 100, current_ctr / 100, (current_ctr - previous_ctr) / previous_ctr if previous_ctr != 0 else 0),
            ('CPC (円)', previous_cpc, current_cpc, (current_cpc - previous_cpc) / previous_cpc if previous_cpc != 0 else 0),
            ('コスト (円)', previous_total['Cost'], current_total['Cost'], cost_change),
            ('コンバージョン数', previous_total['Conversions'], current_total['Conversions'], cv_change),
            ('CVR (%)', previous_cvr / 100, current_cvr / 100, (current_cvr - previous_cvr) / previous_cvr if previous_cvr != 0 else 0),
            ('CPA (円)', previous_cpa, current_cpa, cpa_change),
            ('日数', previous_days, current_days, (current_days - previous_days) / previous_days if previous_days != 0 else 0)
        ]
        
        for i, (metric, prev, curr, change) in enumerate(metrics):
            row += 1
            summary_sheet.write(row, 0, metric, cell_format)
            
            # 指標の種類に応じてフォーマットを変更
            if 'CPA' in metric or 'CPC' in metric or 'CPM' in metric or 'コスト' in metric:
                summary_sheet.write(row, 1, prev, currency_format)
                summary_sheet.write(row, 2, curr, currency_format)
                summary_sheet.write(row, 3, curr - prev, currency_format)
            elif 'CTR' in metric or 'CVR' in metric:
                summary_sheet.write(row, 1, prev, percent_format)
                summary_sheet.write(row, 2, curr, percent_format)
                summary_sheet.write(row, 3, curr - prev, percent_format)
            else:
                summary_sheet.write(row, 1, prev, cell_format)
                summary_sheet.write(row, 2, curr, cell_format)
                summary_sheet.write(row, 3, curr - prev, cell_format)
            
            summary_sheet.write(row, 4, change, percent_format)
        
        # CV寄与度シート
        if 'cv_contribution' in analysis_result:
            cv_contribution = analysis_result['cv_contribution']
            cv_sheet = workbook.add_worksheet('CV寄与度分析')
            
            # ヘッダー
            headers = ['媒体名', '前期CV', '当期CV', 'CV変化', '寄与率(%)', 'ステータス']
            for col, header in enumerate(headers):
                cv_sheet.write(0, col, header, header_format)
            
            # データ
            for i, (_, row_data) in enumerate(cv_contribution.iterrows(), 1):
                media_col = 'ServiceNameJA' if 'ServiceNameJA' in row_data else cv_contribution.columns[0]
                cv_sheet.write(i, 0, row_data[media_col], cell_format)
                cv_sheet.write(i, 1, row_data['previous_cv'], cell_format)
                cv_sheet.write(i, 2, row_data['current_cv'], cell_format)
                cv_sheet.write(i, 3, row_data['cv_change'], cell_format)
                cv_sheet.write(i, 4, row_data['contribution_rate'] / 100, percent_format)
                cv_sheet.write(i, 5, row_data.get('entry_status', '継続'), cell_format)
        
        # CPA変化要因シート
        if 'cpa_attribution' in analysis_result:
            cpa_factors = analysis_result['cpa_attribution']
            cpa_sheet = workbook.add_worksheet('CPA変化要因分析')
            
            # ヘッダー
            headers = ['媒体名', '前期CPA', '当期CPA', 'CPA変化率', 'CVR変化率', 'CPC変化率', 'CVR寄与度', 'CPC寄与度', '主要因', '説明']
            for col, header in enumerate(headers):
                cpa_sheet.write(0, col, header, header_format)
            
            # データ
            for i, (_, row_data) in enumerate(cpa_factors.iterrows(), 1):
                media_col = 'ServiceNameJA' if 'ServiceNameJA' in row_data else cpa_factors.columns[0]
                cpa_sheet.write(i, 0, row_data[media_col], cell_format)
                cpa_sheet.write(i, 1, row_data['previous_cpa'], currency_format)
                cpa_sheet.write(i, 2, row_data['current_cpa'], currency_format)
                cpa_sheet.write(i, 3, row_data['cpa_change_rate'] / 100, percent_format)
                cpa_sheet.write(i, 4, row_data['cvr_change_rate'] / 100, percent_format)
                cpa_sheet.write(i, 5, row_data['cpc_change_rate'] / 100, percent_format)
                cpa_sheet.write(i, 6, row_data['cvr_contribution_to_cpa'] / 100, percent_format)
                cpa_sheet.write(i, 7, row_data['cpc_contribution_to_cpa'] / 100, percent_format)
                cpa_sheet.write(i, 8, row_data['main_factor'], cell_format)
                cpa_sheet.write(i, 9, row_data['description'], cell_format)
        elif 'cpa_change_factors' in analysis_result:
            cpa_factors = analysis_result['cpa_change_factors']
            cpa_sheet = workbook.add_worksheet('CPA変化要因分析')
            
            # ヘッダー
            headers = ['媒体名', '前期CPA', '当期CPA', 'CPA変化率', '主要因', '副要因', '説明']
            for col, header in enumerate(headers):
                cpa_sheet.write(0, col, header, header_format)
            
            # データ
            for i, (_, row_data) in enumerate(cpa_factors.iterrows(), 1):
                media_col = 'ServiceNameJA' if 'ServiceNameJA' in row_data else cpa_factors.columns[0]
                cpa_sheet.write(i, 0, row_data[media_col], cell_format)
                cpa_sheet.write(i, 1, row_data['previous_cpa'], currency_format)
                cpa_sheet.write(i, 2, row_data['current_cpa'], currency_format)
                cpa_sheet.write(i, 3, row_data['cpa_change_rate'] / 100, percent_format)
                cpa_sheet.write(i, 4, row_data['main_factor'], cell_format)
                cpa_sheet.write(i, 5, row_data.get('secondary_factor', '-'), cell_format)
                cpa_sheet.write(i, 6, row_data['description'], cell_format)
        
        # 構造変化分析シート
        if 'structure_analysis' in analysis_result and analysis_result['structure_analysis']:
            structure = analysis_result['structure_analysis']
            structure_df = structure['structure_df']
            structure_sheet = workbook.add_worksheet('構造変化分析')
            
            # サマリー情報
            structure_sheet.write(0, 0, '構造変化サマリー:', workbook.add_format({'bold': True}))
            structure_sheet.write(1, 0, '配分変更の影響:')
            structure_sheet.write(1, 1, structure['summary']['cost_shift_percentage'] / 100, percent_format)
            structure_sheet.write(2, 0, 'パフォーマンス変化の影響:')
            structure_sheet.write(2, 1, structure['summary']['performance_change_percentage'] / 100, percent_format)
            
            # 構造データテーブル
            structure_sheet.write(4, 0, '媒体別の構造変化:', workbook.add_format({'bold': True}))
            
            # ヘッダー
            headers = ['媒体名', '前期コスト', '当期コスト', '前期コスト比率', '当期コスト比率', 'コスト比率変化',
                      '前期CPA', '当期CPA', 'CPA変化率', 'コスト配分影響', 'パフォーマンス影響']
            for col, header in enumerate(headers):
                structure_sheet.write(5, col, header, header_format)
            
            # データ
            for i, (_, row_data) in enumerate(structure_df.iterrows(), 6):
                structure_sheet.write(i, 0, row_data['media'], cell_format)
                structure_sheet.write(i, 1, row_data['previous_cost'], currency_format)
                structure_sheet.write(i, 2, row_data['current_cost'], currency_format)
                structure_sheet.write(i, 3, row_data['previous_cost_ratio'] / 100, percent_format)
                structure_sheet.write(i, 4, row_data['current_cost_ratio'] / 100, percent_format)
                structure_sheet.write(i, 5, row_data['cost_ratio_change'] / 100, percent_format)
                
                # CPA関連列
                if row_data['previous_cpa'] != float('inf'):
                    structure_sheet.write(i, 6, row_data['previous_cpa'], currency_format)
                else:
                    structure_sheet.write(i, 6, 'N/A', cell_format)
                
                if row_data['current_cpa'] != float('inf'):
                    structure_sheet.write(i, 7, row_data['current_cpa'], currency_format)
                else:
                    structure_sheet.write(i, 7, 'N/A', cell_format)
                
                if pd.notnull(row_data.get('cpa_change_rate')) and row_data.get('cpa_change_rate') != float('inf'):
                    structure_sheet.write(i, 8, row_data['cpa_change_rate'] / 100, percent_format)
                else:
                    structure_sheet.write(i, 8, 'N/A', cell_format)
                
                structure_sheet.write(i, 9, row_data['cost_shift_impact'], cell_format)
                structure_sheet.write(i, 10, row_data['performance_impact'], cell_format)
        
        # 深掘り分析シート
        if 'deep_dive_results' in analysis_result and analysis_result['deep_dive_results']:
            deep_dive_sheet = workbook.add_worksheet('深掘り分析')
            
            row = 0
            for media_name, deep_dive in analysis_result['deep_dive_results'].items():
                deep_dive_sheet.write(row, 0, f"{media_name}の深掘り分析", workbook.add_format({'bold': True, 'font_size': 12}))
                row += 1
                
                if 'detailed_analysis' in deep_dive and deep_dive['detailed_analysis']:
                    detailed = deep_dive['detailed_analysis']
                    
                    # CV寄与度
                    if 'cv_contribution' in detailed and not detailed['cv_contribution'].empty:
                        deep_dive_sheet.write(row, 0, "キャンペーンレベルのCV寄与度:", workbook.add_format({'bold': True}))
                        row += 1
                        
                        # ヘッダー
                        campaign_headers = ['キャンペーン名', '前期CV', '当期CV', 'CV変化', '寄与率(%)']
                        for col, header in enumerate(campaign_headers):
                            deep_dive_sheet.write(row, col, header, header_format)
                        row += 1
                        
                        # データ
                        campaign_cv = detailed['cv_contribution'].head(5)
                        for _, campaign_row in campaign_cv.iterrows():
                            campaign_name = campaign_row.get('CampaignName', 'Unknown')
                            deep_dive_sheet.write(row, 0, campaign_name, cell_format)
                            deep_dive_sheet.write(row, 1, campaign_row['previous_cv'], cell_format)
                            deep_dive_sheet.write(row, 2, campaign_row['current_cv'], cell_format)
                            deep_dive_sheet.write(row, 3, campaign_row['cv_change'], cell_format)
                            deep_dive_sheet.write(row, 4, campaign_row['contribution_rate'] / 100, percent_format)
                            row += 1
                        
                        row += 1  # 空行
                
                # 構造変化
                if 'structure_analysis' in deep_dive and deep_dive['structure_analysis']:
                    deep_dive_sheet.write(row, 0, "キャンペーン間の構造変化:", workbook.add_format({'bold': True}))
                    row += 1
                    
                    # ヘッダー
                    structure_headers = ['名前', '前期コスト', '当期コスト', '前期コスト比率', '当期コスト比率', 'コスト比率変化']
                    for col, header in enumerate(structure_headers):
                        deep_dive_sheet.write(row, col, header, header_format)
                    row += 1
                    
                    # データ
                    campaign_structure = deep_dive['structure_analysis']['structure_df'].head(5)
                    for _, structure_row in campaign_structure.iterrows():
                        name = structure_row.get('media', 'Unknown')
                        deep_dive_sheet.write(row, 0, name, cell_format)
                        deep_dive_sheet.write(row, 1, structure_row['previous_cost'], currency_format)
                        deep_dive_sheet.write(row, 2, structure_row['current_cost'], currency_format)
                        deep_dive_sheet.write(row, 3, structure_row['previous_cost_ratio'] / 100, percent_format)
                        deep_dive_sheet.write(row, 4, structure_row['current_cost_ratio'] / 100, percent_format)
                        deep_dive_sheet.write(row, 5, structure_row['cost_ratio_change'] / 100, percent_format)
                        row += 1
                
                row += 2  # 空行
        
        # Excelファイルを保存して返す
        writer.close()
        output.seek(0)
        
        return output.getvalue()
    
    except Exception as e:
        st.error(f"エクセルレポート生成中にエラーが発生しました: {str(e)}")
        return None


# アプリケーションの実行
if __name__ == "__main__":
    main()