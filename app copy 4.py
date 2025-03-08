import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# セッション状態の初期化
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None
if 'auto_analysis_result' not in st.session_state:
    st.session_state['auto_analysis_result'] = None
if 'structural_analysis_result' not in st.session_state:
    st.session_state['structural_analysis_result'] = None

# サイドバーにタイトルを表示
st.sidebar.title("広告パフォーマンス分析システム")

# APIキーをサイドバーに移動
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
        
        # 認証情報の取得
        if 'gcp_service_account' in st.secrets:
            credentials_info = st.secrets['gcp_service_account']
            credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
        else:
            # ローカル開発用（サービスアカウントのJSONファイルが必要）
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

# 派生指標の計算関数 - 追加要件に応じて強化
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
    
    # 追加の派生指標（新要件対応）
    if 'Conversions' in df.columns and 'Impressions' in df.columns:
        df['IPR'] = safe_divide(df['Conversions'], df['Impressions']) * 100  # インプレッション単位のコンバージョン率
    
    # ROAS関連の計算（コンバージョン価値がある場合）
    if 'ConversionValue' in df.columns and 'Cost' in df.columns:
        df['ROAS'] = safe_divide(df['ConversionValue'], df['Cost']) * 100
        df['CPO'] = safe_divide(df['Cost'], df['Conversions'])  # Orderあたりのコスト
        df['RPO'] = safe_divide(df['ConversionValue'], df['Conversions'])  # Orderあたりの収益
    
    return df

# データをフィルタリングする関数（期間指定）
def filter_data_by_date(df, start_date, end_date):
    if 'Date' not in df.columns:
        st.error("データに日付列がありません")
        return df
    
    # datetime.date オブジェクトを pandas の Timestamp に変換
    if hasattr(start_date, 'date') and not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
    elif hasattr(start_date, 'year') and hasattr(start_date, 'month') and hasattr(start_date, 'day'):
        start_date = pd.Timestamp(start_date)
        
    if hasattr(end_date, 'date') and not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp(end_date)
    elif hasattr(end_date, 'year') and hasattr(end_date, 'month') and hasattr(end_date, 'day'):
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
    
    # コンバージョン価値がある場合は追加
    if 'ConversionValue' in df.columns:
        agg_dict['ConversionValue'] = 'sum'
    
    # 集計
    agg_df = df.groupby(group_by_cols).agg(agg_dict).reset_index()
    
    # 派生指標の計算
    agg_df = calculate_derived_metrics(agg_df)
    
    return agg_df

# 新規: 指標変化の寄与度を計算する関数
def calculate_metric_contribution(current_value, previous_value, current_base, previous_base):
    """
    指標変化の寄与度を計算する関数
    
    Parameters:
    current_value (float): 当期の指標値
    previous_value (float): 前期の指標値
    current_base (float): 当期の基準値
    previous_base (float): 前期の基準値
    
    Returns:
    float: 寄与度（%）
    """
    # 基準値の変化率
    if previous_base == 0:
        return 0
    
    base_change_rate = (current_base - previous_base) / previous_base
    
    # 0除算を避ける
    if base_change_rate == 0:
        return 0
    
    # 指標の変化率
    if previous_value == 0:
        metric_change_rate = 0
    else:
        metric_change_rate = (current_value - previous_value) / previous_value
    
    # 寄与度（基準値の変化に対する当該指標変化の寄与割合）
    contribution = (metric_change_rate / base_change_rate) * 100
    
    return contribution

# 1. CV増減の寄与度分析
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
    
    # 新規媒体の処理
    for idx in set(current_df.index) - set(previous_df.index):
        current_cv = current_df.loc[idx, 'Conversions']
        cv_change = current_cv
        
        # 寄与率の計算
        if total_cv_change != 0:
            contribution_rate = (cv_change / total_cv_change) * 100
        else:
            contribution_rate = float('inf') if cv_change > 0 else float('-inf')
        
        contribution_data.append({
            'index_value': idx,
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
            'index_value': idx,
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
    
    # 分析粒度に応じて、CampaignNameとAdgroupNameのカラムを追加
    if 'ServiceNameJA' in group_by_cols:
        # 分析粒度が「媒体」の場合
        if 'CampaignName' not in group_by_cols and 'CampaignName' in current_agg.columns:
            contribution_df['CampaignName'] = ''
        if 'AdgroupName' not in group_by_cols and 'AdgroupName' in current_agg.columns:
            contribution_df['AdgroupName'] = ''
    elif 'CampaignName' in group_by_cols:
        # 分析粒度が「キャンペーン」の場合
        if 'AdgroupName' not in group_by_cols and 'AdgroupName' in current_agg.columns:
            contribution_df['AdgroupName'] = ''
    
    # 寄与率の絶対値で降順ソート
    if 'contribution_rate' in contribution_df.columns:
        contribution_df['abs_contribution'] = contribution_df['contribution_rate'].abs()
        contribution_df = contribution_df.sort_values('abs_contribution', ascending=False)
        contribution_df = contribution_df.drop(columns=['abs_contribution'])
    
    return contribution_df

# 数値フォーマット関数
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
    
    # CPA変化の寄与度分解（あれば）
    cpa_decomposition = {}
    if 'cpa_decomposition' in analysis_result and not analysis_result['cpa_decomposition'].empty:
        cpa_decomp_df = analysis_result['cpa_decomposition'].head(5)
        
        # 平均的な寄与度を計算
        cvr_contribution = cpa_decomp_df['cvr_contribution'].mean()
        cpc_contribution = cpa_decomp_df['cpc_contribution'].mean()
        
        if 'cpm_contribution' in cpa_decomp_df.columns and 'ctr_contribution' in cpa_decomp_df.columns:
            cpm_contribution = cpa_decomp_df['cpm_contribution'].mean()
            ctr_contribution = cpa_decomp_df['ctr_contribution'].mean()
            
            cpa_decomposition = {
                'cvr_contribution': cvr_contribution,
                'cpc_contribution': cpc_contribution,
                'cpm_contribution': cpm_contribution,
                'ctr_contribution': ctr_contribution
            }
        else:
            cpa_decomposition = {
                'cvr_contribution': cvr_contribution,
                'cpc_contribution': cpc_contribution
            }
    
    # 構造変化分析（あれば）
    structure_analysis = {}
    if 'structure_analysis' in analysis_result:
        structure = analysis_result['structure_analysis']
        
        if 'allocation_contribution' in structure and 'performance_contribution' in structure:
            structure_analysis = {
                'allocation_contribution': structure['allocation_contribution'],
                'performance_contribution': structure['performance_contribution']
            }
            
            # 重要な構造変化があれば追加
            if 'structure_df' in structure:
                structure_df = structure['structure_df']
                
                # コスト配分変化の大きい上位3媒体
                top_allocation_changes = structure_df.sort_values('cost_ratio_change', key=abs, ascending=False).head(3)
                
                structure_analysis['top_allocation_changes'] = top_allocation_changes.to_dict('records')
    
    # 変化点検出結果（あれば）
    change_points = {}
    if 'change_points' in analysis_result and analysis_result['change_points']:
        cp_data = analysis_result['change_points']
        
        # 媒体レベルの変化点
        if 0 in cp_data and 'change_points' in cp_data[0]:
            top_cps = cp_data[0]['change_points'][:3]  # 上位3つの変化点
            change_points['top_media_changes'] = [cp['node'] for cp in top_cps]
    
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
        'cpa_decomposition': cpa_decomposition,
        'structure_analysis': structure_analysis,
        'change_points': change_points,
        'auto_analysis': auto_analysis
    }
    
    return formatted_data

# 期間比較のための分析関数を強化
def compare_periods(current_df, previous_df, group_by_cols=['ServiceNameJA']):
    """
    二つの期間のデータを比較して分析結果を返す（強化版）
    
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
    
    # 1. CV増減の寄与度分析（既存）
    cv_contribution = analyze_cv_contribution(current_agg, previous_agg, group_by_cols)
    
    # 2. CPA変化要因分析（既存）
    cpa_change_factors = analyze_cpa_change_factors(current_agg, previous_agg, group_by_cols)
    
    # 3. 媒体グループ・パターン分析（既存）
    media_patterns = analyze_media_patterns(current_agg, previous_agg, group_by_cols)
    
    # 4. 新規: CPA変化の寄与度分解（新機能）
    cpa_decomposition = decompose_cpa_change(current_agg, previous_agg, group_by_cols)
    
    # 5. 新規: CV変化の寄与度分解（新機能）
    cv_decomposition = decompose_cv_change(current_agg, previous_agg, group_by_cols)
    
    # 6. 新規: 構造変化分析（新機能）
    structure_analysis = analyze_structure_change(current_agg, previous_agg, group_by_cols)
    
    # 階層的変化点検出（可能な場合）
    hierarchy_cols = ['ServiceNameJA', 'CampaignName', 'AdgroupName']
    hierarchy_cols = [col for col in hierarchy_cols if col in current_df.columns and col in previous_df.columns]
    
    change_points = {}
    if len(hierarchy_cols) > 1:
        change_points = detect_change_points(current_df, previous_df, hierarchy_cols)
    
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
        'media_patterns': media_patterns,
        'cpa_decomposition': cpa_decomposition,
        'cv_decomposition': cv_decomposition,
        'structure_analysis': structure_analysis,
        'change_points': change_points
    }
    
    return result

# 分析プロンプトを作成する関数を強化
def create_analysis_prompt(data):
    """
    分析用のプロンプトを作成する（強化版）
    
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
    
    # CPA/CV変化の寄与度分解データがあれば追加
    cpa_decomposition = ""
    if 'cpa_decomposition' in data and data['cpa_decomposition']:
        cpa_decomp = data['cpa_decomposition']
        
        cpa_decomposition = "\n## CPA変化の寄与度分解\n\n"
        cpa_decomposition += "CPA変化の主な要因:\n"
        
        if 'cvr_contribution' in cpa_decomp and 'cpc_contribution' in cpa_decomp:
            cvr_contribution = cpa_decomp.get('cvr_contribution', 0)
            cpc_contribution = cpa_decomp.get('cpc_contribution', 0)
            cpa_decomposition += f"- CVR変化による寄与: {cvr_contribution:.1f}%\n"
            cpa_decomposition += f"- CPC変化による寄与: {cpc_contribution:.1f}%\n"
            
            if 'cpm_contribution' in cpa_decomp and 'ctr_contribution' in cpa_decomp:
                cpm_contribution = cpa_decomp.get('cpm_contribution', 0)
                ctr_contribution = cpa_decomp.get('ctr_contribution', 0)
                cpa_decomposition += f"  - CPM変化による寄与: {cpm_contribution:.1f}%\n"
                cpa_decomposition += f"  - CTR変化による寄与: {ctr_contribution:.1f}%\n"
    
    # 構造変化分析データがあれば追加
    structure_analysis = ""
    if 'structure_analysis' in data and data['structure_analysis']:
        structure = data['structure_analysis']
        
        structure_analysis = "\n## 構造変化分析\n\n"
        
        if 'allocation_contribution' in structure and 'performance_contribution' in structure:
            allocation_contribution = structure.get('allocation_contribution', 0)
            performance_contribution = structure.get('performance_contribution', 0)
            
            structure_analysis += "CV変化の要因分解:\n"
            structure_analysis += f"- 媒体間のコスト配分変化による寄与: {allocation_contribution:.1f}%\n"
            structure_analysis += f"- 媒体自体のパフォーマンス変化による寄与: {performance_contribution:.1f}%\n"
            
            # 重要な配分変化があれば追加
            if 'top_allocation_changes' in structure:
                top_changes = structure['top_allocation_changes']
                
                structure_analysis += "\n主要な配分変化:\n"
                for change in top_changes:
                    media_name = change.get('ServiceNameJA', 'Unknown')
                    prev_ratio = change.get('previous_cost_ratio', 0)
                    curr_ratio = change.get('current_cost_ratio', 0)
                    ratio_change = change.get('cost_ratio_change', 0)
                    
                    structure_analysis += f"- {media_name}: {prev_ratio:.1f}% → {curr_ratio:.1f}% ({ratio_change:+.1f}ポイント)\n"
    
    # 変化点検出データがあれば追加
    change_points = ""
    if 'change_points' in data and data['change_points']:
        cp_data = data['change_points']
        
        change_points = "\n## 階層的変化点\n\n"
        
        # 媒体レベルの変化点
        if 'top_media_changes' in cp_data:
            top_changes = cp_data['top_media_changes']
            
            change_points += "重要な媒体レベル変化点:\n"
            for i, node in enumerate(top_changes):
                node_name = node.get('ServiceNameJA', 'Unknown')
                change_points += f"- {node_name}\n"
    
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
| 日平均CV数 | {previous['conversions']/previous['days']:.1f} | {current['conversions']/current['days']:.1f} | {current['conversions']/current['days'] - previous['conversions']/previous['days']:.1f} | {((current['conversions']/current['days']) / (previous['conversions']/previous['days']) - 1) * 100:.1f}% |
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
    
    # プロンプトの作成（強化版）
    prompt = f"""# 広告パフォーマンス分析（強化版）

## 全体サマリー
{summary_table}

## CV増減の寄与度ランキング（上位5媒体）
{cv_table}

## CPA変化要因分析（上位5媒体）
{cpa_table}

## 媒体グループ・パターン分析
{pattern_summary}
{cpa_decomposition}
{structure_analysis}
{change_points}
{auto_analysis_summary}
---

上記の広告パフォーマンスデータを分析し、以下の内容を含む構造化されたレポートを生成してください：

1. **エグゼクティブサマリー（1-3行）**
   - 全体パフォーマンス変化の簡潔な要約
   - 例: 「CPA 3.5%改善・CV 2.1%増加。主要因はGoogle広告のCVR向上(+15%)とYahoo!のコスト配分最適化(-10%)」

2. **全体パフォーマンス変化分析**
   - 主要指標の変化状況
   - 日数差を考慮した評価
   - CPA変化の主な要因分解（CVR変化とCPC変化の寄与度）
   - CV変化の主な要因分解（インプレッション、CTR、CVRの寄与度）

3. **構造変化分析**
   - 媒体間のコスト配分変化とその影響
   - 効率と規模のバランス変化
   - 新規導入または縮小された媒体の評価

4. **主要変化点サマリー（3-5項目）**
   - 最も影響の大きかった変化要因のリスト
   - 各要因の定量的影響度と簡潔な説明

5. **重点的に見るべき問題点と機会**
   - 優先的に対応すべき3つの課題
   - 活用すべき3つの好機
   - 各項目に対する具体的な推奨アクション

以下の注意点を考慮してください：
- 単純な数値比較だけでなく、背景にある戦略的意図を考慮
- 日数の違いがある場合は、日平均値での比較も検討
- CV数が極端に少ない媒体（5件未満等）はCPA等の変動が大きくなるため解釈に注意
- 構造変化（コスト配分変更）と指標変化（CVRやCPC等）の影響を分離して評価
- 階層的な変化点（媒体→キャンペーン→広告グループ）の連鎖を意識した分析
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
    
    # 自動分析設定
    with st.sidebar.expander("自動分析設定", expanded=True):
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
    
    # タブの設定 - 新しいタブを追加
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "データ概要", 
        "期間比較分析", 
        "構造変化分析", 
        "階層的分析",
        "自動分析", 
        "レポート出力", 
        "分析手法の説明"
    ])
    
    # タブ1: データ概要 - 既存のまま
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
        
        # 指標の表示順序を変更
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
        
        # 数値フォーマットの調整
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
    
    # タブ2: 期間比較分析 - 既存のタブを拡張
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
        
        # 期間設定UI（既存のコードと同じ）
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
                
                # 期間比較分析を実行（強化版の関数を使用）
                analysis_result = compare_periods(current_df, previous_df, group_by_cols)
                
                if analysis_result:
                    st.session_state['analysis_result'] = analysis_result
                    st.session_state['previous_df'] = previous_df
                    st.session_state['current_df'] = current_df
                    st.session_state['group_by_cols'] = group_by_cols
                    st.session_state['previous_period'] = (previous_start, previous_end)
                    st.session_state['current_period'] = (current_start, current_end)
                    
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
                    
                    # 構造分析結果を計算
                    if 'structure_analysis' in analysis_result:
                        st.session_state['structural_analysis_result'] = analysis_result['structure_analysis']
                    
                    # ChatGPTによる分析結果の解釈（API Keyが設定されている場合）
                    if openai_api_key:
                        with st.spinner("ChatGPTによる分析結果の解釈中..."):
                            try:
                                interpretation = interpret_analysis_with_chatgpt(analysis_result, openai_api_key, model="gpt-4")
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
            
            # 新機能: CPA変化の寄与度分解表示
            if 'cpa_decomposition' in result:
                st.write("#### CPA変化の寄与度分解")
                
                cpa_decomp = result['cpa_decomposition'].head(5)
                
                # 表形式で表示
                cpa_decomp_display = cpa_decomp[[
                    st.session_state['group_by_cols'][0], 'previous_cpa', 'current_cpa', 
                    'cpa_change_rate', 'cvr_contribution', 'cpc_contribution'
                ]].copy()
                
                # カラム名の変更
                cpa_decomp_display.columns = [
                    '媒体名', '前期CPA', '当期CPA', 'CPA変化率(%)', 
                    'CVR寄与度(%)', 'CPC寄与度(%)'
                ]
                
                # 数値フォーマットの調整
                cpa_decomp_display = format_metrics(
                    cpa_decomp_display,
                    integer_cols=['前期CPA', '当期CPA'],
                    decimal_cols=['CPA変化率(%)', 'CVR寄与度(%)', 'CPC寄与度(%)']
                )
                
                st.dataframe(cpa_decomp_display)
            
            # 2. CV増減の寄与度ランキング
            st.write("#### CV増減の寄与度ランキング")
            
            cv_contribution = result['cv_contribution'].head(5)
            
            # 数値フォーマットの調整
            cv_contribution_formatted = format_metrics(
                cv_contribution,
                integer_cols=[],
                decimal_cols=['previous_cv', 'current_cv', 'cv_change', 'contribution_rate']
            )
            
            st.dataframe(cv_contribution_formatted)
            
            # 3. 媒体グループ・パターン分析
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
    
    # タブ3: 構造変化分析（新規タブ）
    with tab3:
        st.header("構造変化分析")
        
        if 'structural_analysis_result' not in st.session_state or not st.session_state['structural_analysis_result']:
            st.info("「期間比較分析」タブで分析を実行してください")
        else:
            structure_data = st.session_state['structural_analysis_result']
            
            # 全体の変化サマリー
            st.subheader("全体の変化サマリー")
            
            # CPA変化の要約
            total_cpa_change = structure_data['total_cpa_change']
            allocation_contribution = structure_data['allocation_contribution']
            performance_contribution = structure_data['performance_contribution']
            
            st.write(f"全体CPA変化率: {total_cpa_change:.1f}%")
            st.write(f"コスト配分変化の寄与度: {allocation_contribution:.1f}%")
            st.write(f"パフォーマンス変化の寄与度: {performance_contribution:.1f}%")
            
            # CV変化のウォーターフォールチャート
            st.subheader("CV変化の要因分解")
            
            # ウォーターフォールチャートの作成
            fig_waterfall = create_waterfall_chart(
                structure_data,
                "CV変化の要因分解（コスト配分効果 vs パフォーマンス効果）",
                structure_data['total_previous_cv'],
                ['allocation_effect_total_cv', 'performance_effect_total_cv'],
                ['rgba(255, 99, 132, 0.7)', 'rgba(75, 192, 192, 0.7)', 'rgba(54, 162, 235, 0.7)']
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # 媒体間のコスト配分変化
            st.subheader("媒体間のコスト配分変化")
            
            # 構造データの表示
            structure_df = structure_data['structure_df'].copy()
            
            # 表示用にデータを整形
            display_columns = [
                st.session_state['group_by_cols'][0], 
                'previous_cost', 'current_cost', 
                'previous_cost_ratio', 'current_cost_ratio', 'cost_ratio_change',
                'previous_cpa', 'current_cpa', 'cpa_change'
            ]
            
            display_df = structure_df[display_columns].copy()
            
            # カラム名の変更
            display_df.columns = [
                '媒体名', '前期コスト', '当期コスト', 
                '前期比率(%)', '当期比率(%)', '比率変化(pp)',
                '前期CPA', '当期CPA', 'CPA変化(%)'
            ]
            
            # 数値フォーマットの調整
            display_df = format_metrics(
                display_df,
                integer_cols=['前期コスト', '当期コスト', '前期CPA', '当期CPA'],
                decimal_cols=['前期比率(%)', '当期比率(%)', '比率変化(pp)', 'CPA変化(%)']
            )
            
            st.dataframe(display_df)
            
            # サンキーダイアグラム
            st.subheader("コスト配分変化のサンキーダイアグラム")
            
            sankey_fig = create_sankey_diagram(structure_data, st.session_state['group_by_cols'][0])
            
            st.plotly_chart(sankey_fig, use_container_width=True)
            
            # 構造変化の分析ポイント
            st.subheader("構造変化の分析ポイント")
            
            # コスト配分変化の大きい媒体の抽出
            top_allocation_changes = structure_df.sort_values('cost_ratio_change', key=abs, ascending=False).head(3)
            
            st.write("##### 主要なコスト配分変化")
            for _, row in top_allocation_changes.iterrows():
                media_name = row[st.session_state['group_by_cols'][0]]
                prev_ratio = row['previous_cost_ratio']
                curr_ratio = row['current_cost_ratio']
                ratio_change = row['cost_ratio_change']
                
                change_direction = "増加" if ratio_change > 0 else "減少"
                st.write(f"- {media_name}: {prev_ratio:.1f}% → {curr_ratio:.1f}% ({abs(ratio_change):.1f}ポイント{change_direction})")
            
            # CPAパフォーマンス変化の大きい媒体の抽出
            top_performance_changes = structure_df.sort_values('cpa_change', key=abs, ascending=False).head(3)
            
            st.write("##### 主要なパフォーマンス変化")
            for _, row in top_performance_changes.iterrows():
                media_name = row[st.session_state['group_by_cols'][0]]
                prev_cpa = row['previous_cpa']
                curr_cpa = row['current_cpa']
                cpa_change = row['cpa_change']
                
                change_direction = "悪化" if cpa_change > 0 else "改善"
                st.write(f"- {media_name}: {prev_cpa:.0f}円 → {curr_cpa:.0f}円 ({abs(cpa_change):.1f}%{change_direction})")
    
    # タブ4: 階層的分析（新規タブ）
    with tab4:
        st.header("階層的分析")
        
        if 'analysis_result' not in st.session_state or not st.session_state['analysis_result']:
            st.info("「期間比較分析」タブで分析を実行してください")
        else:
            result = st.session_state['analysis_result']
            
            # 階層的変化点検出の結果があるか確認
            if 'change_points' not in result or not result['change_points']:
                st.info("階層的変化点の検出結果がありません。データが不十分か、変化が少ない可能性があります。")
            else:
                change_points = result['change_points']
                
                # 変化点の表示
                st.subheader("重要な変化点")
                
                # レベル選択
                level_options = list(change_points.keys())
                selected_level = st.selectbox(
                    "分析レベル",
                    level_options,
                    format_func=lambda x: f"レベル{x+1}: {change_points[x]['column']}"
                )
                
                # 選択されたレベルの変化点を表示
                level_data = change_points[selected_level]
                
                # 変化点の数
                num_points = len(level_data['change_points'])
                st.write(f"検出された変化点: {num_points}件")
                
                # 表示数の選択
                num_to_show = st.slider("表示する変化点数", min_value=1, max_value=min(10, num_points), value=min(5, num_points))
                
                # 変化点の詳細表示
                for i, cp in enumerate(level_data['change_points'][:num_to_show]):
                    node = cp['node']
                    metrics = cp['metrics']
                    
                    # ノード名の表示（階層に応じた表示形式）
                    node_name = " / ".join([f"{k}: {v}" for k, v in node.items()])
                    
                    st.write(f"##### 変化点 {i+1}: {node_name}")
                    
                    # 指標の変化を表形式で表示
                    metrics_table = pd.DataFrame([
                        {"指標": "CPA", "前期": metrics.get('CPA', {}).get('previous', 0), "当期": metrics.get('CPA', {}).get('current', 0), "変化率(%)": metrics.get('CPA', {}).get('change_rate', 0)},
                        {"指標": "CVR", "前期": metrics.get('CVR', {}).get('previous', 0), "当期": metrics.get('CVR', {}).get('current', 0), "変化率(%)": metrics.get('CVR', {}).get('change_rate', 0)},
                        {"指標": "CPC", "前期": metrics.get('CPC', {}).get('previous', 0), "当期": metrics.get('CPC', {}).get('current', 0), "変化率(%)": metrics.get('CPC', {}).get('change_rate', 0)},
                        {"指標": "CTR", "前期": metrics.get('CTR', {}).get('previous', 0), "当期": metrics.get('CTR', {}).get('current', 0), "変化率(%)": metrics.get('CTR', {}).get('change_rate', 0)},
                        {"指標": "CPM", "前期": metrics.get('CPM', {}).get('previous', 0), "当期": metrics.get('CPM', {}).get('current', 0), "変化率(%)": metrics.get('CPM', {}).get('change_rate', 0)},
                        {"指標": "Conversions", "前期": metrics.get('Conversions', {}).get('previous', 0), "当期": metrics.get('Conversions', {}).get('current', 0), "変化率(%)": metrics.get('Conversions', {}).get('change_rate', 0)},
                        {"指標": "Cost", "前期": metrics.get('Cost', {}).get('previous', 0), "当期": metrics.get('Cost', {}).get('current', 0), "変化率(%)": metrics.get('Cost', {}).get('change_rate', 0)}
                    ])
                    
                    # 数値フォーマットの調整
                    metrics_table = format_metrics(
                        metrics_table,
                        integer_cols=['前期', '当期'],
                        decimal_cols=['変化率(%)']
                    )
                    
                    st.dataframe(metrics_table)
                    
                    # 変化タイミング分析ボタン
                    if st.button(f"変化タイミング分析（{node_name}）", key=f"timing_{i}"):
                        # 前期・当期のデータが存在するか確認
                        if 'previous_df' in st.session_state and 'current_df' in st.session_state and 'previous_period' in st.session_state and 'current_period' in st.session_state:
                            # データの準備
                            prev_df = st.session_state['previous_df']
                            curr_df = st.session_state['current_df']
                            all_df = pd.concat([prev_df, curr_df])
                            
                            # 期間の取得
                            prev_start, prev_end = st.session_state['previous_period']
                            curr_start, curr_end = st.session_state['current_period']
                            
                            # 分析期間の設定（前期開始から当期終了まで）
                            start_date = prev_start
                            end_date = curr_end
                            
                            # 変化タイミング分析の実行
                            timing_result = analyze_change_timing(
                                all_df, node, start_date, end_date, 'Conversions'
                            )
                            
                            if timing_result['status'] == 'success':
                                # 日次データのグラフ表示
                                daily_data = timing_result['daily_data']
                                
                                # 折れ線グラフの作成
                                fig = go.Figure()
                                
                                # 実際の値
                                fig.add_trace(go.Scatter(
                                    x=daily_data['Date'],
                                    y=daily_data['Conversions'],
                                    mode='lines+markers',
                                    name='CV数',
                                    line=dict(color='blue', width=1),
                                    marker=dict(size=4)
                                ))
                                
                                # 移動平均
                                fig.add_trace(go.Scatter(
                                    x=daily_data['Date'],
                                    y=daily_data['moving_avg'],
                                    mode='lines',
                                    name='7日移動平均',
                                    line=dict(color='red', width=2)
                                ))
                                
                                # 急激な変化の日をマーク
                                if not timing_result['significant_changes'].empty:
                                    fig.add_trace(go.Scatter(
                                        x=timing_result['significant_changes']['Date'],
                                        y=timing_result['significant_changes']['Conversions'],
                                        mode='markers',
                                        name='急激な変化',
                                        marker=dict(
                                            size=10,
                                            color='orange',
                                            symbol='star'
                                        )
                                    ))
                                
                                # 前期と当期の境界線
                                fig.add_vline(
                                    x=curr_start.timestamp() * 1000, 
                                    line_dash="dash", 
                                    line_color="gray",
                                    annotation_text="当期開始",
                                    annotation_position="top right"
                                )
                                
                                # レイアウト設定
                                fig.update_layout(
                                    title=f"{node_name}のCV数推移",
                                    xaxis_title="日付",
                                    yaxis_title="CV数",
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 変化パターンの説明
                                st.write(f"##### 変化パターン分析")
                                
                                pattern = timing_result['change_pattern']
                                if pattern == 'gradual':
                                    st.write("段階的な変化が検出されました。複数日にわたって徐々に変化しています。")
                                elif pattern == 'sudden':
                                    st.write("突発的な変化が検出されました。特定の日に急激な変化が生じています。")
                                else:
                                    st.write("安定的なパターンです。有意な変化は検出されませんでした。")
                                
                                # 有意な変化の日を表示
                                if not timing_result['significant_changes'].empty:
                                    st.write("##### 急激な変化が検出された日")
                                    
                                    sig_changes = timing_result['significant_changes'].copy()
                                    sig_changes['change_rate'] = sig_changes['change_rate'].round(1)
                                    
                                    for _, row in sig_changes.iterrows():
                                        date = row['Date'].strftime('%Y-%m-%d')
                                        value = row['Conversions']
                                        change = row['change_rate']
                                        
                                        direction = "増加" if change > 0 else "減少"
                                        st.write(f"- {date}: CV数 {value:.1f} ({abs(change):.1f}% {direction})")
                            else:
                                st.warning(timing_result['message'])
                        else:
                            st.warning("分析に必要なデータが不足しています。期間比較分析を先に実行してください。")
                
                # 指標変化のヒートマップ
                st.subheader("指標変化のヒートマップ")
                
                if 'cpa_decomposition' in result and 'cv_decomposition' in result:
                    # ヒートマップの作成
                    heatmap_fig = create_metric_heatmap(
                        result['cpa_decomposition'],
                        result['cv_decomposition'],
                        st.session_state['group_by_cols'][0]
                    )
                    
                    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # タブ5: 自動分析（既存タブの強化）
    with tab5:
        st.header("自動分析")
        
        if 'auto_analysis_result' not in st.session_state or not st.session_state['auto_analysis_result']:
            st.info("「期間比較分析」タブで自動分析を有効にして分析を実行してください")
        else:
            auto_result = st.session_state['auto_analysis_result']
            important_media = auto_result['important_media']
            
            # エグゼクティブサマリー（新機能）
            st.subheader("エグゼクティブサマリー")
            
            # 分析結果から自動的にサマリーを生成
            if 'analysis_result' in st.session_state:
                result = st.session_state['analysis_result']
                
                # 全体CV、CPA変化
                current_total = result['current_total']
                previous_total = result['previous_total']
                
                cv_change = ((current_total['Conversions'] - previous_total['Conversions']) / previous_total['Conversions']) * 100 if previous_total['Conversions'] != 0 else 0
                
                previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] != 0 else 0
                current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] != 0 else 0
                cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else 0
                
                # CPA変化の主要因
                cpa_direction = "改善" if cpa_change < 0 else "悪化"
                cv_direction = "増加" if cv_change > 0 else "減少"
                
                # 主要な変化要因を特定
                summary_parts = []
                
                # CPA/CV変化の基本情報
                summary_parts.append(f"CPA {abs(cpa_change):.1f}%{cpa_direction}・CV {abs(cv_change):.1f}%{cv_direction}。")
                
                # 主要因の追加
                if 'cpa_decomposition' in result:
                    cpa_decomp = result['cpa_decomposition'].iloc[0] if not result['cpa_decomposition'].empty else None
                    
                    if cpa_decomp is not None:
                        cvr_contribution = cpa_decomp['cvr_contribution']
                        cpc_contribution = cpa_decomp['cpc_contribution']
                        
                        # CPA変化の主要因
                        if abs(cvr_contribution) > abs(cpc_contribution):
                            contribution_str = f"CVR変化({cvr_contribution:.1f}%)"
                        else:
                            contribution_str = f"CPC変化({cpc_contribution:.1f}%)"
                        
                        summary_parts.append(f"CPA変化の主要因は{contribution_str}。")
                
                # 構造変化の寄与度
                if 'structure_analysis' in result:
                    structure = result['structure_analysis']
                    
                    allocation_contribution = structure['allocation_contribution']
                    performance_contribution = structure['performance_contribution']
                    
                    # 構造変化の主要因
                    if abs(allocation_contribution) > abs(performance_contribution):
                        structure_str = f"媒体間コスト配分変化({allocation_contribution:.1f}%)"
                    else:
                        structure_str = f"媒体自体のパフォーマンス変化({performance_contribution:.1f}%)"
                    
                    summary_parts.append(f"CV変化の主要因は{structure_str}。")
                
                # 重要な媒体の変化
                if important_media['high_cv_contribution'] and len(important_media['high_cv_contribution']) > 0:
                    top_media = important_media['high_cv_contribution'][0]
                    media_name = top_media['media_name']
                    contribution = top_media['contribution_rate']
                    
                    summary_parts.append(f"{media_name}のCV寄与率({contribution:.1f}%)が最大。")
                
                # サマリーの結合
                exec_summary = " ".join(summary_parts)
                
                st.info(exec_summary)
            
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
                
                # 数値フォーマットの調整
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
                
                # 数値フォーマットの調整
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
                    
                    # 数値フォーマットの調整
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
                    
                    # 数値フォーマットの調整
                    issue_df_formatted = format_metrics(
                        issue_df,
                        integer_cols=['CPA変化'],
                        decimal_cols=['CV変化']
                    )
                    
                    st.dataframe(issue_df_formatted)
                else:
                    st.info("課題パターンの媒体はありません")
            
            with pattern_tabs[2]:
                # 成長重視パターンの媒体を抽出
                if 'analysis_result' in st.session_state:
                    result = st.session_state['analysis_result']
                    patterns = result['media_patterns']['pattern_df']
                    growth_patterns = patterns[patterns['pattern'] == 'growth']
                    
                    if not growth_patterns.empty:
                        growth_data = []
                        for _, row in growth_patterns.iterrows():
                            growth_data.append({
                                '媒体名': row[st.session_state['group_by_cols'][0]],
                                'CV変化': row['cv_change'],
                                'CPA変化': row['cpa_change']
                            })
                        
                        growth_df = pd.DataFrame(growth_data)
                        
                        # 数値フォーマットの調整
                        growth_df_formatted = format_metrics(
                            growth_df,
                            integer_cols=['CPA変化'],
                            decimal_cols=['CV変化']
                        )
                        
                        st.dataframe(growth_df_formatted)
                    else:
                        st.info("成長重視パターンの媒体はありません")
                else:
                    st.info("分析結果がありません")
            
            with pattern_tabs[3]:
                # 効率重視パターンの媒体を抽出
                if 'analysis_result' in st.session_state:
                    result = st.session_state['analysis_result']
                    patterns = result['media_patterns']['pattern_df']
                    efficiency_patterns = patterns[patterns['pattern'] == 'efficiency']
                    
                    if not efficiency_patterns.empty:
                        efficiency_data = []
                        for _, row in efficiency_patterns.iterrows():
                            efficiency_data.append({
                                '媒体名': row[st.session_state['group_by_cols'][0]],
                                'CV変化': row['cv_change'],
                                'CPA変化': row['cpa_change']
                            })
                        
                        efficiency_df = pd.DataFrame(efficiency_data)
                        
                        # 数値フォーマットの調整
                        efficiency_df_formatted = format_metrics(
                            efficiency_df,
                            integer_cols=['CPA変化'],
                            decimal_cols=['CV変化']
                        )
                        
                        st.dataframe(efficiency_df_formatted)
                    else:
                        st.info("効率重視パターンの媒体はありません")
                else:
                    st.info("分析結果がありません")
            
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
                    
                    # 数値フォーマットの調整
                    campaign_cv_formatted = format_metrics(
                        campaign_cv[['CampaignName', 'previous_cv', 'current_cv', 'cv_change', 'contribution_rate']],
                        integer_cols=[],
                        decimal_cols=['previous_cv', 'current_cv', 'cv_change', 'contribution_rate']
                    )
                    
                    st.dataframe(campaign_cv_formatted)
                    
                    # キャンペーンレベルのCPA変化要因
                    st.write("**キャンペーンレベルのCPA変化要因**")
                    campaign_cpa = campaign_result['cpa_change_factors'].head(5)
                    
                    # CPA変化要因の詳細情報と数値フォーマットの調整
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
    

# 2. CPA変化要因分析
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

# 数値フォーマット関数
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
    
    # CPA変化の寄与度分解（あれば）
    cpa_decomposition = {}
    if 'cpa_decomposition' in analysis_result and not analysis_result['cpa_decomposition'].empty:
        cpa_decomp_df = analysis_result['cpa_decomposition'].head(5)
        
        # 平均的な寄与度を計算
        cvr_contribution = cpa_decomp_df['cvr_contribution'].mean()
        cpc_contribution = cpa_decomp_df['cpc_contribution'].mean()
        
        if 'cpm_contribution' in cpa_decomp_df.columns and 'ctr_contribution' in cpa_decomp_df.columns:
            cpm_contribution = cpa_decomp_df['cpm_contribution'].mean()
            ctr_contribution = cpa_decomp_df['ctr_contribution'].mean()
            
            cpa_decomposition = {
                'cvr_contribution': cvr_contribution,
                'cpc_contribution': cpc_contribution,
                'cpm_contribution': cpm_contribution,
                'ctr_contribution': ctr_contribution
            }
        else:
            cpa_decomposition = {
                'cvr_contribution': cvr_contribution,
                'cpc_contribution': cpc_contribution
            }
    
    # 構造変化分析（あれば）
    structure_analysis = {}
    if 'structure_analysis' in analysis_result:
        structure = analysis_result['structure_analysis']
        
        if 'allocation_contribution' in structure and 'performance_contribution' in structure:
            structure_analysis = {
                'allocation_contribution': structure['allocation_contribution'],
                'performance_contribution': structure['performance_contribution']
            }
            
            # 重要な構造変化があれば追加
            if 'structure_df' in structure:
                structure_df = structure['structure_df']
                
                # コスト配分変化の大きい上位3媒体
                top_allocation_changes = structure_df.sort_values('cost_ratio_change', key=abs, ascending=False).head(3)
                
                structure_analysis['top_allocation_changes'] = top_allocation_changes.to_dict('records')
    
    # 変化点検出結果（あれば）
    change_points = {}
    if 'change_points' in analysis_result and analysis_result['change_points']:
        cp_data = analysis_result['change_points']
        
        # 媒体レベルの変化点
        if 0 in cp_data and 'change_points' in cp_data[0]:
            top_cps = cp_data[0]['change_points'][:3]  # 上位3つの変化点
            change_points['top_media_changes'] = [cp['node'] for cp in top_cps]
    
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
        'cpa_decomposition': cpa_decomposition,
        'structure_analysis': structure_analysis,
        'change_points': change_points,
        'auto_analysis': auto_analysis
    }
    
    return formatted_data

# 期間比較のための分析関数を強化
def compare_periods(current_df, previous_df, group_by_cols=['ServiceNameJA']):
    """
    二つの期間のデータを比較して分析結果を返す（強化版）
    
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
    
    # 1. CV増減の寄与度分析（既存）
    cv_contribution = analyze_cv_contribution(current_agg, previous_agg, group_by_cols)
    
    # 2. CPA変化要因分析（既存）
    cpa_change_factors = analyze_cpa_change_factors(current_agg, previous_agg, group_by_cols)
    
    # 3. 媒体グループ・パターン分析（既存）
    media_patterns = analyze_media_patterns(current_agg, previous_agg, group_by_cols)
    
    # 4. 新規: CPA変化の寄与度分解（新機能）
    cpa_decomposition = decompose_cpa_change(current_agg, previous_agg, group_by_cols)
    
    # 5. 新規: CV変化の寄与度分解（新機能）
    cv_decomposition = decompose_cv_change(current_agg, previous_agg, group_by_cols)
    
    # 6. 新規: 構造変化分析（新機能）
    structure_analysis = analyze_structure_change(current_agg, previous_agg, group_by_cols)
    
    # 階層的変化点検出（可能な場合）
    hierarchy_cols = ['ServiceNameJA', 'CampaignName', 'AdgroupName']
    hierarchy_cols = [col for col in hierarchy_cols if col in current_df.columns and col in previous_df.columns]
    
    change_points = {}
    if len(hierarchy_cols) > 1:
        change_points = detect_change_points(current_df, previous_df, hierarchy_cols)
    
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
        'media_patterns': media_patterns,
        'cpa_decomposition': cpa_decomposition,
        'cv_decomposition': cv_decomposition,
        'structure_analysis': structure_analysis,
        'change_points': change_points
    }
    
    return result

# 分析プロンプトを作成する関数を強化
def create_analysis_prompt(data):
    """
    分析用のプロンプトを作成する（強化版）
    
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
    
    # CPA/CV変化の寄与度分解データがあれば追加
    cpa_decomposition = ""
    if 'cpa_decomposition' in data and data['cpa_decomposition']:
        cpa_decomp = data['cpa_decomposition']
        
        cpa_decomposition = "\n## CPA変化の寄与度分解\n\n"
        cpa_decomposition += "CPA変化の主な要因:\n"
        
        if 'cvr_contribution' in cpa_decomp and 'cpc_contribution' in cpa_decomp:
            cvr_contribution = cpa_decomp.get('cvr_contribution', 0)
            cpc_contribution = cpa_decomp.get('cpc_contribution', 0)
            cpa_decomposition += f"- CVR変化による寄与: {cvr_contribution:.1f}%\n"
            cpa_decomposition += f"- CPC変化による寄与: {cpc_contribution:.1f}%\n"
            
            if 'cpm_contribution' in cpa_decomp and 'ctr_contribution' in cpa_decomp:
                cpm_contribution = cpa_decomp.get('cpm_contribution', 0)
                ctr_contribution = cpa_decomp.get('ctr_contribution', 0)
                cpa_decomposition += f"  - CPM変化による寄与: {cpm_contribution:.1f}%\n"
                cpa_decomposition += f"  - CTR変化による寄与: {ctr_contribution:.1f}%\n"
    
    # 構造変化分析データがあれば追加
    structure_analysis = ""
    if 'structure_analysis' in data and data['structure_analysis']:
        structure = data['structure_analysis']
        
        structure_analysis = "\n## 構造変化分析\n\n"
        
        if 'allocation_contribution' in structure and 'performance_contribution' in structure:
            allocation_contribution = structure.get('allocation_contribution', 0)
            performance_contribution = structure.get('performance_contribution', 0)
            
            structure_analysis += "CV変化の要因分解:\n"
            structure_analysis += f"- 媒体間のコスト配分変化による寄与: {allocation_contribution:.1f}%\n"
            structure_analysis += f"- 媒体自体のパフォーマンス変化による寄与: {performance_contribution:.1f}%\n"
            
            # 重要な配分変化があれば追加
            if 'top_allocation_changes' in structure:
                top_changes = structure['top_allocation_changes']
                
                structure_analysis += "\n主要な配分変化:\n"
                for change in top_changes:
                    media_name = change.get('ServiceNameJA', 'Unknown')
                    prev_ratio = change.get('previous_cost_ratio', 0)
                    curr_ratio = change.get('current_cost_ratio', 0)
                    ratio_change = change.get('cost_ratio_change', 0)
                    
                    structure_analysis += f"- {media_name}: {prev_ratio:.1f}% → {curr_ratio:.1f}% ({ratio_change:+.1f}ポイント)\n"
    
    # 変化点検出データがあれば追加
    change_points = ""
    if 'change_points' in data and data['change_points']:
        cp_data = data['change_points']
        
        change_points = "\n## 階層的変化点\n\n"
        
        # 媒体レベルの変化点
        if 'top_media_changes' in cp_data:
            top_changes = cp_data['top_media_changes']
            
            change_points += "重要な媒体レベル変化点:\n"
            for i, node in enumerate(top_changes):
                node_name = node.get('ServiceNameJA', 'Unknown')
                change_points += f"- {node_name}\n"
    
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
| 日平均CV数 | {previous['conversions']/previous['days']:.1f} | {current['conversions']/current['days']:.1f} | {current['conversions']/current['days'] - previous['conversions']/previous['days']:.1f} | {((current['conversions']/current['days']) / (previous['conversions']/previous['days']) - 1) * 100:.1f}% |
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
    
    # プロンプトの作成（強化版）
    prompt = f"""# 広告パフォーマンス分析（強化版）

## 全体サマリー
{summary_table}

## CV増減の寄与度ランキング（上位5媒体）
{cv_table}

## CPA変化要因分析（上位5媒体）
{cpa_table}

## 媒体グループ・パターン分析
{pattern_summary}
{cpa_decomposition}
{structure_analysis}
{change_points}
{auto_analysis_summary}
---

上記の広告パフォーマンスデータを分析し、以下の内容を含む構造化されたレポートを生成してください：

1. **エグゼクティブサマリー（1-3行）**
   - 全体パフォーマンス変化の簡潔な要約
   - 例: 「CPA 3.5%改善・CV 2.1%増加。主要因はGoogle広告のCVR向上(+15%)とYahoo!のコスト配分最適化(-10%)」

2. **全体パフォーマンス変化分析**
   - 主要指標の変化状況
   - 日数差を考慮した評価
   - CPA変化の主な要因分解（CVR変化とCPC変化の寄与度）
   - CV変化の主な要因分解（インプレッション、CTR、CVRの寄与度）

3. **構造変化分析**
   - 媒体間のコスト配分変化とその影響
   - 効率と規模のバランス変化
   - 新規導入または縮小された媒体の評価

4. **主要変化点サマリー（3-5項目）**
   - 最も影響の大きかった変化要因のリスト
   - 各要因の定量的影響度と簡潔な説明

5. **重点的に見るべき問題点と機会**
   - 優先的に対応すべき3つの課題
   - 活用すべき3つの好機
   - 各項目に対する具体的な推奨アクション

以下の注意点を考慮してください：
- 単純な数値比較だけでなく、背景にある戦略的意図を考慮
- 日数の違いがある場合は、日平均値での比較も検討
- CV数が極端に少ない媒体（5件未満等）はCPA等の変動が大きくなるため解釈に注意
- 構造変化（コスト配分変更）と指標変化（CVRやCPC等）の影響を分離して評価
- 階層的な変化点（媒体→キャンペーン→広告グループ）の連鎖を意識した分析
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
    
    # 自動分析設定
    with st.sidebar.expander("自動分析設定", expanded=True):
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
    
    # タブの設定 - 新しいタブを追加
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "データ概要", 
        "期間比較分析", 
        "構造変化分析", 
        "階層的分析",
        "自動分析", 
        "レポート出力", 
        "分析手法の説明"
    ])
    
    # タブ1: データ概要 - 既存のまま
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
        
        # 指標の表示順序を変更
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
        
        # 数値フォーマットの調整
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
    
    # タブ2: 期間比較分析 - 既存のタブを拡張
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
        
        # 期間設定UI（既存のコードと同じ）
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
                
                # 期間比較分析を実行（強化版の関数を使用）
                analysis_result = compare_periods(current_df, previous_df, group_by_cols)
                
                if analysis_result:
                    st.session_state['analysis_result'] = analysis_result
                    st.session_state['previous_df'] = previous_df
                    st.session_state['current_df'] = current_df
                    st.session_state['group_by_cols'] = group_by_cols
                    st.session_state['previous_period'] = (previous_start, previous_end)
                    st.session_state['current_period'] = (current_start, current_end)
                    
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
                    
                    # 構造分析結果を計算
                    if 'structure_analysis' in analysis_result:
                        st.session_state['structural_analysis_result'] = analysis_result['structure_analysis']
                    
                    # ChatGPTによる分析結果の解釈（API Keyが設定されている場合）
                    if openai_api_key:
                        with st.spinner("ChatGPTによる分析結果の解釈中..."):
                            try:
                                interpretation = interpret_analysis_with_chatgpt(analysis_result, openai_api_key, model="gpt-4")
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
            
            # 新機能: CPA変化の寄与度分解表示
            if 'cpa_decomposition' in result:
                st.write("#### CPA変化の寄与度分解")
                
                cpa_decomp = result['cpa_decomposition'].head(5)
                
                # 表形式で表示
                cpa_decomp_display = cpa_decomp[[
                    st.session_state['group_by_cols'][0], 'previous_cpa', 'current_cpa', 
                    'cpa_change_rate', 'cvr_contribution', 'cpc_contribution'
                ]].copy()
                
                # カラム名の変更
                cpa_decomp_display.columns = [
                    '媒体名', '前期CPA', '当期CPA', 'CPA変化率(%)', 
                    'CVR寄与度(%)', 'CPC寄与度(%)'
                ]
                
                # 数値フォーマットの調整
                cpa_decomp_display = format_metrics(
                    cpa_decomp_display,
                    integer_cols=['前期CPA', '当期CPA'],
                    decimal_cols=['CPA変化率(%)', 'CVR寄与度(%)', 'CPC寄与度(%)']
                )
                
                st.dataframe(cpa_decomp_display)
            
            # 2. CV増減の寄与度ランキング
            st.write("#### CV増減の寄与度ランキング")
            
            cv_contribution = result['cv_contribution'].head(5)
            
            # 数値フォーマットの調整
            cv_contribution_formatted = format_metrics(
                cv_contribution,
                integer_cols=[],
                decimal_cols=['previous_cv', 'current_cv', 'cv_change', 'contribution_rate']
            )
            
            st.dataframe(cv_contribution_formatted)
            
            # 3. 媒体グループ・パターン分析
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
    
    # タブ3: 構造変化分析（新規タブ）
    with tab3:
        st.header("構造変化分析")
        
        if 'structural_analysis_result' not in st.session_state or not st.session_state['structural_analysis_result']:
            st.info("「期間比較分析」タブで分析を実行してください")
        else:
            structure_data = st.session_state['structural_analysis_result']
            
            # 全体の変化サマリー
            st.subheader("全体の変化サマリー")
            
            # CPA変化の要約
            total_cpa_change = structure_data['total_cpa_change']
            allocation_contribution = structure_data['allocation_contribution']
            performance_contribution = structure_data['performance_contribution']
            
            st.write(f"全体CPA変化率: {total_cpa_change:.1f}%")
            st.write(f"コスト配分変化の寄与度: {allocation_contribution:.1f}%")
            st.write(f"パフォーマンス変化の寄与度: {performance_contribution:.1f}%")
            
            # CV変化のウォーターフォールチャート
            st.subheader("CV変化の要因分解")
            
            # ウォーターフォールチャートの作成
            fig_waterfall = create_waterfall_chart(
                structure_data,
                "CV変化の要因分解（コスト配分効果 vs パフォーマンス効果）",
                structure_data['total_previous_cv'],
                ['allocation_effect_total_cv', 'performance_effect_total_cv'],
                ['rgba(255, 99, 132, 0.7)', 'rgba(75, 192, 192, 0.7)', 'rgba(54, 162, 235, 0.7)']
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            # 媒体間のコスト配分変化
            st.subheader("媒体間のコスト配分変化")
            
            # 構造データの表示
            structure_df = structure_data['structure_df'].copy()
            
            # 表示用にデータを整形
            display_columns = [
                st.session_state['group_by_cols'][0], 
                'previous_cost', 'current_cost', 
                'previous_cost_ratio', 'current_cost_ratio', 'cost_ratio_change',
                'previous_cpa', 'current_cpa', 'cpa_change'
            ]
            
            display_df = structure_df[display_columns].copy()
            
            # カラム名の変更
            display_df.columns = [
                '媒体名', '前期コスト', '当期コスト', 
                '前期比率(%)', '当期比率(%)', '比率変化(pp)',
                '前期CPA', '当期CPA', 'CPA変化(%)'
            ]
            
            # 数値フォーマットの調整
            display_df = format_metrics(
                display_df,
                integer_cols=['前期コスト', '当期コスト', '前期CPA', '当期CPA'],
                decimal_cols=['前期比率(%)', '当期比率(%)', '比率変化(pp)', 'CPA変化(%)']
            )
            
            st.dataframe(display_df)
            
            # サンキーダイアグラム
            st.subheader("コスト配分変化のサンキーダイアグラム")
            
            sankey_fig = create_sankey_diagram(structure_data, st.session_state['group_by_cols'][0])
            
            st.plotly_chart(sankey_fig, use_container_width=True)
            
            # 構造変化の分析ポイント
            st.subheader("構造変化の分析ポイント")
            
            # コスト配分変化の大きい媒体の抽出
            top_allocation_changes = structure_df.sort_values('cost_ratio_change', key=abs, ascending=False).head(3)
            
            st.write("##### 主要なコスト配分変化")
            for _, row in top_allocation_changes.iterrows():
                media_name = row[st.session_state['group_by_cols'][0]]
                prev_ratio = row['previous_cost_ratio']
                curr_ratio = row['current_cost_ratio']
                ratio_change = row['cost_ratio_change']
                
                change_direction = "増加" if ratio_change > 0 else "減少"
                st.write(f"- {media_name}: {prev_ratio:.1f}% → {curr_ratio:.1f}% ({abs(ratio_change):.1f}ポイント{change_direction})")
            
            # CPAパフォーマンス変化の大きい媒体の抽出
            top_performance_changes = structure_df.sort_values('cpa_change', key=abs, ascending=False).head(3)
            
            st.write("##### 主要なパフォーマンス変化")
            for _, row in top_performance_changes.iterrows():
                media_name = row[st.session_state['group_by_cols'][0]]
                prev_cpa = row['previous_cpa']
                curr_cpa = row['current_cpa']
                cpa_change = row['cpa_change']
                
                change_direction = "悪化" if cpa_change > 0 else "改善"
                st.write(f"- {media_name}: {prev_cpa:.0f}円 → {curr_cpa:.0f}円 ({abs(cpa_change):.1f}%{change_direction})")
    
    # タブ4: 階層的分析（新規タブ）
    with tab4:
        st.header("階層的分析")
        
        if 'analysis_result' not in st.session_state or not st.session_state['analysis_result']:
            st.info("「期間比較分析」タブで分析を実行してください")
        else:
            result = st.session_state['analysis_result']
            
            # 階層的変化点検出の結果があるか確認
            if 'change_points' not in result or not result['change_points']:
                st.info("階層的変化点の検出結果がありません。データが不十分か、変化が少ない可能性があります。")
            else:
                change_points = result['change_points']
                
                # 変化点の表示
                st.subheader("重要な変化点")
                
                # レベル選択
                level_options = list(change_points.keys())
                selected_level = st.selectbox(
                    "分析レベル",
                    level_options,
                    format_func=lambda x: f"レベル{x+1}: {change_points[x]['column']}"
                )
                
                # 選択されたレベルの変化点を表示
                level_data = change_points[selected_level]
                
                # 変化点の数
                num_points = len(level_data['change_points'])
                st.write(f"検出された変化点: {num_points}件")
                
                # 表示数の選択
                num_to_show = st.slider("表示する変化点数", min_value=1, max_value=min(10, num_points), value=min(5, num_points))
                
                # 変化点の詳細表示
                for i, cp in enumerate(level_data['change_points'][:num_to_show]):
                    node = cp['node']
                    metrics = cp['metrics']
                    
                    # ノード名の表示（階層に応じた表示形式）
                    node_name = " / ".join([f"{k}: {v}" for k, v in node.items()])
                    
                    st.write(f"##### 変化点 {i+1}: {node_name}")
                    
                    # 指標の変化を表形式で表示
                    metrics_table = pd.DataFrame([
                        {"指標": "CPA", "前期": metrics.get('CPA', {}).get('previous', 0), "当期": metrics.get('CPA', {}).get('current', 0), "変化率(%)": metrics.get('CPA', {}).get('change_rate', 0)},
                        {"指標": "CVR", "前期": metrics.get('CVR', {}).get('previous', 0), "当期": metrics.get('CVR', {}).get('current', 0), "変化率(%)": metrics.get('CVR', {}).get('change_rate', 0)},
                        {"指標": "CPC", "前期": metrics.get('CPC', {}).get('previous', 0), "当期": metrics.get('CPC', {}).get('current', 0), "変化率(%)": metrics.get('CPC', {}).get('change_rate', 0)},
                        {"指標": "CTR", "前期": metrics.get('CTR', {}).get('previous', 0), "当期": metrics.get('CTR', {}).get('current', 0), "変化率(%)": metrics.get('CTR', {}).get('change_rate', 0)},
                        {"指標": "CPM", "前期": metrics.get('CPM', {}).get('previous', 0), "当期": metrics.get('CPM', {}).get('current', 0), "変化率(%)": metrics.get('CPM', {}).get('change_rate', 0)},
                        {"指標": "Conversions", "前期": metrics.get('Conversions', {}).get('previous', 0), "当期": metrics.get('Conversions', {}).get('current', 0), "変化率(%)": metrics.get('Conversions', {}).get('change_rate', 0)},
                        {"指標": "Cost", "前期": metrics.get('Cost', {}).get('previous', 0), "当期": metrics.get('Cost', {}).get('current', 0), "変化率(%)": metrics.get('Cost', {}).get('change_rate', 0)}
                    ])
                    
                    # 数値フォーマットの調整
                    metrics_table = format_metrics(
                        metrics_table,
                        integer_cols=['前期', '当期'],
                        decimal_cols=['変化率(%)']
                    )
                    
                    st.dataframe(metrics_table)
                    
                    # 変化タイミング分析ボタン
                    if st.button(f"変化タイミング分析（{node_name}）", key=f"timing_{i}"):
                        # 前期・当期のデータが存在するか確認
                        if 'previous_df' in st.session_state and 'current_df' in st.session_state and 'previous_period' in st.session_state and 'current_period' in st.session_state:
                            # データの準備
                            prev_df = st.session_state['previous_df']
                            curr_df = st.session_state['current_df']
                            all_df = pd.concat([prev_df, curr_df])
                            
                            # 期間の取得
                            prev_start, prev_end = st.session_state['previous_period']
                            curr_start, curr_end = st.session_state['current_period']
                            
                            # 分析期間の設定（前期開始から当期終了まで）
                            start_date = prev_start
                            end_date = curr_end
                            
                            # 変化タイミング分析の実行
                            timing_result = analyze_change_timing(
                                all_df, node, start_date, end_date, 'Conversions'
                            )
                            
                            if timing_result['status'] == 'success':
                                # 日次データのグラフ表示
                                daily_data = timing_result['daily_data']
                                
                                # 折れ線グラフの作成
                                fig = go.Figure()
                                
                                # 実際の値
                                fig.add_trace(go.Scatter(
                                    x=daily_data['Date'],
                                    y=daily_data['Conversions'],
                                    mode='lines+markers',
                                    name='CV数',
                                    line=dict(color='blue', width=1),
                                    marker=dict(size=4)
                                ))
                                
                                # 移動平均
                                fig.add_trace(go.Scatter(
                                    x=daily_data['Date'],
                                    y=daily_data['moving_avg'],
                                    mode='lines',
                                    name='7日移動平均',
                                    line=dict(color='red', width=2)
                                ))
                                
                                # 急激な変化の日をマーク
                                if not timing_result['significant_changes'].empty:
                                    fig.add_trace(go.Scatter(
                                        x=timing_result['significant_changes']['Date'],
                                        y=timing_result['significant_changes']['Conversions'],
                                        mode='markers',
                                        name='急激な変化',
                                        marker=dict(
                                            size=10,
                                            color='orange',
                                            symbol='star'
                                        )
                                    ))
                                
                                # 前期と当期の境界線
                                fig.add_vline(
                                    x=curr_start.timestamp() * 1000, 
                                    line_dash="dash", 
                                    line_color="gray",
                                    annotation_text="当期開始",
                                    annotation_position="top right"
                                )
                                
                                # レイアウト設定
                                fig.update_layout(
                                    title=f"{node_name}のCV数推移",
                                    xaxis_title="日付",
                                    yaxis_title="CV数",
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 変化パターンの説明
                                st.write(f"##### 変化パターン分析")
                                
                                pattern = timing_result['change_pattern']
                                if pattern == 'gradual':
                                    st.write("段階的な変化が検出されました。複数日にわたって徐々に変化しています。")
                                elif pattern == 'sudden':
                                    st.write("突発的な変化が検出されました。特定の日に急激な変化が生じています。")
                                else:
                                    st.write("安定的なパターンです。有意な変化は検出されませんでした。")
                                
                                # 有意な変化の日を表示
                                if not timing_result['significant_changes'].empty:
                                    st.write("##### 急激な変化が検出された日")
                                    
                                    sig_changes = timing_result['significant_changes'].copy()
                                    sig_changes['change_rate'] = sig_changes['change_rate'].round(1)
                                    
                                    for _, row in sig_changes.iterrows():
                                        date = row['Date'].strftime('%Y-%m-%d')
                                        value = row['Conversions']
                                        change = row['change_rate']
                                        
                                        direction = "増加" if change > 0 else "減少"
                                        st.write(f"- {date}: CV数 {value:.1f} ({abs(change):.1f}% {direction})")
                            else:
                                st.warning(timing_result['message'])
                        else:
                            st.warning("分析に必要なデータが不足しています。期間比較分析を先に実行してください。")
                
                # 指標変化のヒートマップ
                st.subheader("指標変化のヒートマップ")
                
                if 'cpa_decomposition' in result and 'cv_decomposition' in result:
                    # ヒートマップの作成
                    heatmap_fig = create_metric_heatmap(
                        result['cpa_decomposition'],
                        result['cv_decomposition'],
                        st.session_state['group_by_cols'][0]
                    )
                    
                    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # タブ5: 自動分析（既存タブの強化）
    with tab5:
        st.header("自動分析")
        
        if 'auto_analysis_result' not in st.session_state or not st.session_state['auto_analysis_result']:
            st.info("「期間比較分析」タブで自動分析を有効にして分析を実行してください")
        else:
            auto_result = st.session_state['auto_analysis_result']
            important_media = auto_result['important_media']
            
            # エグゼクティブサマリー（新機能）
            st.subheader("エグゼクティブサマリー")
            
            # 分析結果から自動的にサマリーを生成
            if 'analysis_result' in st.session_state:
                result = st.session_state['analysis_result']
                
                # 全体CV、CPA変化
                current_total = result['current_total']
                previous_total = result['previous_total']
                
                cv_change = ((current_total['Conversions'] - previous_total['Conversions']) / previous_total['Conversions']) * 100 if previous_total['Conversions'] != 0 else 0
                
                previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] != 0 else 0
                current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] != 0 else 0
                cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else 0
                
                # CPA変化の主要因
                cpa_direction = "改善" if cpa_change < 0 else "悪化"
                cv_direction = "増加" if cv_change > 0 else "減少"
                
                # 主要な変化要因を特定
                summary_parts = []
                
                # CPA/CV変化の基本情報
                summary_parts.append(f"CPA {abs(cpa_change):.1f}%{cpa_direction}・CV {abs(cv_change):.1f}%{cv_direction}。")
                
                # 主要因の追加
                if 'cpa_decomposition' in result:
                    cpa_decomp = result['cpa_decomposition'].iloc[0] if not result['cpa_decomposition'].empty else None
                    
                    if cpa_decomp is not None:
                        cvr_contribution = cpa_decomp['cvr_contribution']
                        cpc_contribution = cpa_decomp['cpc_contribution']
                        
                        # CPA変化の主要因
                        if abs(cvr_contribution) > abs(cpc_contribution):
                            contribution_str = f"CVR変化({cvr_contribution:.1f}%)"
                        else:
                            contribution_str = f"CPC変化({cpc_contribution:.1f}%)"
                        
                        summary_parts.append(f"CPA変化の主要因は{contribution_str}。")
                
                # 構造変化の寄与度
                if 'structure_analysis' in result:
                    structure = result['structure_analysis']
                    
                    allocation_contribution = structure['allocation_contribution']
                    performance_contribution = structure['performance_contribution']
                    
                    # 構造変化の主要因
                    if abs(allocation_contribution) > abs(performance_contribution):
                        structure_str = f"媒体間コスト配分変化({allocation_contribution:.1f}%)"
                    else:
                        structure_str = f"媒体自体のパフォーマンス変化({performance_contribution:.1f}%)"
                    
                    summary_parts.append(f"CV変化の主要因は{structure_str}。")
                
                # 重要な媒体の変化
                if important_media['high_cv_contribution'] and len(important_media['high_cv_contribution']) > 0:
                    top_media = important_media['high_cv_contribution'][0]
                    media_name = top_media['media_name']
                    contribution = top_media['contribution_rate']
                    
                    summary_parts.append(f"{media_name}のCV寄与率({contribution:.1f}%)が最大。")
                
                # サマリーの結合
                exec_summary = " ".join(summary_parts)
                
                st.info(exec_summary)
            
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
                
                # 数値フォーマットの調整
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
                
                # 数値フォーマットの調整
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
                    
                    # 数値フォーマットの調整
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
                    
                    # 数値フォーマットの調整
                    issue_df_formatted = format_metrics(
                        issue_df,
                        integer_cols=['CPA変化'],
                        decimal_cols=['CV変化']
                    )
                    
                    st.dataframe(issue_df_formatted)
                else:
                    st.info("課題パターンの媒体はありません")
            
            with pattern_tabs[2]:
                # 成長重視パターンの媒体を抽出
                if 'analysis_result' in st.session_state:
                    result = st.session_state['analysis_result']
                    patterns = result['media_patterns']['pattern_df']
                    growth_patterns = patterns[patterns['pattern'] == 'growth']
                    
                    if not growth_patterns.empty:
                        growth_data = []
                        for _, row in growth_patterns.iterrows():
                            growth_data.append({
                                '媒体名': row[st.session_state['group_by_cols'][0]],
                                'CV変化': row['cv_change'],
                                'CPA変化': row['cpa_change']
                            })
                        
                        growth_df = pd.DataFrame(growth_data)
                        
                        # 数値フォーマットの調整
                        growth_df_formatted = format_metrics(
                            growth_df,
                            integer_cols=['CPA変化'],
                            decimal_cols=['CV変化']
                        )
                        
                        st.dataframe(growth_df_formatted)
                    else:
                        st.info("成長重視パターンの媒体はありません")
                else:
                    st.info("分析結果がありません")
            
            with pattern_tabs[3]:
                # 効率重視パターンの媒体を抽出
                if 'analysis_result' in st.session_state:
                    result = st.session_state['analysis_result']
                    patterns = result['media_patterns']['pattern_df']
                    efficiency_patterns = patterns[patterns['pattern'] == 'efficiency']
                    
                    if not efficiency_patterns.empty:
                        efficiency_data = []
                        for _, row in efficiency_patterns.iterrows():
                            efficiency_data.append({
                                '媒体名': row[st.session_state['group_by_cols'][0]],
                                'CV変化': row['cv_change'],
                                'CPA変化': row['cpa_change']
                            })
                        
                        efficiency_df = pd.DataFrame(efficiency_data)
                        
                        # 数値フォーマットの調整
                        efficiency_df_formatted = format_metrics(
                            efficiency_df,
                            integer_cols=['CPA変化'],
                            decimal_cols=['CV変化']
                        )
                        
                        st.dataframe(efficiency_df_formatted)
                    else:
                        st.info("効率重視パターンの媒体はありません")
                else:
                    st.info("分析結果がありません")
            
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
                    
                    # 数値フォーマットの調整
                    campaign_cv_formatted = format_metrics(
                        campaign_cv[['CampaignName', 'previous_cv', 'current_cv', 'cv_change', 'contribution_rate']],
                        integer_cols=[],
                        decimal_cols=['previous_cv', 'current_cv', 'cv_change', 'contribution_rate']
                    )
                    
                    st.dataframe(campaign_cv_formatted)
                    
                    # キャンペーンレベルのCPA変化要因
                    st.write("**キャンペーンレベルのCPA変化要因**")
                    campaign_cpa = campaign_result['cpa_change_factors'].head(5)
                    
                    # CPA変化要因の詳細情報と数値フォーマットの調整
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
    

# 2. CPA変化要因分析
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

# 3. 媒体グループ・パターン分析
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

# 新規: CPA変化の寄与度分解関数
def decompose_cpa_change(current_agg, previous_agg, group_by_cols=['ServiceNameJA']):
    """
    CPA変化を構成要素（CVR変化とCPC変化）に分解し、寄与度を計算する
    
    Parameters:
    current_agg (DataFrame): 当期の集計データ
    previous_agg (DataFrame): 前期の集計データ
    group_by_cols (list): グループ化したカラム
    
    Returns:
    DataFrame: CPA変化の寄与度分解結果
    """
    # 媒体名などのマッチング用にインデックスを設定
    current_df = current_agg.set_index(group_by_cols)
    previous_df = previous_agg.set_index(group_by_cols)
    
    # 共通のインデックスを取得
    common_indices = set(current_df.index) & set(previous_df.index)
    
    # データフレームの準備
    decomp_data = []
    
    # 各媒体のCPA変化の寄与度を分析
    for idx in common_indices:
        try:
            # 前期・当期の値を取得
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
            cpa_change_rate = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else 0
            cvr_change_rate = ((current_cvr - previous_cvr) / previous_cvr) * 100 if previous_cvr != 0 else 0
            cpc_change_rate = ((current_cpc - previous_cpc) / previous_cpc) * 100 if previous_cpc != 0 else 0
            cpm_change_rate = ((current_cpm - previous_cpm) / previous_cpm) * 100 if previous_cpm != 0 else 0
            ctr_change_rate = ((current_ctr - previous_ctr) / previous_ctr) * 100 if previous_ctr != 0 else 0
            
            # 1/CVR と CPC の影響を分離
            # CPA = CPC / CVR なので、CPA変化率 ≈ CPC変化率 - CVR変化率 という近似が成り立つ
            # (厳密には掛け算なので加法的に分解できないが、小さな変化であれば近似として有効)
            
            # CVR変化の寄与度 (減少=CPA増加要因、増加=CPA減少要因)
            cvr_contribution = -cvr_change_rate  # 符号を反転
            
            # CPC変化の寄与度
            cpc_contribution = cpc_change_rate
            
            # CPC変化をCPMとCTRに分解（同様に近似）
            # CPC = CPM / CTR なので、CPC変化率 ≈ CPM変化率 - CTR変化率
            cpm_contribution = cpm_change_rate
            ctr_contribution = -ctr_change_rate  # 符号を反転
            
            # 合計への近似度を確認（理論上は100%に近いはず）
            # 残差はモデルの近似誤差
            total_contribution = cvr_contribution + cpc_contribution
            contribution_accuracy = (total_contribution / cpa_change_rate) * 100 if cpa_change_rate != 0 else 100
            
            total_cpc_contribution = cpm_contribution + ctr_contribution
            cpc_contribution_accuracy = (total_cpc_contribution / cpc_change_rate) * 100 if cpc_change_rate != 0 else 100
            
            # インプレッション数、クリック数、CV数などのボリューム情報も追加
            decomp_data.append({
                'index_value': idx,
                'previous_impressions': previous_df.loc[idx, 'Impressions'],
                'current_impressions': current_df.loc[idx, 'Impressions'],
                'previous_clicks': previous_df.loc[idx, 'Clicks'],
                'current_clicks': current_df.loc[idx, 'Clicks'],
                'previous_cv': previous_df.loc[idx, 'Conversions'],
                'current_cv': current_df.loc[idx, 'Conversions'],
                'previous_cost': previous_df.loc[idx, 'Cost'],
                'current_cost': current_df.loc[idx, 'Cost'],
                'previous_cpa': previous_cpa,
                'current_cpa': current_cpa,
                'previous_cpc': previous_cpc,
                'current_cpc': current_cpc,
                'previous_cvr': previous_cvr,
                'current_cvr': current_cvr,
                'previous_cpm': previous_cpm,
                'current_cpm': current_cpm,
                'previous_ctr': previous_ctr,
                'current_ctr': current_ctr,
                'cpa_change_rate': cpa_change_rate,
                'cvr_change_rate': cvr_change_rate,
                'cpc_change_rate': cpc_change_rate,
                'cpm_change_rate': cpm_change_rate,
                'ctr_change_rate': ctr_change_rate,
                'cvr_contribution': cvr_contribution,
                'cpc_contribution': cpc_contribution,
                'cpm_contribution': cpm_contribution,
                'ctr_contribution': ctr_contribution,
                'total_contribution': total_contribution,
                'contribution_accuracy': contribution_accuracy,
                'cpc_contribution_accuracy': cpc_contribution_accuracy
            })
        
        except Exception as e:
            st.warning(f"CPA変化寄与度分解でエラーが発生しました（{idx}）: {str(e)}")
            continue
    
    # データが空の場合、空のDataFrameを返す
    if not decomp_data:
        # 空のデータフレームを作成（必要な列を持つ）
        empty_df = pd.DataFrame(columns=group_by_cols + [
            'previous_cpa', 'current_cpa', 'cpa_change_rate', 
            'cvr_contribution', 'cpc_contribution'])
        return empty_df
    
    # DataFrameに変換
    decomp_df = pd.DataFrame(decomp_data)
    
    # マルチインデックスの場合の処理
    if len(group_by_cols) > 1:
        # インデックスの分解
        for i, col in enumerate(group_by_cols):
            decomp_df[col] = decomp_df['index_value'].apply(lambda x: x[i])
    else:
        # 単一インデックスの場合
        decomp_df[group_by_cols[0]] = decomp_df['index_value']
    
    # インデックス列を削除
    if 'index_value' in decomp_df.columns:
        decomp_df = decomp_df.drop(columns=['index_value'])
    
    # CPA変化率の絶対値で降順ソート
    if 'cpa_change_rate' in decomp_df.columns:
        decomp_df['abs_cpa_change'] = decomp_df['cpa_change_rate'].abs()
        decomp_df = decomp_df.sort_values('abs_cpa_change', ascending=False)
        decomp_df = decomp_df.drop(columns=['abs_cpa_change'])
    
    return decomp_df

# 新規: CV変化の寄与度分解関数
def decompose_cv_change(current_agg, previous_agg, group_by_cols=['ServiceNameJA']):
    """
    CV変化を構成要素（インプレッション変化、CTR変化、CVR変化）に分解する
    
    Parameters:
    current_agg (DataFrame): 当期の集計データ
    previous_agg (DataFrame): 前期の集計データ
    group_by_cols (list): グループ化したカラム
    
    Returns:
    DataFrame: CV変化の寄与度分解結果
    """
    # 媒体名などのマッチング用にインデックスを設定
    current_df = current_agg.set_index(group_by_cols)
    previous_df = previous_agg.set_index(group_by_cols)
    
    # 共通のインデックスを取得
    common_indices = set(current_df.index) & set(previous_df.index)
    
    # データフレームの準備
    decomp_data = []
    
    # 各媒体のCV変化の寄与度を分析
    for idx in common_indices:
        try:
            # 前期・当期の値を取得
            current_cv = current_df.loc[idx, 'Conversions']
            previous_cv = previous_df.loc[idx, 'Conversions']
            
            current_imp = current_df.loc[idx, 'Impressions']
            previous_imp = previous_df.loc[idx, 'Impressions']
            
            current_ctr = current_df.loc[idx, 'CTR']
            previous_ctr = previous_df.loc[idx, 'CTR']
            
            current_cvr = current_df.loc[idx, 'CVR']
            previous_cvr = previous_df.loc[idx, 'CVR']
            
            # 変化率の計算
            cv_change_rate = ((current_cv - previous_cv) / previous_cv) * 100 if previous_cv != 0 else 0
            imp_change_rate = ((current_imp - previous_imp) / previous_imp) * 100 if previous_imp != 0 else 0
            ctr_change_rate = ((current_ctr - previous_ctr) / previous_ctr) * 100 if previous_ctr != 0 else 0
            cvr_change_rate = ((current_cvr - previous_cvr) / previous_cvr) * 100 if previous_cvr != 0 else 0
            
            # CV = Impressions * (CTR/100) * (CVR/100) なので、
            # CV変化率 ≈ インプレッション変化率 + CTR変化率 + CVR変化率 という近似が成り立つ
            
            # 各要素の寄与度
            imp_contribution = imp_change_rate
            ctr_contribution = ctr_change_rate
            cvr_contribution = cvr_change_rate
            
            # 合計への近似度を確認
            total_contribution = imp_contribution + ctr_contribution + cvr_contribution
            contribution_accuracy = (total_contribution / cv_change_rate) * 100 if cv_change_rate != 0 else 100
            
            decomp_data.append({
                'index_value': idx,
                'previous_impressions': previous_imp,
                'current_impressions': current_imp,
                'previous_cv': previous_cv,
                'current_cv': current_cv,
                'previous_ctr': previous_ctr,
                'current_ctr': current_ctr,
                'previous_cvr': previous_cvr,
                'current_cvr': current_cvr,
                'cv_change_rate': cv_change_rate,
                'imp_change_rate': imp_change_rate,
                'ctr_change_rate': ctr_change_rate,
                'cvr_change_rate': cvr_change_rate,
                'imp_contribution': imp_contribution,
                'ctr_contribution': ctr_contribution,
                'cvr_contribution': cvr_contribution,
                'total_contribution': total_contribution,
                'contribution_accuracy': contribution_accuracy
            })
        
        except Exception as e:
            st.warning(f"CV変化寄与度分解でエラーが発生しました（{idx}）: {str(e)}")
            continue
    
    # データが空の場合、空のDataFrameを返す
    if not decomp_data:
        empty_df = pd.DataFrame(columns=group_by_cols + [
            'previous_cv', 'current_cv', 'cv_change_rate', 
            'imp_contribution', 'ctr_contribution', 'cvr_contribution'])
        return empty_df
    
    # DataFrameに変換
    decomp_df = pd.DataFrame(decomp_data)
    
    # マルチインデックスの場合の処理
    if len(group_by_cols) > 1:
        for i, col in enumerate(group_by_cols):
            decomp_df[col] = decomp_df['index_value'].apply(lambda x: x[i])
    else:
        decomp_df[group_by_cols[0]] = decomp_df['index_value']
    
    # インデックス列を削除
    if 'index_value' in decomp_df.columns:
        decomp_df = decomp_df.drop(columns=['index_value'])
    
    # CV変化率の絶対値で降順ソート
    if 'cv_change_rate' in decomp_df.columns:
        decomp_df['abs_cv_change'] = decomp_df['cv_change_rate'].abs()
        decomp_df = decomp_df.sort_values('abs_cv_change', ascending=False)
        decomp_df = decomp_df.drop(columns=['abs_cv_change'])
    
    return decomp_df

# 新規: 構造変化分析関数
def analyze_structure_change(current_agg, previous_agg, group_by_cols=['ServiceNameJA']):
    """
    媒体間のコスト配分変化とパフォーマンス変化の影響を分離する
    
    Parameters:
    current_agg (DataFrame): 当期の集計データ
    previous_agg (DataFrame): 前期の集計データ
    group_by_cols (list): グループ化したカラム
    
    Returns:
    dict: 構造変化分析結果
    """
    # 全体の合計コストとCV
    total_current_cost = current_agg['Cost'].sum()
    total_previous_cost = previous_agg['Cost'].sum()
    
    total_current_cv = current_agg['Conversions'].sum()
    total_previous_cv = previous_agg['Conversions'].sum()
    
    # 全体のCPA
    total_current_cpa = total_current_cost / total_current_cv if total_current_cv > 0 else 0
    total_previous_cpa = total_previous_cost / total_previous_cv if total_previous_cv > 0 else 0
    total_cpa_change = ((total_current_cpa - total_previous_cpa) / total_previous_cpa) * 100 if total_previous_cpa > 0 else 0
    
    # 媒体別の分析データ
    structure_data = []
    
    # 媒体名を取得（グループ化カラムが複数の場合は最初のカラムを使用）
    primary_key = group_by_cols[0]
    
    # 各媒体の前期・当期データを取得
    for _, row in current_agg.iterrows():
        media_name = row[primary_key]
        
        # 前期データを探す
        previous_row = previous_agg[previous_agg[primary_key] == media_name]
        
        if not previous_row.empty:
            # 前期データが存在する場合
            prev_row = previous_row.iloc[0]
            
            # コスト配分比率
            current_cost_ratio = row['Cost'] / total_current_cost if total_current_cost > 0 else 0
            previous_cost_ratio = prev_row['Cost'] / total_previous_cost if total_previous_cost > 0 else 0
            cost_ratio_change = current_cost_ratio - previous_cost_ratio
            
            # パフォーマンス（CPA）
            current_cpa = row['CPA']
            previous_cpa = prev_row['CPA']
            cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa > 0 else 0
            
            # シミュレーション1: 配分比率だけが変わった場合のCPA
            # 前期のCPAを維持し、当期の配分比率を適用
            if total_current_cost > 0:
                allocation_effect_cv = (row['Cost'] / previous_cpa) if previous_cpa > 0 else 0
            else:
                allocation_effect_cv = 0
            
            # シミュレーション2: パフォーマンスだけが変わった場合のCV
            # 前期の配分比率を維持し、当期のCPAを適用
            if current_cpa > 0:
                performance_effect_cv = (prev_row['Cost'] / current_cpa)
            else:
                performance_effect_cv = 0
            
            structure_data.append({
                primary_key: media_name,
                'previous_cost': prev_row['Cost'],
                'current_cost': row['Cost'],
                'previous_cost_ratio': previous_cost_ratio * 100,  # パーセント表示
                'current_cost_ratio': current_cost_ratio * 100,    # パーセント表示
                'cost_ratio_change': cost_ratio_change * 100,      # パーセントポイント
                'previous_cpa': previous_cpa,
                'current_cpa': current_cpa,
                'cpa_change': cpa_change,
                'previous_cv': prev_row['Conversions'],
                'current_cv': row['Conversions'],
                'allocation_effect_cv': allocation_effect_cv,
                'performance_effect_cv': performance_effect_cv
            })
        else:
            # 前期データが存在しない場合（新規媒体）
            current_cost_ratio = row['Cost'] / total_current_cost if total_current_cost > 0 else 0
            
            structure_data.append({
                primary_key: media_name,
                'previous_cost': 0,
                'current_cost': row['Cost'],
                'previous_cost_ratio': 0,
                'current_cost_ratio': current_cost_ratio * 100,
                'cost_ratio_change': current_cost_ratio * 100,
                'previous_cpa': 0,
                'current_cpa': row['CPA'],
                'cpa_change': float('inf'),
                'previous_cv': 0,
                'current_cv': row['Conversions'],
                'allocation_effect_cv': row['Conversions'],
                'performance_effect_cv': 0
            })
    
    # 前期のみに存在した媒体（削除された媒体）
    for _, row in previous_agg.iterrows():
        media_name = row[primary_key]
        
        # 当期データを探す
        current_row = current_agg[current_agg[primary_key] == media_name]
        
        if current_row.empty:
            # 当期データが存在しない場合（削除された媒体）
            previous_cost_ratio = row['Cost'] / total_previous_cost if total_previous_cost > 0 else 0
            
            structure_data.append({
                primary_key: media_name,
                'previous_cost': row['Cost'],
                'current_cost': 0,
                'previous_cost_ratio': previous_cost_ratio * 100,
                'current_cost_ratio': 0,
                'cost_ratio_change': -previous_cost_ratio * 100,
                'previous_cpa': row['CPA'],
                'current_cpa': 0,
                'cpa_change': -100,
                'previous_cv': row['Conversions'],
                'current_cv': 0,
                'allocation_effect_cv': -row['Conversions'],
                'performance_effect_cv': 0
            })
    
    # DataFrameに変換
    structure_df = pd.DataFrame(structure_data)
    
    # コスト配分変化の絶対値で降順ソート
    if 'cost_ratio_change' in structure_df.columns:
        structure_df['abs_cost_ratio_change'] = structure_df['cost_ratio_change'].abs()
        structure_df = structure_df.sort_values('abs_cost_ratio_change', ascending=False)
        structure_df = structure_df.drop(columns=['abs_cost_ratio_change'])
    
    # 全体への影響の計算
    # 配分効果によるCV変化の合計
    allocation_effect_total_cv = structure_df['allocation_effect_cv'].sum() - total_previous_cv
    
    # パフォーマンス効果によるCV変化の合計
    performance_effect_total_cv = structure_df['performance_effect_cv'].sum() - total_previous_cv
    
    # 実際のCV変化
    actual_cv_change = total_current_cv - total_previous_cv
    
    # 配分効果とパフォーマンス効果の寄与度
    if actual_cv_change != 0:
        allocation_contribution = (allocation_effect_total_cv / actual_cv_change) * 100
        performance_contribution = (performance_effect_total_cv / actual_cv_change) * 100
    else:
        allocation_contribution = 0
        performance_contribution = 0
    
    # 結果をまとめて返す
    result = {
        'structure_df': structure_df,
        'total_current_cost': total_current_cost,
        'total_previous_cost': total_previous_cost,
        'total_current_cv': total_current_cv,
        'total_previous_cv': total_previous_cv,
        'total_current_cpa': total_current_cpa,
        'total_previous_cpa': total_previous_cpa,
        'total_cpa_change': total_cpa_change,
        'allocation_effect_total_cv': allocation_effect_total_cv,
        'performance_effect_total_cv': performance_effect_total_cv,
        'actual_cv_change': actual_cv_change,
        'allocation_contribution': allocation_contribution,
        'performance_contribution': performance_contribution
    }
    
    return result

# 新規: 階層的変化点検出関数
def detect_change_points(current_df, previous_df, hierarchy_cols=['ServiceNameJA', 'CampaignName', 'AdgroupName'], 
                         metric_thresholds={'CPA': 15, 'CVR': 15, 'CTR': 15, 'CPM': 15}):
    """
    階層的に重要な変化点を検出する関数
    
    Parameters:
    current_df (DataFrame): 当期のデータ
    previous_df (DataFrame): 前期のデータ
    hierarchy_cols (list): 階層を示すカラム（順序指定）
    metric_thresholds (dict): 各指標の変化率閾値（%）
    
    Returns:
    dict: 階層別の変化点データ
    """
    hierarchy_results = {}
    remaining_df_current = current_df.copy()
    remaining_df_previous = previous_df.copy()
    
    # 各階層レベルで分析を実施
    for level, col in enumerate(hierarchy_cols):
        # 現在の階層までのカラムを使用
        current_hierarchy = hierarchy_cols[:level+1]
        
        # 指定されたカラムが存在するか確認
        if not all(c in remaining_df_current.columns for c in current_hierarchy) or \
           not all(c in remaining_df_previous.columns for c in current_hierarchy):
            # この階層のカラムが存在しない場合、ループを終了
            break
        
        # データを集計
        current_agg = aggregate_data_by_period(remaining_df_current, current_hierarchy)
        previous_agg = aggregate_data_by_period(remaining_df_previous, current_hierarchy)
        
        if current_agg is None or previous_agg is None:
            continue
        
        # 主要指標の変化点を検出
        change_points = []
        
        # インデックス用にグループ化カラムを設定
        current_df_indexed = current_agg.set_index(current_hierarchy)
        previous_df_indexed = previous_agg.set_index(current_hierarchy)
        
        # 共通のインデックスを取得
        common_indices = set(current_df_indexed.index) & set(previous_df_indexed.index)
        
        for idx in common_indices:
            try:
                # 前期・当期の値を取得
                metrics_changes = {}
                significant_changes = False
                
                # 各指標の変化率を計算
                for metric, threshold in metric_thresholds.items():
                    if metric in current_df_indexed.columns and metric in previous_df_indexed.columns:
                        current_value = current_df_indexed.loc[idx, metric]
                        previous_value = previous_df_indexed.loc[idx, metric]
                        
                        # 変化率の計算
                        if previous_value != 0:
                            change_rate = ((current_value - previous_value) / previous_value) * 100
                        else:
                            change_rate = float('inf') if current_value > 0 else 0
                        
                        metrics_changes[metric] = {
                            'previous': previous_value,
                            'current': current_value,
                            'change_rate': change_rate
                        }
                        
                        # 閾値を超える変化があるか確認
                        if abs(change_rate) >= threshold:
                            significant_changes = True
                
                # インプレッション、クリック、コスト、CVの変化も追加
                volume_metrics = ['Impressions', 'Clicks', 'Cost', 'Conversions']
                for metric in volume_metrics:
                    if metric in current_df_indexed.columns and metric in previous_df_indexed.columns:
                        current_value = current_df_indexed.loc[idx, metric]
                        previous_value = previous_df_indexed.loc[idx, metric]
                        
                        # 変化率の計算
                        if previous_value != 0:
                            change_rate = ((current_value - previous_value) / previous_value) * 100
                        else:
                            change_rate = float('inf') if current_value > 0 else 0
                        
                        metrics_changes[metric] = {
                            'previous': previous_value,
                            'current': current_value,
                            'change_rate': change_rate
                        }
                
                # 重要な変化が見つかった場合のみ追加
                if significant_changes:
                    # インデックスが単一の場合とタプルの場合の処理
                    if isinstance(idx, tuple):
                        node_info = dict(zip(current_hierarchy, idx))
                    else:
                        node_info = {current_hierarchy[0]: idx}
                    
                    change_points.append({
                        'node': node_info,
                        'metrics': metrics_changes,
                        'level': level
                    })
            
            except Exception as e:
                st.warning(f"変化点検出でエラーが発生しました（{idx}）: {str(e)}")
                continue
        
        # 検出した変化点を重要度でソート（CPA変化の絶対値を基準に）
        if change_points:
            sorted_points = sorted(
                change_points, 
                key=lambda x: abs(x['metrics'].get('CPA', {}).get('change_rate', 0)), 
                reverse=True
            )
            
            hierarchy_results[level] = {
                'column': col,
                'change_points': sorted_points
            }
            
            # 次のレベルの分析用に、重要な変化点のみにフィルタリング
            if level < len(hierarchy_cols) - 1 and sorted_points:
                # 上位5つの変化点に関連するデータのみを抽出
                top_nodes = sorted_points[:5]
                
                # フィルタ条件の構築
                filters_current = []
                filters_previous = []
                
                for node in top_nodes:
                    node_filter_current = True
                    node_filter_previous = True
                    
                    for col_name, col_value in node['node'].items():
                        node_filter_current = node_filter_current & (remaining_df_current[col_name] == col_value)
                        node_filter_previous = node_filter_previous & (remaining_df_previous[col_name] == col_value)
                    
                    filters_current.append(node_filter_current)
                    filters_previous.append(node_filter_previous)
                
                # 複数のフィルタを「OR」条件で結合
                final_filter_current = np.zeros(len(remaining_df_current), dtype=bool)
                final_filter_previous = np.zeros(len(remaining_df_previous), dtype=bool)
                
                for f in filters_current:
                    final_filter_current = final_filter_current | f
                
                for f in filters_previous:
                    final_filter_previous = final_filter_previous | f
                
                # フィルタを適用
                remaining_df_current = remaining_df_current[final_filter_current]
                remaining_df_previous = remaining_df_previous[final_filter_previous]
    
    return hierarchy_results

# 新規: 変化のタイミング分析関数
def analyze_change_timing(df, node_info, start_date, end_date, metric='Conversions'):
    """
    特定のノード（媒体・キャンペーンなど）の変化タイミングを分析する
    
    Parameters:
    df (DataFrame): 全期間のデータ
    node_info (dict): 分析対象ノードの情報（カラム名: 値）
    start_date (datetime): 分析開始日
    end_date (datetime): 分析終了日
    metric (str): 分析対象の指標
    
    Returns:
    dict: 変化タイミング分析結果
    """
    # 指定されたノードに関するデータだけをフィルタリング
    node_filter = pd.Series(True, index=df.index)
    for col, value in node_info.items():
        if col in df.columns:
            node_filter = node_filter & (df[col] == value)
    
    filtered_df = df[node_filter]
    
    # 日付範囲内のデータだけを抽出
    date_filter = (filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)
    date_filtered_df = filtered_df[date_filter]
    
    # 日別データを集計
    daily_data = date_filtered_df.groupby('Date')[metric].sum().reset_index()
    
    # データが少なすぎる場合は分析不可
    if len(daily_data) < 7:
        return {
            'status': 'insufficient_data',
            'message': '日次データが不足しているため、変化タイミングの分析ができません',
            'daily_data': daily_data
        }
    
    # 7日間の移動平均を計算
    daily_data['moving_avg'] = daily_data[metric].rolling(window=7, min_periods=1).mean()
    
    # 変化率の計算（前日比）
    daily_data['change_rate'] = daily_data[metric].pct_change() * 100
    
    # 変化率の移動平均
    daily_data['change_rate_ma'] = daily_data['change_rate'].rolling(window=3, min_periods=1).mean()
    
    # 急激な変化があった日を特定（標準偏差の2倍以上の変化）
    std_dev = daily_data['change_rate'].std()
    threshold = 2 * std_dev
    
    significant_changes = daily_data[abs(daily_data['change_rate']) > threshold].copy()
    
    # 結果をまとめる
    result = {
        'status': 'success',
        'daily_data': daily_data,
        'significant_changes': significant_changes,
        'threshold': threshold,
        'mean_value': daily_data[metric].mean(),
        'std_dev': std_dev
    }
    
    # 変化パターンの分析（段階的 vs 突発的）
    if len(significant_changes) > 0:
        # 変化の連続性をチェック
        consecutive_days = 0
        for i in range(1, len(daily_data)):
            if abs(daily_data['change_rate'].iloc[i]) > threshold/2:  # 閾値の半分以上の変化
                consecutive_days += 1
            else:
                consecutive_days = 0
            
            if consecutive_days >= 3:  # 3日連続で変化がある場合は段階的変化と判断
                result['change_pattern'] = 'gradual'
                break
        else:
            # 連続した変化がない場合は突発的変化と判断
            result['change_pattern'] = 'sudden'
    else:
        result['change_pattern'] = 'stable'  # 有意な変化なし
    
    return result

# 新規: ヒートマップを作成する関数
def create_metric_heatmap(cpa_decomp_df, cv_decomp_df, group_by_col='ServiceNameJA'):
    """
    指標変化のヒートマップを作成する
    
    Parameters:
    cpa_decomp_df (DataFrame): CPA変化の寄与度分解結果
    cv_decomp_df (DataFrame): CV変化の寄与度分解結果
    group_by_col (str): グループ化カラム名
    
    Returns:
    plotly.graph_objects.Figure: ヒートマップ図
    """
    # 指標リスト
    metrics = ['CPA', 'CVR', 'CPC', 'CPM', 'CTR', 'CV']
    
    # データの準備
    heatmap_data = []
    
    # 共通の媒体を取得
    media_list = list(set(cpa_decomp_df[group_by_col]) & set(cv_decomp_df[group_by_col]))
    
    for media in media_list:
        row_data = {group_by_col: media}
        
        # CPA関連指標
        cpa_row = cpa_decomp_df[cpa_decomp_df[group_by_col] == media].iloc[0] if not cpa_decomp_df[cpa_decomp_df[group_by_col] == media].empty else None
        if cpa_row is not None:
            row_data['CPA'] = cpa_row['cpa_change_rate']
            row_data['CVR'] = cpa_row['cvr_change_rate']
            row_data['CPC'] = cpa_row['cpc_change_rate']
            row_data['CPM'] = cpa_row['cpm_change_rate']
            row_data['CTR'] = cpa_row['ctr_change_rate']
        
        # CV関連指標
        cv_row = cv_decomp_df[cv_decomp_df[group_by_col] == media].iloc[0] if not cv_decomp_df[cv_decomp_df[group_by_col] == media].empty else None
        if cv_row is not None:
            row_data['CV'] = cv_row['cv_change_rate']
        
        heatmap_data.append(row_data)
    
    # DataFrameに変換
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # 指標データのみを抽出
    z_data = heatmap_df[metrics].values.tolist()
    
    # メディア名のリスト
    media_names = heatmap_df[group_by_col].tolist()
    
    # カラーマップの設定（赤: 悪化、緑: 改善）
    # CPAとCPC、CPMは負の値が改善、CVRとCTRとCVは正の値が改善
    custom_color_scales = {
        'CPA': [[0, 'green'], [0.5, 'white'], [1, 'red']],     # 負の値（減少）が緑
        'CPC': [[0, 'green'], [0.5, 'white'], [1, 'red']],     # 負の値（減少）が緑
        'CPM': [[0, 'green'], [0.5, 'white'], [1, 'red']],     # 負の値（減少）が緑
        'CVR': [[0, 'red'], [0.5, 'white'], [1, 'green']],     # 正の値（増加）が緑
        'CTR': [[0, 'red'], [0.5, 'white'], [1, 'green']],     # 正の値（増加）が緑
        'CV': [[0, 'red'], [0.5, 'white'], [1, 'green']],      # 正の値（増加）が緑
    }
    
    # 各指標の値を正規化（-100〜100%の範囲に収める）
    normalized_z = np.zeros_like(z_data, dtype=float)
    for i, metric in enumerate(metrics):
        # 値を取得
        metric_values = [row[i] for row in z_data]
        
        # 無限大やNaNを除外
        valid_values = [v for v in metric_values if not np.isinf(v) and not np.isnan(v)]
        
        # 有効な値がない場合はスキップ
        if not valid_values:
            continue
        
        # 最大絶対値を取得（最小-100%、最大100%に制限）
        max_abs_value = min(max(abs(min(valid_values)), abs(max(valid_values)), 100), 100)
        
        # 値を-1〜1の範囲に正規化
        for j, value in enumerate(metric_values):
            if np.isinf(value) or np.isnan(value):
                normalized_z[j][i] = 0  # 無効な値は0に設定
            else:
                normalized_z[j][i] = value / max_abs_value
    
    # ヒートマップの作成
    fig = make_subplots(rows=1, cols=1)
    
    # 各指標ごとにヒートマップを作成し、透明度で表示
    for i, metric in enumerate(metrics):
        # 指標の値を抽出
        z_values = [[normalized_z[j][i]] for j in range(len(media_names))]
        
        # カラースケールの調整（値に応じて色を変える）
        colorscale = custom_color_scales[metric]
        
        # ヒートマップの追加
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=[metric],
                y=media_names,
                colorscale=colorscale,
                showscale=False,
                text=[[f"{z_data[j][i]:.1f}%" if not np.isnan(z_data[j][i]) and not np.isinf(z_data[j][i]) else "N/A" for j in range(len(media_names))]],
                hovertemplate="%{y}の%{x}変化率: %{text}<extra></extra>",
                texttemplate="%{text}",
                textfont={"size": 10}
            )
        )
    
    # レイアウトの設定
    fig.update_layout(
        title="指標変化率ヒートマップ（赤: 悪化、緑: 改善）",
        height=max(400, len(media_names) * 30),  # 媒体数に応じた高さ
        width=800,
        xaxis=dict(title="指標"),
        yaxis=dict(title=group_by_col),
        margin=dict(l=100, r=20, t=70, b=50)
    )
    
    return fig

# 新規: ウォーターフォールチャートを作成する関数
def create_waterfall_chart(decomp_data, title, baseline_value, contribution_fields, colors):
    """
    寄与度をウォーターフォールチャートで表示する
    
    Parameters:
    decomp_data (dict): 分解データ
    title (str): チャートタイトル
    baseline_value (float): 基準値
    contribution_fields (list): 寄与度フィールド名のリスト
    colors (list): 使用する色のリスト
    
    Returns:
    plotly.graph_objects.Figure: ウォーターフォールチャート
    """
    # データの準備
    measure = ['absolute']  # 1番目は基準値（絶対値）
    x = ['前期']  # 1番目のラベル
    
    # 基準値から開始
    y = [baseline_value]
    
    # 各寄与度を追加
    for i, field in enumerate(contribution_fields):
        if field in decomp_data:
            contribution = decomp_data[field]
            x.append(field.replace('_contribution', ''))
            y.append(contribution)
            
            # プラスかマイナスかで表示を変える
            measure.append('relative')
    
    # 最終値を追加
    final_value = baseline_value
    for i in range(1, len(y)):
        final_value += y[i]
    
    x.append('当期')
    y.append(0)  # 調整値（最終合計が正しくなるように）
    measure.append('total')
    
    # チャートの作成
    fig = go.Figure(go.Waterfall(
        name="寄与度",
        orientation="v",
        measure=measure,
        x=x,
        y=y,
        textposition="outside",
        text=[f"{val:.1f}" for val in y],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": colors[0]}},
        increasing={"marker": {"color": colors[1]}},
        totals={"marker": {"color": colors[2]}}
    ))
    
    # レイアウトの設定
    fig.update_layout(
        title=title,
        showlegend=False,
        height=400,
        width=700,
        margin=dict(l=50, r=50, t=70, b=50)
    )
    
    return fig

# 新規: サンキーダイアグラム作成関数
def create_sankey_diagram(structure_data, group_by_col='ServiceNameJA'):
    """
    媒体間のコスト配分変化をサンキーダイアグラムで表示
    
    Parameters:
    structure_data (dict): 構造変化分析結果
    group_by_col (str): グループ化カラム名
    
    Returns:
    plotly.graph_objects.Figure: サンキーダイアグラム
    """
    # 構造データのDataFrameを取得
    df = structure_data['structure_df']
    
    # 媒体リストを取得
    media_list = df[group_by_col].tolist()
    
    # ノードの作成（前期の媒体、当期の媒体、合計ノードを含む）
    nodes = (
        ["前期 Total"] +  # 0: 前期合計
        [f"前期 {media}" for media in media_list] +  # 1〜n: 前期の各媒体
        ["当期 Total"] +  # n+1: 当期合計
        [f"当期 {media}" for media in media_list]  # n+2〜2n+1: 当期の各媒体
    )
    
    # リンクの作成
    links_source = []
    links_target = []
    links_value = []
    links_color = []
    
    # 全体のコスト変化率を計算
    total_cost_change = ((structure_data['total_current_cost'] - structure_data['total_previous_cost']) / 
                         structure_data['total_previous_cost']) * 100 if structure_data['total_previous_cost'] > 0 else 0
    
    # 前期合計から各媒体へのリンク
    for i, media in enumerate(media_list):
        links_source.append(0)  # 前期合計
        links_target.append(i + 1)  # 前期の各媒体
        
        # 前期のコスト
        previous_cost = df[df[group_by_col] == media]['previous_cost'].values[0]
        links_value.append(previous_cost)
        links_color.append("rgba(180, 180, 180, 0.3)")  # 薄いグレー
    
    # 各媒体から当期の媒体へのリンク
    n = len(media_list)
    for i, media in enumerate(media_list):
        media_data = df[df[group_by_col] == media]
        
        # コストと変化率
        previous_cost = media_data['previous_cost'].values[0]
        current_cost = media_data['current_cost'].values[0]
        
        # 前期の媒体から当期の媒体へのリンク
        if previous_cost > 0 and current_cost > 0:
            links_source.append(i + 1)  # 前期の媒体
            links_target.append(n + 2 + i)  # 当期の媒体
            links_value.append(min(previous_cost, current_cost))  # 小さい方の値
            
            # コスト変化率に基づいて色を設定
            cost_change = ((current_cost - previous_cost) / previous_cost) * 100
            
            # 全体の変化より良い（コスト減少または増加率が小さい）場合は緑、悪い場合は赤
            if cost_change < total_cost_change:
                links_color.append("rgba(75, 192, 192, 0.7)")  # 緑
            else:
                links_color.append("rgba(255, 99, 132, 0.7)")  # 赤
    
    # 新規追加された媒体（前期コスト=0）
    for i, media in enumerate(media_list):
        media_data = df[df[group_by_col] == media]
        
        previous_cost = media_data['previous_cost'].values[0]
        current_cost = media_data['current_cost'].values[0]
        
        if previous_cost == 0 and current_cost > 0:
            links_source.append(n + 1)  # 当期合計
            links_target.append(n + 2 + i)  # 当期の媒体
            links_value.append(current_cost)
            links_color.append("rgba(54, 162, 235, 0.7)")  # 青（新規）
    
    # 削除された媒体（当期コスト=0）
    for i, media in enumerate(media_list):
        media_data = df[df[group_by_col] == media]
        
        previous_cost = media_data['previous_cost'].values[0]
        current_cost = media_data['current_cost'].values[0]
        
        if previous_cost > 0 and current_cost == 0:
            links_source.append(i + 1)  # 前期の媒体
            links_target.append(n + 1)  # 当期合計
            links_value.append(previous_cost)
            links_color.append("rgba(255, 159, 64, 0.7)")  # オレンジ（削除）
    
    # 各媒体から当期合計へのリンク
    for i, media in enumerate(media_list):
        media_data = df[df[group_by_col] == media]
        current_cost = media_data['current_cost'].values[0]
        
        if current_cost > 0:
            links_source.append(n + 2 + i)  # 当期の媒体
            links_target.append(n + 1)  # 当期合計
            links_value.append(current_cost)
            links_color.append("rgba(180, 180, 180, 0.3)")  # 薄いグレー
    
    # ノードの色設定
    node_colors = (
        ["rgba(150, 150, 150, 0.8)"] +  # 前期合計
        ["rgba(200, 200, 200, 0.5)"] * n +  # 前期の各媒体
        ["rgba(150, 150, 150, 0.8)"] +  # 当期合計
        ["rgba(200, 200, 200, 0.5)"] * n  # 当期の各媒体
    )
    
    # サンキーダイアグラムの作成
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value,
            color=links_color
        )
    )])
    
    # レイアウトの設定
    fig.update_layout(
        title_text="媒体間のコスト配分変化",
        font_size=10,
        height=600,
        width=800,
        margin=dict(l=50, r=50, t=70, b=50)
    )
    
    return fig

# 重要な媒体を自動的に特定する関数
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

# 自動分析プロセスを実行する関数
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

    
    # タブ6: レポート出力（既存タブの強化）
    with tab6:
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
            
            # 追加のグラフを表示するオプション
            with st.expander("分析グラフの表示"):
                # 分析結果を使用
                result = st.session_state['analysis_result']
                
                # CPA変化の要因分解グラフ
                if 'cpa_decomposition' in result:
                    st.write("##### CPA変化の要因分解")
                    
                    # CPA変化の寄与度データを準備
                    cpa_decomp = result['cpa_decomposition'].head(10)  # 上位10媒体
                    
                    # バーチャートの作成
                    fig = go.Figure()
                    
                    # CVR寄与度
                    fig.add_trace(go.Bar(
                        y=cpa_decomp[st.session_state['group_by_cols'][0]],
                        x=cpa_decomp['cvr_contribution'],
                        name='CVR寄与度',
                        orientation='h',
                        marker=dict(color='rgba(55, 126, 184, 0.7)')
                    ))
                    
                    # CPC寄与度
                    fig.add_trace(go.Bar(
                        y=cpa_decomp[st.session_state['group_by_cols'][0]],
                        x=cpa_decomp['cpc_contribution'],
                        name='CPC寄与度',
                        orientation='h',
                        marker=dict(color='rgba(255, 127, 14, 0.7)')
                    ))
                    
                    # レイアウト設定
                    fig.update_layout(
                        title="媒体別CPA変化の要因分解",
                        xaxis_title="寄与度（%）",
                        yaxis_title="媒体名",
                        barmode='relative',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # CV変化の要因分解グラフ
                if 'cv_decomposition' in result:
                    st.write("##### CV変化の要因分解")
                    
                    # CV変化の寄与度データを準備
                    cv_decomp = result['cv_decomposition'].head(10)  # 上位10媒体
                    
                    # バーチャートの作成
                    fig = go.Figure()
                    
                    # インプレッション寄与度
                    fig.add_trace(go.Bar(
                        y=cv_decomp[st.session_state['group_by_cols'][0]],
                        x=cv_decomp['imp_contribution'],
                        name='インプレッション寄与度',
                        orientation='h',
                        marker=dict(color='rgba(44, 160, 44, 0.7)')
                    ))
                    
                    # CTR寄与度
                    fig.add_trace(go.Bar(
                        y=cv_decomp[st.session_state['group_by_cols'][0]],
                        x=cv_decomp['ctr_contribution'],
                        name='CTR寄与度',
                        orientation='h',
                        marker=dict(color='rgba(214, 39, 40, 0.7)')
                    ))
                    
                    # CVR寄与度
                    fig.add_trace(go.Bar(
                        y=cv_decomp[st.session_state['group_by_cols'][0]],
                        x=cv_decomp['cvr_contribution'],
                        name='CVR寄与度',
                        orientation='h',
                        marker=dict(color='rgba(148, 103, 189, 0.7)')
                    ))
                    
                    # レイアウト設定
                    fig.update_layout(
                        title="媒体別CV変化の要因分解",
                        xaxis_title="寄与度（%）",
                        yaxis_title="媒体名",
                        barmode='relative',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 構造変化分析のサンキーダイアグラム
                if 'structure_analysis' in result:
                    st.write("##### コスト配分変化のサンキーダイアグラム")
                    
                    sankey_fig = create_sankey_diagram(
                        result['structure_analysis'],
                        st.session_state['group_by_cols'][0]
                    )
                    
                    st.plotly_chart(sankey_fig, use_container_width=True)
                
                # 指標変化のヒートマップ
                if 'cpa_decomposition' in result and 'cv_decomposition' in result:
                    st.write("##### 指標変化のヒートマップ")
                    
                    heatmap_fig = create_metric_heatmap(
                        result['cpa_decomposition'],
                        result['cv_decomposition'],
                        st.session_state['group_by_cols'][0]
                    )
                    
                    st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # レポートダウンロードボタン
            report_text = interpretation['interpretation']
            
            # 自動生成されたエグゼクティブサマリーがあれば追加
            if 'auto_analysis_result' in st.session_state and 'analysis_result' in st.session_state:
                auto_result = st.session_state['auto_analysis_result']
                result = st.session_state['analysis_result']
                
                # エグゼクティブサマリーの生成
                current_total = result['current_total']
                previous_total = result['previous_total']
                
                cv_change = ((current_total['Conversions'] - previous_total['Conversions']) / previous_total['Conversions']) * 100 if previous_total['Conversions'] != 0 else 0
                
                previous_cpa = previous_total['Cost'] / previous_total['Conversions'] if previous_total['Conversions'] != 0 else 0
                current_cpa = current_total['Cost'] / current_total['Conversions'] if current_total['Conversions'] != 0 else 0
                cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else 0
                
                # CPA変化の主要因
                cpa_direction = "改善" if cpa_change < 0 else "悪化"
                cv_direction = "増加" if cv_change > 0 else "減少"
                
                # 主要な変化要因を特定
                summary_parts = []
                
                # CPA/CV変化の基本情報
                summary_parts.append(f"CPA {abs(cpa_change):.1f}%{cpa_direction}・CV {abs(cv_change):.1f}%{cv_direction}。")
                
                # 主要因の追加
                if 'cpa_decomposition' in result:
                    cpa_decomp = result['cpa_decomposition'].iloc[0] if not result['cpa_decomposition'].empty else None
                    
                    if cpa_decomp is not None:
                        cvr_contribution = cpa_decomp['cvr_contribution']
                        cpc_contribution = cpa_decomp['cpc_contribution']
                        
                        # CPA変化の主要因
                        if abs(cvr_contribution) > abs(cpc_contribution):
                            contribution_str = f"CVR変化({cvr_contribution:.1f}%)"
                        else:
                            contribution_str = f"CPC変化({cpc_contribution:.1f}%)"
                        
                        summary_parts.append(f"CPA変化の主要因は{contribution_str}。")
                
                # 構造変化の寄与度
                if 'structure_analysis' in result:
                    structure = result['structure_analysis']
                    
                    allocation_contribution = structure['allocation_contribution']
                    performance_contribution = structure['performance_contribution']
                    
                    # 構造変化の主要因
                    if abs(allocation_contribution) > abs(performance_contribution):
                        structure_str = f"媒体間コスト配分変化({allocation_contribution:.1f}%)"
                    else:
                        structure_str = f"媒体自体のパフォーマンス変化({performance_contribution:.1f}%)"
                    
                    summary_parts.append(f"CV変化の主要因は{structure_str}。")
                
                # 重要な媒体の変化
                important_media = auto_result['important_media']
                if important_media['high_cv_contribution'] and len(important_media['high_cv_contribution']) > 0:
                    top_media = important_media['high_cv_contribution'][0]
                    media_name = top_media['media_name']
                    contribution = top_media['contribution_rate']
                    
                    summary_parts.append(f"{media_name}のCV寄与率({contribution:.1f}%)が最大。")
                
                # サマリーの結合
                exec_summary = "# エグゼクティブサマリー\n\n" + " ".join(summary_parts) + "\n\n"
                
                # レポートの先頭にエグゼクティブサマリーを追加
                report_text = exec_summary + report_text
            
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
    
    # タブ7: 分析手法の説明（既存タブを強化）
    with tab7:
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
        
        # 2. 全体パフォーマンス変化の構造化分析（新規）
        with st.expander("2. 全体パフォーマンス変化の構造化分析", expanded=False):
            st.markdown("""
            ### CPA変化の寄与度分解
            1. CPA変化率の計算:
            ```
            CPA変化率 = (当期CPA - 前期CPA) / 前期CPA × 100%
            ```
            
            2. CPA変化の要因分解（近似式）:
            ```
            CPA = CPC / CVR
            ```
            から、
            ```
            CPA変化率 ≈ CPC変化率 - CVR変化率
            ```
            
            3. CPC変化の要因分解:
            ```
            CPC = CPM / CTR
            ```
            から、
            ```
            CPC変化率 ≈ CPM変化率 - CTR変化率
            ```
            
            ### CV変化の寄与度分解
            1. CV変化率の計算:
            ```
            CV変化率 = (当期CV - 前期CV) / 前期CV × 100%
            ```
            
            2. CV変化の要因分解（近似式）:
            ```
            CV = インプレッション数 × (CTR/100) × (CVR/100)
            ```
            から、
            ```
            CV変化率 ≈ インプレッション変化率 + CTR変化率 + CVR変化率
            ```
            """)
        
        # 3. 構造変化分析（新規）
        with st.expander("3. 構造変化分析", expanded=False):
            st.markdown("""
            ### 媒体間のコスト配分変化分析
            
            1. コスト配分比率の計算:
            ```
            コスト配分比率 = 媒体コスト ÷ 全体コスト × 100%
            ```
            
            2. 配分変化の計算:
            ```
            配分変化(pp) = 当期配分比率 - 前期配分比率
            ```
            
            3. 配分効果のシミュレーション:
            * 前期のCPA効率を維持し、当期の配分比率を適用した場合のCV
            * 前期の配分比率を維持し、当期のCPA効率を適用した場合のCV
            
            4. 寄与度の分離:
            * 配分変化の寄与度: 配分効果によるCV変化 ÷ 実際のCV変化 × 100%
            * パフォーマンス変化の寄与度: パフォーマンス効果によるCV変化 ÷ 実際のCV変化 × 100%
            """)
        
        # 4. 階層的変化点特定（新規）
        with st.expander("4. 階層的変化点特定", expanded=False):
            st.markdown("""
            ### 階層的変化点検出
            
            1. 各階層レベル（媒体→キャンペーン→広告グループ）で重要な変化を検出:
            * CPA変化率が閾値（例: ±15%）を超える
            * CVR変化率が閾値を超える
            * CTR変化率が閾値を超える
            * CPM変化率が閾値を超える
            
            2. 重要度でランキング（CPA変化の絶対値を基準）
            
            3. 重要な変化点のみに絞って次の階層を分析
            
            ### 変化のタイミング分析
            
            1. 日次データを時系列で分析
            
            2. 7日間の移動平均を計算
            
            3. 急激な変化があった日を特定（標準偏差の2倍以上の変化）
            
            4. 変化パターンの判別:
            * 段階的変化: 複数日にわたる継続的な変化
            * 突発的変化: 特定日に大きな急激な変化
            * 安定: 有意な変化なし
            """)
        
        # 5. ビジュアライゼーション（新規）
        with st.expander("5. データビジュアライゼーション", expanded=False):
            st.markdown("""
            ### 主要なビジュアライゼーション
            
            1. 指標変化のヒートマップ
            * 媒体×指標のマトリクスで変化率を色分け表示
            * 赤: 悪化、緑: 改善
            * 一目で問題領域と成功領域を把握可能
            
            2. 変化要因の寄与度チャート
            * CPAとCV変化への各要因の寄与度をバーチャートで表示
            * プラス寄与とマイナス寄与を視覚的に区別
            
            3. 構造変化のサンキーダイアグラム
            * 媒体間の予算配分変化をサンキーダイアグラムで表示
            * 配分シフトの方向と規模を直感的に把握
            
            4. 変化タイミングの時系列チャート
            * 日次の値と移動平均を表示
            * 重要な変化点をマークで強調
            """)
        
        # 6. レポート生成（新規）
        with st.expander("6. レポート生成", expanded=False):
            st.markdown("""
            ### 階層的レポート構成
            
            1. エグゼクティブサマリー (1-3行)
            * 全体パフォーマンス変化の簡潔な要約
            * 例: 「CPA 3.5%改善・CV 2.1%増加。主要因はGoogle広告のCVR向上(+15%)とYahoo!のコスト配分最適化(-10%)」
            
            2. 全体パフォーマンス変化分析
            * 主要指標の変化状況
            * CPA/CV変化の構造的要因分解
            
            3. 構造変化分析
            * 媒体間のコスト配分変化とその影響
            * 効率と規模のバランス評価
            
            4. 主要変化点サマリー
            * 最も影響の大きかった変化要因のリスト
            * 各要因の定量的影響度と説明
            
            5. 問題点と機会の特定
            * 優先的に対応すべき課題と好機
            * 具体的な推奨アクション
            """)
        
        st.subheader("その他の注意点")
        st.markdown("""
        1. 単純な数値比較だけでなく、背景にある戦略的意図を考慮します
        2. 日数の違いがある場合は、日平均値での比較も検討します
        3. CV数が極端に少ない媒体（5件未満等）はCPA等の変動が大きくなるため解釈に注意します
        4. 新規追加や停止された媒体については、特別に言及します
        5. 構造変化（コスト配分変更）と指標変化（CVRやCPC等）の影響を分離して評価します
        6. 階層的な変化点（媒体→キャンペーン→広告グループ）の連鎖を意識した分析を行います
        """)

# アプリケーションの実行
if __name__ == "__main__":
    main()
