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
            'index': idx,
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
            'index': idx,
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
            'index': idx,
            'previous_cv': previous_cv,
            'current_cv': 0,
            'cv_change': cv_change,
            'contribution_rate': contribution_rate,
            'entry_status': "終了"
        })
    
    # DataFrameに変換
    contribution_df = pd.DataFrame(contribution_data)
    
    # マルチインデックスの場合の処理
    if len(group_by_cols) > 1:
        # インデックスの分解
        for i, col in enumerate(group_by_cols):
            contribution_df[col] = contribution_df['index'].apply(lambda x: x[i])
    else:
        # 単一インデックスの場合
        contribution_df[group_by_cols[0]] = contribution_df['index']
    
    # インデックス列を削除
    contribution_df = contribution_df.drop(columns=['index'])
    
    # 寄与率の絶対値で降順ソート
    contribution_df['abs_contribution'] = contribution_df['contribution_rate'].abs()
    contribution_df = contribution_df.sort_values('abs_contribution', ascending=False)
    contribution_df = contribution_df.drop(columns=['abs_contribution'])
    
    return contribution_df

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
            else:
                main_factor = "CPC"
                # 副要因判定
                if abs(cpm_change_rate) > abs(ctr_change_rate):
                    secondary_factor = "CPM"
                else:
                    secondary_factor = "CTR"
            
            # パフォーマンスの変化の説明を生成
            if current_cpa < previous_cpa:
                performance_change = "改善"
            else:
                performance_change = "悪化"
            
            factor_data.append({
                'index': idx,
                'previous_cpa': previous_cpa,
                'current_cpa': current_cpa,
                'cpa_change_rate': cpa_change_rate,
                'cvr_change_rate': cvr_change_rate,
                'cpc_change_rate': cpc_change_rate,
                'cpm_change_rate': cpm_change_rate,
                'ctr_change_rate': ctr_change_rate,
                'main_factor': main_factor,
                'secondary_factor': secondary_factor,
                'performance_change': performance_change
            })
        
        except Exception as e:
            st.warning(f"CPA変化要因分析でエラーが発生しました（{idx}）: {str(e)}")
            continue
    
    # DataFrameに変換
    factor_df = pd.DataFrame(factor_data)
    
    # マルチインデックスの場合の処理
    if len(group_by_cols) > 1:
        # インデックスの分解
        for i, col in enumerate(group_by_cols):
            factor_df[col] = factor_df['index'].apply(lambda x: x[i])
    else:
        # 単一インデックスの場合
        factor_df[group_by_cols[0]] = factor_df['index']
    
    # インデックス列を削除
    factor_df = factor_df.drop(columns=['index'])
    
    # CPA変化率の絶対値で降順ソート
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
                'index': idx,
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
    
    # DataFrameに変換
    pattern_df = pd.DataFrame(pattern_data)
    
    # マルチインデックスの場合の処理
    if len(group_by_cols) > 1:
        # インデックスの分解
        for i, col in enumerate(group_by_cols):
            pattern_df[col] = pattern_df['index'].apply(lambda x: x[i])
    else:
        # 単一インデックスの場合
        pattern_df[group_by_cols[0]] = pattern_df['index']
    
    # インデックス列を削除
    pattern_df = pattern_df.drop(columns=['index'])
    
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
        'media_patterns': media_patterns.to_dict('records')
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
    
    # CPAの計算
    previous_cpa = previous['cost'] / previous['conversions'] if previous['conversions'] != 0 else 0
    current_cpa = current['cost'] / current['conversions'] if current['conversions'] != 0 else 0
    cpa_change = ((current_cpa - previous_cpa) / previous_cpa) * 100 if previous_cpa != 0 else float('inf')
    
    # 日平均値の計算
    previous_daily_cv = previous['conversions'] / previous['days']
    current_daily_cv = current['conversions'] / current['days']
    daily_cv_change = ((current_daily_cv - previous_daily_cv) / previous_daily_cv) * 100 if previous_daily_cv != 0 else float('inf')
    
    # サマリーテーブル
    summary_table = f"""
| 指標 | 前期 | 当期 | 差分 | 変化率 |
|------|-----|-----|------|--------|
| インプレッション数 | {previous['impressions']:,} | {current['impressions']:,} | {current['impressions'] - previous['impressions']:,} | {imp_change:.1f}% |
| クリック数 | {previous['clicks']:,} | {current['clicks']:,} | {current['clicks'] - previous['clicks']:,} | {clicks_change:.1f}% |
| コスト | {previous['cost']:,}円 | {current['cost']:,}円 | {current['cost'] - previous['cost']:,}円 | {cost_change:.1f}% |
| コンバージョン数 | {previous['conversions']:,} | {current['conversions']:,} | {current['conversions'] - previous['conversions']:,} | {cv_change:.1f}% |
| CPA | {previous_cpa:,.0f}円 | {current_cpa:,.0f}円 | {current_cpa - previous_cpa:,.0f}円 | {cpa_change:.1f}% |
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
        
        cv_table += f"| {i} | {media_name} | {previous_cv:.0f} | {current_cv:.0f} | {cv_change:.0f} | {contribution_rate:.1f}% |\n"
    
    # CPA変化要因分析
    cpa_factors = data['cpa_change_factors'][:5]  # 上位5件
    
    cpa_table = "| 媒体名 | 前期CPA | 当期CPA | CPA変化率 | 主要因 | 副要因 |\n|------|------|------|------|------|------|\n"
    
    for item in cpa_factors:
        media_name = item.get('ServiceNameJA', 'Unknown')
        previous_cpa = item.get('previous_cpa', 0)
        current_cpa = item.get('current_cpa', 0)
        cpa_change_rate = item.get('cpa_change_rate', 0)
        main_factor = item.get('main_factor', 'Unknown')
        secondary_factor = item.get('secondary_factor', '-')
        
        cpa_table += f"| {media_name} | {previous_cpa:.0f}円 | {current_cpa:.0f}円 | {cpa_change_rate:.1f}% | {main_factor} | {secondary_factor} |\n"
    
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
    
    # データが読み込まれているか確認
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.info("サイドバーからデータを読み込んでください")
        return
    
    # データの概要を表示
    df = st.session_state['data']
    
    # タブの設定
    tab1, tab2, tab3 = st.tabs(["データ概要", "期間比較分析", "レポート出力"])
    
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
        st.dataframe(df.head(10))
        
        # 日次データの表示（折れ線グラフ）
        if 'Date' in df.columns:
            st.subheader("日次推移")
            
            # グラフ選択
            metric_option = st.selectbox(
                "指標選択",
                ["Impressions", "Clicks", "Cost", "Conversions", "CTR", "CVR", "CPC", "CPA", "CPM"],
                index=3  # デフォルトはConversions
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
            elif metric_option in ['Cost', 'CPC', 'CPA']:
                fig.update_layout(yaxis_ticksuffix='円')
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 媒体別データの表示（円グラフ）
        if 'ServiceNameJA' in df.columns:
            st.subheader("媒体別データ")
            
            # グラフ選択
            media_metric = st.selectbox(
                "指標選択（媒体別）",
                ["Impressions", "Clicks", "Cost", "Conversions"],
                index=2  # デフォルトはCost
            )
            
            # 媒体別集計
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
                    value=min(current_start + timedelta(days=6), max_date),
                    min_value=current_start,
                    max_value=max_date
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
        
        # OpenAI API Keyの設定
        st.subheader("分析設定")
        
        openai_api_key = st.text_input("OpenAI API Key (レポート生成に必要)", type="password")
        
        # 分析実行ボタン
        if st.button("分析実行"):
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
                    f"{current_total['Conversions']:,.0f}",
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
                    f"{current_cpa:,.0f}円",
                    f"{cpa_change:.1f}%",
                    delta_color="inverse" # CPAは下がる方がプラス表示
                )
            
            # 2. CV増減の寄与度ランキング
            st.write("#### CV増減の寄与度ランキング")
            
            cv_contribution = result['cv_contribution'].head(5)
            st.dataframe(cv_contribution)
            
            # 3. CPA変化要因分析
            st.write("#### CPA変化要因分析")
            
            cpa_factors = result['cpa_change_factors'].head(5)
            st.dataframe(cpa_factors)
            
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
    
    # タブ3: レポート出力
    with tab3:
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

# アプリケーションの実行
if __name__ == "__main__":
    main()
