
import datetime
import os.path
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
from collections import defaultdict
from workalendar.asia import Taiwan
from web_utils import *


dic_fb_main = {
    'page': '',
    'sid_name': ['getText', 2, By.XPATH, '/html/body/div/div/main/div/div[3]/div/div/div[2]/div[1]/div[1]/h1'],
    '股價': ['getText', 2, By.XPATH, '/html/body/div/div/main/div/div[3]/div/div/div[2]/div[1]/div[1]/div/span/strong'],
    '大盤領先指標': ['getText', 2, By.XPATH, '/html/body/div/div/main/div/div[3]/div/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p'],
    '產業領先指標': ['getText', 2, By.XPATH, '/html/body/div/div/main/div/div[3]/div/div/div[3]/div[2]/div[2]/div/div[1]/div[1]/div[2]/p'],
    }

dic_fb_revenue = {
    'page': '/revenue',
    '勝率(%)_營收': ['getText', 2, By.XPATH, '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
  }

dic_fb_eps = {
    'page': '/eps',
    '勝率(%)_EPS': ['getText', 2, By.XPATH, '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
  }

dic_fb_cash_dividend = {
    'page': '/cash_dividend',
    '勝率(%)_殖利率': ['getText', 2, By.XPATH, '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
}


def fn_get_stock_info(sid):
    df = pd.DataFrame()
    df['sid'] = [sid]

    for dic in [dic_fb_main, dic_fb_revenue, dic_fb_eps, dic_fb_cash_dividend]:
        link = rf'https://www.findbillion.com/twstock/{sid}{dic["page"]}'
        driver, action = fn_web_init(link, is_headless=True)

        for k in dic.keys():
            if k != 'page':
                typ, slp, by, val = dic[k]

                if typ == 'getText':
                    try:
                        read = fn_web_get_text(driver, val, slp, by)
                        read = '不適用' if '不適用' in read else read
                        df[k] = read.split(' ')[-1] if k == 'sid_name' else read
                    except:
                        df[k] = ['']
                        print(f'{sid} get {dic["page"]} {k} Fail !')

        df['date'] = [datetime.date.today()]
        # df['link'] = [link]
        driver.close()

    return df


def fn_make_clickable(x):
    name = x
    sid = x if str(x).isnumeric() else x.split(" ")[0]
    url = rf'https://www.findbillion.com/twstock/{sid}'

    return '<a href="{}">{}</a>'.format(url, name)


def fn_fb_recommend_stock():
    dic_fb_pick = {
        '月營收': r'https://www.findbillion.com/twstock/picking/result?type=0&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
        'EPS': 'https://www.findbillion.com/twstock/picking/result?type=1&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
        '現金股利': 'https://www.findbillion.com/twstock/picking/result?type=2&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
    }

    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    }

    df_smry = pd.DataFrame()
    for k in dic_fb_pick.keys():
        url = dic_fb_pick[k]
        r = requests.get(url, headers=header)
        dfs = pd.read_html(r.text)

        for df in dfs:
            df = df[[c for c in df.columns if 'Unnamed' not in c]]
            df = df[['公司名稱', '收盤價', '合理價差']]
            df.rename(columns={'合理價差': f'合理價差_{k}'}, inplace=True)
            df_smry = pd.concat([df_smry, df], axis=0)

        for sid in df_smry['公司名稱'].unique():
            df_sid = df_smry[df_smry['公司名稱'] == sid ]

            if df_sid.shape[0] > 1:
                dic_sid = {}
                for c in df_sid.columns:
                    v_sel = ''
                    for v in df_sid[c].values:
                        if str(v) != 'nan':
                            v_sel = v
                            break
                    dic_sid[c] = [v_sel]

                df_combined = pd.DataFrame(dic_sid)
                df_smry = df_smry[df_smry['公司名稱'] != sid]
                df_smry = pd.concat([df_smry, df_combined], axis=0, ignore_index=True)

    for idx in df_smry.index:
        df_smry.at[idx, '策略數'] = str(int(sum([1 for _ in df_smry.loc[idx, :].values if '%' in str(_)])))

    tzone = datetime.timezone(datetime.timedelta(hours=8))
    df_smry['日期'] = datetime.date.today(tzone=tzone)
    df_smry = df_smry.fillna("")
    # print(df_smry)

    # st.markdown(f'### Summary: {df_smry.shape[0]}檔')
    # st.dataframe(df_smry, use_container_width=True, width=1000)

    df_show = df_smry.copy()
    df_show['公司名稱'] = df_show['公司名稱'].apply(fn_make_clickable)
    st.markdown(f'### Select: {df_show.shape[0]}檔')
    st.write(df_show.to_html(escape=False, index=True), unsafe_allow_html=True)

    return df_smry


def fn_find_billion(df, stocks=None):
    stock_ids = list(df['公司名稱'].apply(lambda x: str(x.split(" ")[0])))

    if stocks is None:
        stock_ids = list(set(stock_ids))
    else:
        stock_ids = list(set(stock_ids + stocks))

    stock_file = 'stock.csv'
    if os.path.exists(stock_file):
        df_all = pd.read_csv(stock_file, na_filter=False, dtype=str, index_col=0)
    else:
        df_all = pd.DataFrame()

    today = datetime.date.today()
    for sid in stock_ids:
        is_bypass = False
        if 'sid' in df_all.columns:
            if sid in df_all['sid'].values.astype(str):
                is_bypass = str(today) in df_all[df_all['sid'] == sid]['date'].values

        if is_bypass:
            pass
        else:
            t = time.time()
            df_sid = fn_get_stock_info(sid)
            df_sid['耗時(秒)'] = int(time.time() - t)
            df_sid = df_sid.astype(str)
            df_all = pd.concat([df_all, df_sid], axis=0, ignore_index=True)

    st.write('')
    st.markdown(f'### 勝率分析: {df_all.shape[0]}檔')
    st.dataframe(df_all, use_container_width=True, width=1200)
    df_all.to_csv('stock.csv', encoding='utf-8-sig')

    df_all['篩選'] = 0
    sel_win_rat = 50
    sel_max_price = 500

    for idx in df_all.index:
        for c in df_all.columns:
            if '勝率' in c:
                v = df_all.loc[idx, c]
                if v != '':
                    if int(v) >= sel_win_rat:
                        df_all.at[idx, '篩選'] = 1
                        break

    df_sel = df_all[df_all['篩選'] == 1]
    df_sel = df_sel[df_sel['股價'].apply(lambda x: float(x) < sel_max_price if x != '' else True)]

    df_sel = df_sel[[c for c in df_sel.columns if '篩選' not in c and '耗時' not in c]]
    df_sel.reset_index(drop=True, inplace=True)

    st.write('')
    st.markdown(f'### 由 {df_all.shape[0]}檔中 篩選 任一策略的勝率大於 {sel_win_rat}% 且 股價小於 {sel_max_price}元 的有: {df_sel.shape[0]}檔')

    if df_sel.shape[0] > 0:
        df_show = df_sel.copy()
        df_show['sid'] = df_show['sid'].apply(fn_make_clickable)
        st.write(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)


def fn_main():
    st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto",
                       menu_items=None)
    df = fn_fb_recommend_stock()
    stocks = ['2454']
    try:
        fn_find_billion(df, stocks)
    except:
        pass


if __name__ == '__main__':
    fn_main()
