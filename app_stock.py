
import datetime
import numpy as np
import pandas as pd
import random
import requests
import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
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
    sid = x.split(" ")[0]
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
                df_smry = df_smry[df_smry[''] != sid]
                df_smry = pd.concat([df_smry, df_combined], axis=0, ignore_index=True)

        for idx in df_smry.index:
            df_smry.at[idx, '策略數'] = str(int(sum([1 for _ in df_smry.loc[idx, :].values if '%' in str(_)])))

        df_smry['日期'] = datetime.date.today()
        df_smry = df_smry.fillna("")
        # print(df_smry)

        st.markdown(f'### Summary: {df_smry.shape[0]}檔')
        st.dataframe(df_smry, use_container_width=True, width=1000)

        df_smry['公司名稱'] = df_smry['公司名稱'].apply(fn_make_clickable)
        st.markdown(f'### Select: {df_smry.shape[0]}檔')
        st.write(df_smry.to_html(escape=False, index=True), unsafe_allow_html=True)


def fn_find_billion():
    stock_ids = ['2330', '0050']

    df_all = pd.DataFrame()
    for sid in stock_ids:
        df_sid = fn_get_stock_info(sid)
        df_all = pd.concat([df_all, df_sid], axis=0, ignore_index=True)

    print(df_all)
    st.dataframe(df_all, use_container_width=True, width=1200)


def fn_main():
    st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto",
                       menu_items=None)
    # fn_fb_recommend_stock()
    fn_find_billion()


if __name__ == '__main__':
    fn_main()
