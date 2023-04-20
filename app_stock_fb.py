import datetime
import os.path
import time

import pandas as pd
import requests
from web_utils import *
from twstock import Stock
from collections import defaultdict

dic_cfg = {
    'slp': 2,
    'get_txt_slp': 2,
    'is_force': True,
    'batch_size': 5,
    'sel_rat': 50,
    'sel_price': 500,
    'sel_corr': 0.8,
    'sel_rat_h': 90,
    'stocks': ['1232', '8435', '1593', '6803', '6214', '8083', '8349', '1521', '9933', '6277', '2382'],
    'stock_file': 'stock.csv',
    'header': {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    },
    'mops_path': 'mops',
    'mops_fin_path': 'mops_fin_0413',
    'month_path': 'Month',
    'per_latest_path': r'./PER/PER_Latest',
    'per_history_path': r'./PER/PER_History',
}

dic_fb_main = {
    'page': '',
    'sid_name': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                 '/html/body/div/div/main/div/div[3]/div/div/div[2]/div[1]/div[1]/h1'],
    '股價': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
             '/html/body/div/div/main/div/div[3]/div/div/div[2]/div[1]/div[1]/div/span/strong'],
    '大盤領先指標': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                     '/html/body/div/div/main/div/div[3]/div/div/div[4]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p'],
    '產業領先指標': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                     '/html/body/div/div/main/div/div[3]/div/div/div[4]/div[2]/div[2]/div/div[1]/div[1]/div[2]/p'],
}

dic_fb_revenue = {
    'page': '/revenue',
    '勝率(%)_營收': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                     '/html/body/div/div/main/div/div[3]/div/div[4]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
    '合理價差(%)_營收': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                         '/html/body/div/div/main/div/div[3]/div/div[4]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p[1]/span[1]'],
    '相關性_營收': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                    '/html/body/div/div/main/div/div[3]/div/div[6]/div[2]/div/div[2]/div[2]/div/p[2]'],

}

dic_fb_eps = {
    'page': '/eps',
    '勝率(%)_EPS': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                    '/html/body/div/div/main/div/div[3]/div/div[4]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
    '合理價差(%)_EPS': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                        '/html/body/div/div/main/div/div[3]/div/div[4]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p[1]/span[1]'],
    '相關性_EPS': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                   '/html/body/div/div/main/div/div[3]/div/div[6]/div[2]/div/div[2]/div[2]/div/p[2]'],

}

dic_fb_cash_dividend = {
    'page': '/cash_dividend ',
    '勝率(%)_殖利率': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                       '/html/body/div/div/main/div/div[3]/div/div[4]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
    '合理價差(%)_殖利率': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                           '/html/body/div/div/main/div/div[3]/div/div[4]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p[1]/span[1]'],
    '相關性_殖利率': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                      '/html/body/div/div/main/div/div[3]/div/div[6]/div[2]/div/div[2]/div[2]/div/p[2]'],

}

dic_fb_pick = {
    '月營收': r'https://www.findbillion.com/strategy/twstock/result?type=0&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
    'EPS': r'https://www.findbillion.com/strategy/twstock/result?type=1&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
    '現金股利': r'https://www.findbillion.com/strategy/twstock/result?type=2&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
}

dic_s_rename = {
    '月營收': '策略_營收',
    'EPS': '策略_EPS',
    '現金股利': '策略_殖利率',
}

dic_mops_fin_roe = {
    'page': 'MOPS_FIN_ROE',
    'btn_profitability ': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[6]/a'],
    'btn_ROE': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[6]/div/ul/li[5]/a'],
    'keyin_sid': ['keyin', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[1]/div[1]/input[1]'],
    'btn_start': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[3]/a[2]'],
    'btn_excel': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[2]/div[3]/div/div[1]/a'],
}

dic_mops_fin_roa = {
    'page': 'MOPS_FIN_ROA',
    'btn_profitability ': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[6]/a'],
    'btn_ROA': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[6]/div/ul/li[4]/a'],
    'keyin_sid': ['keyin', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[1]/div[1]/input[1]'],
    'btn_start': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[3]/a[2]'],
    'btn_excel': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[2]/div[3]/div/div[1]/a'],
}

dic_mops_fin_opm = {
    'page': 'MOPS_FIN_OPM',
    'btn_profitability ': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[6]/a'],
    'btn_OPM': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[6]/div/ul/li[2]/a'],
    'keyin_sid': ['keyin', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[1]/div[1]/input[1]'],
    'btn_start': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[3]/a[2]'],
    'btn_excel': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[2]/div[3]/div/div[1]/a'],
}

dic_mops_fin_debt_ratio = {
    'page': 'MOPS_FIN_DR',
    'btn_fin_structure ': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[3]/a'],
    'btn_DR': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[3]/div/ul/li[1]/a'],
    'keyin_sid': ['keyin', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[1]/div[1]/input[1]'],
    'btn_start': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[3]/a[2]'],
    'btn_excel': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[2]/div[3]/div/div[1]/a'],
}

dic_mops_fin_cash_flow = {
    'page': 'MOPS_FIN_CF',
    'btn_cash_flow ': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[8]/a'],
    'btn_CF': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[1]/ul/li[8]/div/ul/li[2]/a'],
    'keyin_sid': ['keyin', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[1]/div[1]/input[1]'],
    'btn_start': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[5]/div/div[3]/a[2]'],
    'btn_excel': ['click', dic_cfg['slp'], By.XPATH, '/html/body/div[2]/div[2]/div[3]/div/div[1]/a'],
}

dic_month_deal_lst = {
    'link': r'https://www.twse.com.tw/zh/trading/historical/fmsrfk.html',
    'clear_sid': ['clear_txt', dic_cfg['slp'], By.XPATH,
                  '/html/body/div[1]/div/div[2]/main/form/div/div[1]/div[2]/input'],
    'keyin_sid': ['keyin', dic_cfg['slp'], By.XPATH,
                  '/html/body/div[1]/div/div[2]/main/form/div/div[1]/div[2]/input'],
    'click_yr': ['click', dic_cfg['slp'], By.XPATH,
                 '/html/body/div[1]/div/div[2]/main/form/div/div[1]/div[1]/span/select[1]'],
    'sel_yr': ['sel_val', dic_cfg['slp'], By.ID, 'label0'],
    'click_sel': ['click', dic_cfg['slp'], By.XPATH,
                  '/html/body/div[1]/div/div[2]/main/form/div/div[1]/div[3]/button'],
    'click_sav': ['click', dic_cfg['slp'], By.XPATH,
                  '/html/body/div[1]/div/div[2]/main/div[2]/div[1]/button[2]'],

}

dic_month_deal_otc = {
    'link': r'https://www.tpex.org.tw/web/stock/statistics/monthly/st44.php?l=zh-tw',
    'clear_sid': ['clear_txt', dic_cfg['slp'], By.XPATH,
                  '/html/body/center/div[3]/div[2]/div[2]/div[1]/div[1]/div/form/input[1]'],
    'keyin_sid': ['keyin', dic_cfg['slp'], By.XPATH,
                  '/html/body/center/div[3]/div[2]/div[2]/div[1]/div[1]/div/form/input[1]'],
    'click_yr': ['click', dic_cfg['slp'], By.XPATH,
                 '/html/body/center/div[3]/div[2]/div[2]/div[1]/div[1]/div/form/select'],
    'sel_yr': ['sel_val', dic_cfg['slp'], By.ID, 'y_date1'],
    'click_sel': ['click', dic_cfg['slp'], By.XPATH,
                  '/html/body/center/div[3]/div[2]/div[2]/div[1]/div[1]/div/form/input[2]'],
    'click_sav': ['click', dic_cfg['slp'], By.XPATH,
                  '/html/body/center/div[3]/div[2]/div[2]/div[2]/table[1]/tbody/tr/td[3]/table/tbody/tr/td/button[2]'],

}

dic_dog = {
    'page': 'Dog',
    'risk_chk ': ['getText', dic_cfg['slp'], By.XPATH,
                  '/html/body/div[1]/div[2]/div/div[2]/div/div/div[3]/div[2]/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[1]'],
    'time_deposit_chk': ['getText', dic_cfg['slp'], By.XPATH,
                         '/html/body/div[1]/div[2]/div/div[2]/div/div/div[3]/div[2]/div[2]/div[2]/div[2]/div[1]/div[1]/div[1]'],

}

dic_wantrich_main = {
    'time': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
             '/html/body/div[4]/div[2]/div/div/article/div/div/div/div/section[1]/div/div[2]/ul/li[1]/div[1]/div[2]'],
    'grow': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
             '/html/body/div[4]/div[2]/div/div/article/div/div/div/div/section[1]/div/div[2]/ul/li[2]/div[1]/div[2]'],
    'tech': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
             '/html/body/div[4]/div[2]/div/div/article/div/div/div/div/section[1]/div/div[2]/ul/li[3]/div[1]/div[2]'],

}

dic_cmoney = {
    'ave_score ': ['getText', dic_cfg['slp'], By.XPATH,
                   '/html/body/div[1]/div/div/div/div[3]/div[3]/div/div[2]/aside[2]/div[1]/div/div[1]/div'],
    'comment': ['getText', dic_cfg['slp'], By.XPATH,
                '/html/body/div/div/div/div/div[3]/div[3]/div/div[2]/aside[2]/div[1]/div/div[1]/div/div/div[1]/div[2]'],

}

dic_cnyes = {'getHyperlinks': [],
             }


dic_yahoo_health = {'get_idx_Grow': ['getText', dic_cfg['slp'], By.XPATH, '/html/body/div[1]/div/div/div/div/div[5]/div[1]/div[1]/div/div[3]/div/div[2]/div/div[1]/div/span[1]'],
                    'get_idx_Stable': ['getText', dic_cfg['slp'], By.XPATH, '/html/body/div[1]/div/div/div/div/div[5]/div[1]/div[1]/div/div[3]/div/div[3]/div/div[1]/div/span[1]']}


dic_web_handle = {
    'dog': dic_dog,
    'WantRich': dic_wantrich_main,
    'CMoney': dic_cmoney,
    'Cnyes': dic_cnyes,
    'Yahoo': dic_yahoo_health,
}

dic_mops_fin = {
    'ROE': dic_mops_fin_roe,
    'ROA': dic_mops_fin_roa,
    'Operating_Margin': dic_mops_fin_opm,
    'DR': dic_mops_fin_debt_ratio,
    'CF': dic_mops_fin_cash_flow,
}

dic_fin = {
    '權益報酬率': 'ROE',
    '資產報酬率': 'ROA',
    '營業利益率': 'Operating_Margin',
    '負債佔資產比率': 'Debt_Ratio',
    '營業現金對負債比': 'Cash_Flow'
}

dic_url = {
    'FindBillion': 'https://www.findbillion.com/twstock/',
    'Yahoo': 'https://tw.stock.yahoo.com/quote/',
    'CMoney': 'https://www.cmoney.tw/forum/stock/',
    'FinLab': r'https://ai.finlab.tw/stock/?stock_id=',
    'WantRich': r'https://wantrich.chinatimes.com/tw-market/listed/stock/',
    'Yahoo_field': r'https://tw.stock.yahoo.com/t/nine.php?cat_id=%',
    'PChome': r'https://pchome.megatime.com.tw/stock/sto2/ock2/sid',
    'Wantgoo': r'https://www.wantgoo.com/stock/',
    'Cnyes': r'https://www.cnyes.com/twstock/',
    'dog': r'https://statementdog.com/analysis/',
    'tdcc': r'https://www.tdcc.com.tw/portal/zh/smWeb/qryStock',
}


def fn_get_stock_info(sid, url, webs, is_headless=True):
    df = pd.DataFrame()
    df['sid'] = [sid]

    # for dic in [dic_fb_main, dic_fb_revenue, dic_fb_eps, dic_fb_cash_dividend]:
    for dic in webs:
        page = dic["page"] if 'page' in dic.keys() else ''
        link = rf'{url}{sid}{page}'
        # link = rf'{url}{sid}{dic["page"]}'
        # link = rf'https://www.findbillion.com/twstock/{sid}{dic["page"]}'
        driver, action = fn_web_init(link, is_headless=is_headless)
        time.sleep(1)
        for k in dic.keys():
            if k != 'page':
                typ, slp, by, val = dic[k]

                if typ == 'getText':
                    try:
                        if sid == '0050':
                            if k == '股價':
                                val = '/html/body/div/div/main/div/div[3]/div/div[2]/div[1]/div[1]/div/span/strong'
                            elif k == '勝率(%)_營收':
                                val = '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'
                            elif k == '相關性_營收':
                                val = r'/html/body/div/div/main/div/div[3]/div/div[5]/div[2]/div/div[2]/div[2]/div/p[2]'
                            elif k == '合理價差(%)_營收':
                                val = r'/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p[1]/span[1]'
                            else:
                                pass

                        read = fn_web_get_text(driver, val, slp, by)
                        read = '不適用' if '不適用' in read else read
                        df[k] = read.split(' ')[-1] if k == 'sid_name' else read

                        if 'grow' in dic.keys():
                            score = int(read.split('%')[0].split(' ')[-1])
                            if score >= 80:
                                print(f'{sid} --> {k}: {read}')

                    except:
                        df[k] = ['']
                        print(f'{sid} get {dic["page"]} {k} Fail !')

        driver.close()

    df['date'] = [datetime.date.today()]
    return df


def fn_make_clickable(x):
    name = x
    sid = x if str(x).isnumeric() else x.split(" ")[0]
    url = rf'https://www.findbillion.com/twstock/{sid}'

    return '<a href="{}">{}</a>'.format(url, name)


def fn_fb_recommend_stock():
    df_smry = pd.DataFrame()
    for k in dic_fb_pick.keys():
        url = dic_fb_pick[k]
        r = requests.get(url, headers=dic_cfg['header'])
        dfs = pd.read_html(r.text)

        for df in dfs:
            df = df[[c for c in df.columns if 'Unnamed' not in c]]
            df = df[['公司名稱', '收盤價', '合理價差']]
            df.rename(columns={'合理價差': f'合理價差_{k}'}, inplace=True)
            for s in dic_fb_pick.keys():
                df[dic_s_rename[s]] = 1 if s == k else 0

            df_smry = pd.concat([df_smry, df], axis=0)

    for sid in df_smry['公司名稱'].unique():
        df_sid = df_smry[df_smry['公司名稱'] == sid]

        if df_sid.shape[0] > 1:
            dic_sid = {}
            for c in df_sid.columns:
                v_sel = ''
                for v in df_sid[c].values:
                    if '策略_' in c:
                        if str(v) != 'nan' and str(v) != '0':
                            v_sel = v
                            break
                    else:
                        if str(v) != 'nan':
                            v_sel = v
                            break
                dic_sid[c] = [v_sel]

            df_combined = pd.DataFrame(dic_sid)
            df_smry = df_smry[df_smry['公司名稱'] != sid]
            df_smry = pd.concat([df_smry, df_combined], axis=0, ignore_index=True)

    for idx in df_smry.index:
        df_smry.at[idx, '策略數'] = str(int(sum([1 for _ in df_smry.loc[idx, :].values if '%' in str(_)])))

    df_smry['日期'] = datetime.date.today()
    df_smry = df_smry.fillna("")

    return df_smry


def fn_find_billion(df, stocks=None, is_force=False):
    df['sid'] = df['公司名稱'].apply(lambda x: str(x.split(" ")[0]))
    # stock_ids = list(df['公司名稱'].apply(lambda x: str(x.split(" ")[0])))

    stock_ids = list(df['sid'])
    stock_ids = list(set(stock_ids)) if stocks is None else list(set(stock_ids + stocks))

    stock_file = dic_cfg['stock_file']
    if os.path.exists(stock_file):
        df_all = pd.read_csv(stock_file, na_filter=False, dtype=str, index_col=0)
        stock_ids = list(set(stock_ids + df_all['sid'].to_list()))
    else:
        df_all = pd.DataFrame()

    today = datetime.date.today()
    fr = 0
    stock_num = len(stock_ids)
    batch_num = dic_cfg['batch_size']

    for s in ['策略_營收', '策略_EPS', '策略_殖利率']:
        if s not in df_all.columns:
            df_all[s] = 0

    while fr < stock_num:
        to = min(fr + batch_num, stock_num)
        stock_ids_sub = stock_ids[fr: to]
        fr = fr + batch_num

        for sid in stock_ids_sub:
            is_bypass = False
            if 'sid' in df_all.columns:
                if sid in df_all['sid'].values.astype(str):
                    is_bypass = str(today) in df_all[df_all['sid'] == sid]['date'].values

            if is_bypass and is_force is False:
                pass
            else:
                t = time.time()

                url = rf'https://www.findbillion.com/twstock/'
                webs = [dic_fb_main, dic_fb_revenue, dic_fb_eps, dic_fb_cash_dividend]
                df_sid = fn_get_stock_info(sid, url, webs)

                # df_sid = fn_get_stock_info(sid)
                df_sid['耗時(秒)'] = int(time.time() - t)
                for s in ['策略_營收', '策略_EPS', '策略_殖利率']:
                    if sid in df['sid'].values:
                        # print(s, df[df['sid'] == sid][s], df[df['sid'] == sid][s].max())
                        df_sid[s] = df[df['sid'] == sid][s].max()
                    elif sid in df_all['sid'].values:
                        # print(s, df_all[df_all['sid'] == sid][s], df_all[df_all['sid'] == sid][s].max())
                        df_sid[s] = df_all[df_all['sid'] == sid][s].max()
                    else:
                        df_sid[s] = 0

                    df_sid[s] = 0 if df_sid[s].values[0] == '' else df_sid[s].values[0]

                df_sid = df_sid.astype(str)
                df_all = pd.concat([df_all, df_sid], axis=0, ignore_index=True)

                print(f'({stock_ids.index(sid) + 1}/{len(stock_ids)}): {sid} --> {df_sid["耗時(秒)"].values[0]}(秒)')

        df_all = df_all[df_all['股價'].apply(lambda x: str(x) != 'nan' and str(x) != '')]
        df_all = df_all[df_all['sid_name'].apply(lambda x: str(x) != 'nan')]

        recommend = df['sid'].unique().tolist()
        df_all['Recommend'] = df_all['sid'].apply(lambda x: 1 if x in recommend else 0)

        df_all.reset_index(drop=True, inplace=True)
        df_all.to_csv(dic_cfg['stock_file'], encoding='utf_8_sig')


def fn_want_rich(df, stocks=None):
    df['sid'] = df['公司名稱'].apply(lambda x: str(x.split(" ")[0]))

    stock_ids = list(df['sid'])
    stock_ids = list(set(stock_ids)) if stocks is None else list(set(stock_ids + stocks))

    url = r'https://wantrich.chinatimes.com/tw-market/listed/stock/'

    webs = [dic_wantrich_main]

    for sid in stock_ids:
        print('')
        print(f'{sid} --> {stock_ids.index(sid) + 1}/{len(stock_ids)}')
        df_sid = fn_get_stock_info(sid, url, webs)


def fn_is_parsing(is_force=dic_cfg["is_force"]):
    is_parsing = is_force

    if os.path.exists(dic_cfg['stock_file']):
        df_all = pd.read_csv(dic_cfg['stock_file'], na_filter=False, encoding='utf_8_sig', index_col=0, dtype=str)
        today = datetime.date.today()

        for s in dic_cfg['stocks']:
            if s not in df_all['sid'].values:
                print(f'parsing for {s}')
                is_parsing = is_parsing or True
                break

        if str(today) not in df_all['date'].values:
            print(f'parsing for {today}')
            is_parsing = is_parsing or True
    else:
        is_parsing = is_parsing or True

    return is_parsing


def fn_gen_stock_field_info():
    link1 = r'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
    link2 = r'https://isin.twse.com.tw/isin/C_public.jsp?strMode=4'

    df_field = pd.DataFrame()

    for link in [link1, link2]:
        res = requests.get(link)
        df = pd.read_html(res.text)[0]
        df.columns = df.iloc[0]
        df = df.iloc[2:]

        df['sid'] = df['有價證券代號及名稱'].apply(lambda x: x.split('　')[0])
        df = df[df['sid'].apply(lambda x: str(x).isnumeric())]
        df = df[df['產業別'].apply(lambda x: str(x) != 'nan')]
        df['產業別'] = df['產業別'].apply(
            lambda x: x.replace('工業', '').replace('事業', '').replace('業', '').replace('及週邊設備', '週邊'))

        df_field = pd.concat([df_field, df], axis=0)

    df_field.sort_values(by=['sid'], inplace=True, ignore_index=True)

    print(df_field['產業別'].unique())
    df_field.to_csv('stock_field.csv', encoding='utf_8_sig')


def fn_test():
    df = pd.read_csv(dic_cfg['stock_file'], na_filter=False, dtype=str, index_col=0)

    # 策略_營收,策略_EPS,策略_殖利率
    df_r = fn_fb_recommend_stock()

    df_r['sid'] = df_r['公司名稱'].apply(lambda x: str(x.split(" ")[0]))

    for c in ['策略_營收', '策略_EPS', '策略_殖利率']:
        df_s = df_r[df_r[c] == 1]
        for idx in df.index:
            if df.loc[idx, 'sid'] in df_s['sid'].values:
                df.at[idx, c] = '1'
            else:
                df.at[idx, c] = '0'

    df.to_csv(dic_cfg['stock_file'], encoding='utf_8_sig')


def fn_twstock(sid):
    stock = Stock(sid)  # 擷取台積電股價
    ma_p = stock.moving_average(stock.price, 5)  # 計算五日均價
    ma_c = stock.moving_average(stock.capacity, 5)  # 計算五日均量
    ma_p_cont = stock.continuous(ma_p)  # 計算五日均價持續天數
    ma_br = stock.ma_bias_ratio(5, 10)  # 計算五日、十日乖離值

    print(f'{sid} --> {stock.data[-1].date} {stock.price[-1]}元 {int(stock.capacity[-1] / 1000)}張')


def fn_mops_file_move():
    pass


def fn_mops_twse_parser():
    df_mops = pd.DataFrame()
    for root, dirs, files in os.walk(dic_cfg['mops_path']):
        for name in files:
            if 'mops.csv' in name:
                pass
            else:
                csv = os.path.join(root, name)
                try:
                    df = pd.read_csv(csv, encoding='ANSI')
                except:
                    df = pd.read_csv(csv)
                df['year'] = name.split('年')[0].split('司')[-1]
                df['market'] = name.split('公司')[0]
                print(f'{name} --> {df.shape}')
                df_mops = pd.concat([df_mops, df], ignore_index=True)

    print(df_mops.shape)
    df_mops.to_csv('mops.csv', encoding='utf_8_sig', index=False)


def fn_month_deal_download(dic, sid, years):
    link = dic['link']

    try:
        drv, act = fn_web_init(link, is_headless=False)
    except:
        time.sleep(10)
        drv, act = fn_web_init(link, is_headless=False)

    time.sleep(1)

    for year in years:
        for k in dic.keys():
            if k != 'page' and k != 'link':
                typ, slp, by, val = dic[k]
                key = ''
                if k == 'sel_yr':
                    key = year
                elif k == 'keyin_sid':
                    key = sid

                try:
                    fn_web_handle(drv, act, typ, slp, by, val, key)
                except:
                    pass

    time.sleep(2)
    drv.close()


def fn_mops_fin_download(dic, sids):
    assert len(sids) <= 10, f'MOPS FIN allow 10 sid max and input len(sids) = {len(sids)}'
    link = r'https://mopsfin.twse.com.tw/'

    try:
        drv, act = fn_web_init(link, is_headless=False)
    except:
        time.sleep(10)
        drv, act = fn_web_init(link, is_headless=False)

    time.sleep(1)

    for k in dic.keys():
        if k != 'page':
            typ, slp, by, val = dic[k]

            if typ == 'click':
                fn_web_click(drv, val, slp=slp)
            elif typ == 'keyin':
                for sid in sids:
                    idx = sids.index(sid) + 1
                    col = int(idx / 6) + 1
                    row = idx - 5 * (col - 1)
                    val_new = val.replace('/div[1]/input[1]', f'/div[{col}]/input[{row}]')
                    fn_web_send_keys(drv, val_new, sid, slp=slp)

    time.sleep(2)
    drv.close()


def fn_mops_fin_excl_2_csv(fin, is_new_season=False):
    df_mops_fin = pd.DataFrame()
    for root, dirs, files in os.walk(dic_cfg['mops_fin_path']):
        for name in files:
            if fin in name:
                xlsx = os.path.join(root, name)
                df = pd.read_excel(xlsx)
                df = df[df["年度/季度"].apply(lambda x: '(' in x)]
                df_mops_fin = pd.concat([df_mops_fin, df], ignore_index=True)

    df_mops_fin['sid'] = df_mops_fin["年度/季度"].apply(lambda x: x.split(' ')[0])
    df_mops_fin['name'] = df_mops_fin["年度/季度"].apply(lambda x: x.split(' ')[1])
    df_mops_fin['產業'] = df_mops_fin["年度/季度"].apply(lambda x: x.split(' ')[2])
    df_mops_fin['產業'] = df_mops_fin["產業"].apply(lambda x: x.replace("(", "").replace(")", ""))
    df_mops_fin['market'] = df_mops_fin["產業"].apply(lambda x: x[:2])
    df_mops_fin['產業'] = df_mops_fin["產業"].apply(lambda x: x[2:])

    cols_h = ['sid', 'name', 'market', '產業']
    cols_q = [c for c in df_mops_fin.columns if 'Q' in c]
    # cols_q.reverse()
    cols = cols_h + cols_q
    df_mops_fin = df_mops_fin[cols]  # [c for c in df_mops_fin.columns if c not in ["年度/季度"]+cols]]

    mops_fin_csv = f'mops_fin_{dic_fin[fin]}.csv'

    if is_new_season:
        df_mops_fin_new = df_mops_fin
    else:
        if os.path.isfile(mops_fin_csv):
            df_mops_fin_old = pd.read_csv(mops_fin_csv)
            df_mops_fin_old = df_mops_fin_old[cols]
            df_mops_fin_new = pd.concat([df_mops_fin_old, df_mops_fin], ignore_index=True)
        else:
            df_mops_fin_new = df_mops_fin

    df_mops_fin_new = df_mops_fin_new.sort_values(by=['market', '產業', 'sid'], ignore_index=True)
    df_mops_fin_new.to_csv(mops_fin_csv, encoding='utf_8_sig', index=False)


def fn_mops_fin(is_new_season=False):
    df_sid = pd.read_csv('stock.csv')
    df_fin = pd.read_csv('mops_fin_ROE.csv')
    sids = [str(s) for s in df_sid['sid'].unique() if len(str(s)) == 4] + ['1905']

    if is_new_season:
        pass
    else:
        sids = [s for s in sids if int(s) not in df_fin['sid'].values]

    print(f'New Stock ID num: {len(sids)}')

    if len(sids) > 0 or is_new_season:
        fr, to = 0, min(10, len(sids) + 1)

        while fr < len(sids):
            sids_sub = sids[fr:to]
            print(len(sids), fr, to, sids_sub)

            if len(sids_sub) == 0:
                break
            else:
                for it in dic_mops_fin.keys():
                    dic = dic_mops_fin[it]
                    fn_mops_fin_download(dic, sids_sub)
                    time.sleep(1)

            fr = to
            to = min(to + 10, len(sids))


def fn_get_company_report2():
    df = pd.read_csv('Company_Report_link.csv', na_filter=False, dtype=str, index_col=None)

    for i in df.index:
        sid = df.loc[i, 'sid']
        report = df.loc[i, 'report']
        if report == 'NA':
            try:
                dic_info = fn_get_web_info(sid, 'Cnyes')
                report2 = dic_info['getHyperlinks']
                df.at[i, 'report'] = report2
            except:
                pass

    df.to_csv('Company_Report_link2.csv', encoding='utf_8_sig', index=False)


def fn_get_company_report():
    if False:
        fn_get_company_report2()
    else:
        dic_cpy_rp = {'sid': [], 'report': []}
        df = pd.read_csv(dic_cfg['stock_file'], na_filter=False, dtype=str, index_col=0)
        sids = df['sid'].unique().tolist()
        sid_num = len(sids)
        web = 'Cnyes'
        for sid in sids:

            dic_cpy_rp['sid'].append(sid)

            try:
                dic_info = fn_get_web_info(sid, web)
                report_lnk = dic_info['getHyperlinks']
                status = 'Pass'
            except:
                report_lnk = 'NA'
                status = 'Fail !'

            dic_cpy_rp['report'].append(report_lnk)

            print(f'({sids.index(sid)}/{sid_num}) --> {sid} {status} {report_lnk}')

        df_report = pd.DataFrame(dic_cpy_rp)
        df_report.to_csv('Company_Report_link.csv', encoding='utf_8_sig', index=False)


def fn_get_web_info(sid, web):
    sid_info = sid
    if web == 'dog':
        sid_info = sid_info + '/stock-health-check'
    elif web == 'Cnyes':
        sid_info = sid_info + '/company/profile'
    elif web == 'Yahoo':
        sid_info = sid_info + '/health-check'

    link = dic_url[web] + sid_info

    drv, action = fn_web_init(link, is_headless=True)
    time.sleep(1)
    # print(link)

    dic = dic_web_handle[web]
    dic_info = {}

    for k in dic.keys():
        if k == 'page':
            pass
        elif k == 'getHyperlinks':
            urls = fn_web_get_hyperlink(drv, val="a", k1="M00")
            print(f'{sid} --> {urls}')
            dic_info[k] = urls[0]

        else:
            typ, slp, by, val = dic[k]
            if typ == 'getText':
                txt = fn_web_get_text(drv, val, slp=slp)
                # print(k, txt)
                dic_info[k] = txt

    dic_info['link'] = link
    time.sleep(1)
    drv.close()

    return dic_info


def fn_post_proc():
    stock_file = dic_cfg['stock_file']
    if os.path.exists(stock_file):
        df_all = pd.read_csv(stock_file, na_filter=False, dtype=str, index_col=0)

        for c in ['合理價_營收', '合理價_EPS', '合理價_殖利率']:
            r = c.replace('合理價', '合理價差(%)')

            for i in df_all.index:
                ip = df_all.loc[i, '股價']
                ir = df_all.loc[i, r]
                ia = ''
                if len(str(ip)) > 0 and len(str(ir)) > 0 and str(ir) != '0':
                    ia = round(float(ip) - float(ip) * float(ir) / 100, 1)
                    # print(f'{ip} {ir} --> {ia}')

                df_all.at[i, c] = ia

        df_all.to_csv(dic_cfg['stock_file'], encoding='utf_8_sig')


def fn_get_month_deal_data(is_force=False, years=6):
    try:
        df_month = pd.read_csv('Month.csv', na_filter=False, dtype=str)
        sids_existed = df_month['sid'].unique()
    except:
        sids_existed = []

    df = pd.read_csv('mops_fin_ROE.csv', na_filter=False, dtype=str)
    df_lst = df[df['market'] == '上市']
    df_otc = df[df['market'] == '上櫃']

    sids_lst = df_lst['sid'].unique().tolist()
    sids_otc = df_otc['sid'].unique().tolist()
    to_yr = datetime.datetime.today().year - 1911
    fr_yr = to_yr - years
    years_otc = [str(y) for y in range(fr_yr, to_yr + 1, 1)]
    years_lst = [f'民國 {y} 年' for y in range(fr_yr, to_yr + 1, 1)]

    sids_lst_len = len(sids_lst)
    sids_otc_len = len(sids_otc)
    for sid in sids_otc:
        if sid not in sids_existed or is_force:
            print(f'otc ({sids_otc.index(sid)}/{sids_otc_len}), downloading {sid}')
            fn_month_deal_download(dic_month_deal_otc, sid, years_otc)

    for sid in sids_lst:
        if sid not in sids_existed or is_force:
            print(f'lst ({sids_lst.index(sid)}/{sids_lst_len}), downloading {sid}')
            fn_month_deal_download(dic_month_deal_lst, sid, years_lst)


def fn_parse_month_data():
    def fn_m2s(x):
        m = int(x)
        if m <= 3:
            s = 'Q1'
        elif m <= 6:
            s = 'Q2'
        elif m <= 9:
            s = 'Q3'
        elif m <= 12:
            s = 'Q4'
        else:
            assert False

        return s

    df = pd.DataFrame()
    cols = ['market', 'sid', 'year', 'season', 'month', 'price', 'share']
    for root, dirs, files in os.walk(dic_cfg['month_path']):
        for f in files:
            sid = f.split('_')[1]
            year = f.split('_')[-1].split('.csv')[0]
            # print(f'{sid} {year} ')
            if 'FMSRFK_' in f:
                # print(f)
                mk = '上市'
                csv = os.path.join(dic_cfg['month_path'], f)
                df_lst = pd.read_csv(csv, encoding='ANSI', skiprows=[0])
                df_lst = df_lst.dropna(subset=['月份'])

                df_lst['market'] = mk

                df_lst['sid'] = sid
                df_lst['year'] = year
                df_lst['month'] = df_lst['月份'].apply(lambda x: int(x))
                df_lst['season'] = df_lst['month'].apply(fn_m2s)
                df_lst['price'] = df_lst['成交金額(A)']
                df_lst['share'] = df_lst['成交股數(B)']
                df_lst['price'] = df_lst['price'].apply(lambda x: int(str(x).replace(',', '')) * 1)
                df_lst['share'] = df_lst['share'].apply(lambda x: int(str(x).replace(',', '')) * 1)
                df_lst = df_lst[cols]
                df = pd.concat([df, df_lst], axis=0)
                # print(df_lst)
            elif 'ST44_' in f:
                # print(f)
                mk = '上櫃'
                csv = os.path.join(dic_cfg['month_path'], f)
                df_otc = pd.read_csv(csv, encoding='cp950', skiprows=[0, 1, 2, 3], encoding_errors='ignore')
                df_otc['market'] = mk
                df_otc['sid'] = sid
                df_otc['year'] = year
                df_otc['month'] = df_otc['月']
                df_otc['season'] = df_otc['month'].apply(fn_m2s)
                df_otc['price'] = df_otc['成交金額仟元(A)']
                df_otc['share'] = df_otc['成交股數仟股(B)']
                df_otc['price'] = df_otc['price'].apply(lambda x: int(str(x).replace(',', '')) * 1000)
                df_otc['share'] = df_otc['share'].apply(lambda x: int(str(x).replace(',', '')) * 1000)
                df_otc = df_otc[cols]
                # print(df_otc)
                df = pd.concat([df, df_otc], axis=0)
            else:
                pass

    df.reset_index(inplace=True, drop=True)
    # print(df.head(15))

    dic_1 = {}
    df_all = pd.DataFrame()
    for sid in df['sid'].unique():
        df_s = df[df['sid'] == sid]
        df_g = df_s.groupby(['year', 'season']).sum()

        # print(df_g.index)
        # print(df_g.index[0][0])

        dic_1['price'] = list(df_g['price'].values)
        dic_1['share'] = list(df_g['share'].values)

        df1 = pd.DataFrame(dic_1)
        df1['sid'] = sid
        df1['yr_sn'] = [str(y) for y in df_g.index]
        df1['year'] = df1['yr_sn'].apply(lambda x: int(x.split(',')[0].replace('(', '').replace("'", "")))
        df1['season'] = df1['yr_sn'].apply(lambda x: x.split(',')[-1].replace(')', '').replace("'", ""))
        df1 = df1[['sid', 'year', 'season', 'price', 'share']]

        df_all = pd.concat([df_all, df1])

    df_all['ave'] = df_all['price'] / df_all['share']
    df_all['ave'] = df_all['ave'].apply(lambda x: int(x))
    df_all.reset_index(drop=True, inplace=True)

    # print(df_all[df_all['sid'] == '3426'])

    df_all.to_csv('Month.csv', encoding='utf_8_sig')


def fn_get_yahoo_health():
    dic_yh = {'sid': [], 'grow': [], 'stable': [], 'market': [], 'link': []}
    df = pd.read_csv(dic_cfg['stock_file'], na_filter=False, dtype=str, index_col=0)
    df_field = pd.read_csv('stock_field.csv', na_filter=False, dtype=str, index_col=0)
    sids = df['sid'].unique().tolist()
    sid_num = len(sids)
    web = 'Yahoo'
    for sid in sids:

        if sid in df_field['sid'].values:
            mk = df_field[df_field['sid'] == sid]['市場別'].values[0]
        else:
            mk = '上櫃'

        dic_yh['market'].append(mk)
        dic_yh['sid'].append(sid)
        mk_code = '.TW' if mk == '上市' else '.TWO'

        try:
            dic_info = fn_get_web_info(sid+mk_code, web)
            grow = dic_info['get_idx_Grow']
            stable = dic_info['get_idx_Stable']
            link = dic_info['link']
            status = 'Pass'
        except:
            grow = 'NA'
            stable = 'NA'
            status = 'Fail !'
            link = 'NA'

        dic_yh['grow'].append(grow)
        dic_yh['stable'].append(stable)
        dic_yh['link'].append(link)

        print(f'({sids.index(sid)}/{sid_num}) --> {sid} {status} 成長:{grow} 穩健:{stable} {link}')

    df_yh = pd.DataFrame(dic_yh)
    df_yh['date'] = datetime.datetime.today()
    df_yh.to_csv('Yahoo_Health.csv', encoding='utf_8_sig', index=False)


def fn_main():
    t = time.time()

    # fn_test()
    # fn_twstock('1514')
    # fn_gen_stock_field_info()
    # fn_mops_twse_parser()

    # fn_get_month_deal_data(is_force=False, years=6)
    # fn_parse_month_data()

    if fn_is_parsing():
        df = fn_fb_recommend_stock()
        fn_find_billion(df, dic_cfg["stocks"], is_force=False)
        fn_post_proc()
        fn_get_yahoo_health()
        fn_get_company_report()


    # webs = ['Cnyes']
    # for w in webs:
    #     fn_get_web_info('2929', w)

    # is_new_season = False
    # fn_mops_fin(is_new_season=is_new_season)
    # 手動步驟 fn_move_file_TBD() Move download excl files to D:\02_Project\proj_python\proj_findbillion\mops_fin_0106
    # fn_mops_file_move()
    # for fin in dic_fin.keys():
    #     fn_mops_fin_excl_2_csv(fin, is_new_season=is_new_season)

    dur = int(time.time() - t)
    h = int(dur / 3600)
    m = int((dur - h * 3600) / 60)
    s = int((dur - h * 3600 - m * 60))

    print(f"爬蟲耗時: {dur} --> {h} hr {m} min {s} sec")


if __name__ == '__main__':
    fn_main()
