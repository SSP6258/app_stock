import datetime
import os.path
import time

import pandas as pd
import requests
from web_utils import *

dic_cfg = {
    'get_txt_slp': 1,
    'is_force': True,
    'batch_size': 5,
    'sel_rat': 50,
    'sel_price': 500,
    'sel_corr': 0.75,
    'sel_rat_h': 90,
    'stocks': ['1535'],
    'stock_file': 'stock.csv',
    'header': {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    },
}

dic_fb_main = {
    'page': '',
    'sid_name': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                 '/html/body/div/div/main/div/div[3]/div/div/div[2]/div[1]/div[1]/h1'],
    '股價': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
           '/html/body/div/div/main/div/div[3]/div/div/div[2]/div[1]/div[1]/div/span/strong'],
    '大盤領先指標': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
               '/html/body/div/div/main/div/div[3]/div/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p'],
    '產業領先指標': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
               '/html/body/div/div/main/div/div[3]/div/div/div[3]/div[2]/div[2]/div/div[1]/div[1]/div[2]/p'],
}

dic_fb_revenue = {
    'page': '/revenue',
    '勝率(%)_營收': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                 '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
    '合理價差(%)_營收': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                   '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p[1]/span[1]'],
    '相關性_營收': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
               '/html/body/div/div/main/div/div[3]/div/div[5]/div[2]/div/div[2]/div[2]/div/p[2]'],

}

dic_fb_eps = {
    'page': '/eps',
    '勝率(%)_EPS': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                  '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
    '合理價差(%)_EPS': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                    '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p[1]/span[1]'],
    '相關性_EPS': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                '/html/body/div/div/main/div/div[3]/div/div[5]/div[2]/div/div[2]/div[2]/div/p[2]'],

}

dic_fb_cash_dividend = {
    'page': '/cash_dividend',
    '勝率(%)_殖利率': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                  '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[2]/div/div/div[1]/div[2]/p[1]/span[2]'],
    '合理價差(%)_殖利率': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                    '/html/body/div/div/main/div/div[3]/div/div[3]/div[2]/div[1]/div/div[1]/div[1]/div[2]/p[1]/span[1]'],
    '相關性_殖利率': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
                '/html/body/div/div/main/div/div[3]/div/div[5]/div[2]/div/div[2]/div[2]/div/p[2]'],

}

dic_fb_pick = {
    '月營收': r'https://www.findbillion.com/twstock/picking/result?type=0&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
    'EPS': 'https://www.findbillion.com/twstock/picking/result?type=1&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
    '現金股利': 'https://www.findbillion.com/twstock/picking/result?type=2&subtypeStep2=4&subtypeStep3=4&subtypeStep4=0',
}

dic_s_rename = {
    '月營收': '策略_營收',
    'EPS': '策略_EPS',
    '現金股利': '策略_殖利率',
}

dic_wantrich_main = {
    'time': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
             '/html/body/div[4]/div[2]/div/div/article/div/div/div/div/section[1]/div/div[2]/ul/li[1]/div[1]/div[2]'],
    'grow': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
             '/html/body/div[4]/div[2]/div/div/article/div/div/div/div/section[1]/div/div[2]/ul/li[2]/div[1]/div[2]'],
    'tech': ['getText', dic_cfg['get_txt_slp'], By.XPATH,
             '/html/body/div[4]/div[2]/div/div/article/div/div/div/div/section[1]/div/div[2]/ul/li[3]/div[1]/div[2]'],

}


def fn_get_stock_info(sid, url, webs):
    df = pd.DataFrame()
    df['sid'] = [sid]

    # for dic in [dic_fb_main, dic_fb_revenue, dic_fb_eps, dic_fb_cash_dividend]:
    for dic in webs:
        page = dic["page"] if 'page' in dic.keys() else ''
        link = rf'{url}{sid}{page}'
        # link = rf'{url}{sid}{dic["page"]}'
        # link = rf'https://www.findbillion.com/twstock/{sid}{dic["page"]}'
        driver, action = fn_web_init(link, is_headless=True)
        time.sleep(1)
        for k in dic.keys():
            if k != 'page':
                typ, slp, by, val = dic[k]

                if typ == 'getText':
                    try:
                        if sid == '0050' and k == '股價':
                            val = '/html/body/div/div/main/div/div[3]/div/div[2]/div[1]/div[1]/div/span/strong'
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


def fn_find_billion(df, stocks=None):
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

            if is_bypass:
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

        # df_all = df_all[]
        df_all.to_csv(dic_cfg['stock_file'], encoding='utf_8_sig')


def fn_want_rich(df, stocks=None):
    df['sid'] = df['公司名稱'].apply(lambda x: str(x.split(" ")[0]))

    stock_ids = list(df['sid'])
    stock_ids = list(set(stock_ids)) if stocks is None else list(set(stock_ids + stocks))

    url = r'https://wantrich.chinatimes.com/tw-market/listed/stock/'

    webs = [dic_wantrich_main]

    for sid in stock_ids:
        print('')
        print(f'{sid} --> {stock_ids.index(sid)+1}/{len(stock_ids)}')
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


def fn_main():
    t = time.time()

    # fn_test()

    if fn_is_parsing():
        df = fn_fb_recommend_stock()
        fn_find_billion(df, dic_cfg["stocks"])
        # fn_want_rich(df, dic_cfg["stocks"])

    dur = int(time.time() - t)
    h = int(dur / 3600)
    m = int((dur - h * 3600) / 60)
    s = int((dur - h * 3600 - m * 60))

    print(f"爬蟲耗時: {dur} --> {h} hr {m} min {s} sec")


if __name__ == '__main__':
    fn_main()
