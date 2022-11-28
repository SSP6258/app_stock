import pandas as pd
import streamlit as st
from app_stock_fb import *
from collections import defaultdict

dic_url = {
    'FindBillion': 'https://www.findbillion.com/twstock/',
    'Yahoo': 'https://tw.stock.yahoo.com/quote/',
    'Cmoney': 'https://www.cmoney.tw/forum/stock/',
    'FinLab': r'https://ai.finlab.tw/stock/?stock_id=',
    'WantRich': r'https://wantrich.chinatimes.com/tw-market/listed/stock/',
}

dic_sel = {
    'pick': []
}


def fn_make_clickable(x):
    name = x
    sid = x if str(x).isnumeric() else x.split(" ")[0]
    url = rf'{dic_url["WantRich"]}{sid}'

    return '<a href="{}">{}</a>'.format(url, name)


def fn_click_name(sid, name, url):
    url = rf'{url}{sid}'

    return '<a href="{}">{}</a>'.format(url, name)


def fn_color_map(x):
    css = 'background-color: white; color: black'
    css_h = 'background-color: pink; color: black'
    if '%' in str(x) and '%%' not in str(x):
        if float(x.replace('%', '')) >= 50.0:
            css = css_h
    elif str(x) == '1':
        css = css_h
    elif str(x) in dic_sel['pick']:
        css = css_h

    return css


def fn_pick_date(df, col_pick, col_date):
    df_sel_pick = pd.DataFrame()
    for sid in df[col_pick].unique():
        df_sid = df[df[col_pick] == sid]
        df_sid_pick = df_sid[df_sid[col_date].apply(lambda x: x in [min(df_sid[col_date]), max(df_sid[col_date])])]
        df_sel_pick = pd.concat([df_sel_pick, df_sid_pick], axis=0)

    return df_sel_pick


def fn_stock_sel(df_all):
    for idx in df_all.index:
        for c in df_all.columns:
            if '勝率' in c:
                v = df_all.loc[idx, c]
                corr = df_all.loc[idx, '相關性_' + c.split('_')[-1]].split(' ')[-1]
                if v != '' and corr != '':
                    if int(v) >= dic_cfg["sel_rat"] and float(corr) > dic_cfg["sel_corr"]:
                        df_all.at[idx, "篩選"] = 1
                        break
                    elif int(v) >= dic_cfg["sel_rat_h"]:
                        df_all.at[idx, "篩選"] = 1
                        break

    for s in df_all[df_all['篩選'] == 1]['sid'].unique():
        df_sid = df_all[df_all['sid'] == s]
        s_date = df_sid[df_sid['篩選'] == 1]['date'].min()
        for idx in df_all.index:
            if df_all.loc[idx, 'sid'] == s and df_all.loc[idx, 'date'] > s_date:
                df_all.at[idx, "篩選"] = 1

    df_sel = df_all[df_all["篩選"] == 1]
    df_sel = df_sel[df_sel["股價"].apply(lambda x: float(x) < dic_cfg["sel_price"] if x != '' else True)]
    df_sel = df_sel[[c for c in df_sel.columns if '篩選' not in c and
                     '耗時' not in c and
                     '合理價差' not in c]]

    df_sel.reset_index(drop=True, inplace=True)

    df_sel_pick = fn_pick_date(df_sel, 'sid', 'date')

    df_sel_pick.reset_index(drop=True, inplace=True)

    return df_sel_pick


def fn_st_add_space(s):
    for _ in range(s):
        st.write('')


def fn_st_stock_sel(df_all):
    df_sel = fn_stock_sel(df_all)

    if df_sel.shape[0] > 0:
        def f(s, name):
            return s if name == '' else name

        df_sel['sid_name'] = df_sel.apply(lambda x: f(x.sid, x.sid_name), axis=1)
        df_sel['sid_name'] = df_sel['sid_name'].apply(lambda x: x.replace('0050', '元大台灣50'))
        df_sel['max'] = df_sel[[c for c in df_sel.columns if '勝率' in c]].max(axis=1)
        df_sel.sort_values(by=['max'], ascending=False, inplace=True, ignore_index=True)

        c1, c2 = st.columns([2.5, 1])
        sel_sid = list(df_sel["sid_name"].unique())
        sel_num = df_sel["sid"].nunique()
        c1.error(f'#### 👉 篩選出{sel_num}檔: {", ".join(sel_sid)}')
        fn_st_add_space(1)

        cs = st.columns(sel_num + 4)
        # cs[0].markdown('# 👀')
        cs[0].metric('關注個股', '👀', '績效/天數', delta_color='inverse')
        # j = 1
        profs = []
        metrics = []
        for i in range(sel_num):
            sid_name = sel_sid[i]

            df_sid = df_sel[df_sel['sid_name'] == sid_name]
            df_sid['date'] = pd.to_datetime(df_sid['date'])
            sid = df_sid['sid'].values[0]
            price_old = df_sid[df_sid['date'] == min(df_sid['date'])]['股價'].values[0]
            price_new = df_sid[df_sid['date'] == max(df_sid['date'])]['股價'].values[0]
            # price_old, price_new = df_sid['股價'].values[0], df_sid['股價'].values[-1]
            if str(price_old) != '' and str(price_new) != '':
                diff = float(price_new) - float(price_old)
                prof = round(100 * diff / float(price_old), 2)

                delta_time = max(df_sid['date']) - min(df_sid['date'])
                days = delta_time.days

                profs.append(prof + 0.000001 * i)
                metrics.append([f'⭐{sid_name} {sid}', f'{price_new}', f'{prof}% / {days}天'])

                # cs[j].metric(f'{sid_name} {sid}', f'{price_new}', f'{prof}% / {days}天', delta_color='inverse')
                # j = j + 1

        profs_sort = sorted(profs, reverse=True)

        j = 0
        for p in profs:
            i = profs_sort.index(p)
            cs[i + 1].metric(*metrics[j], delta_color='inverse')
            j += 1

        df_sel = df_sel[[c for c in df_sel.columns if 'max' not in c]]
        df_show = df_sel.copy()
        df_show.sort_values(by=['sid_name', 'date'], ascending=[True, False], inplace=True, ignore_index=True)
        df_show = df_show[['date'] + [c for c in df_show.columns if c != 'date']]

        # df_show['股價'] = df_show['股價'].apply(lambda x: str(x) if x == '' else '🔺' + str(x))

        dic_page = {
            '營收': '/revenue',
            'EPS': '/eps',
            '殖利率': '/cash_dividend',
        }

        def fn_sel(x):
            if x == '' or x == '0':
                return '不適用'
            elif int(x) < dic_cfg['sel_rat']:
                return str(x) + '%'
            else:
                return str(x) + '% 👍'

        for c in df_show.columns:
            if '勝率' in c:
                df_show[c] = df_show[c].apply(fn_sel)
                page = dic_page[c.split('_')[-1]]
                df_show[c] = df_show.apply(lambda x: fn_click_name(x['sid'] + page, x[c], dic_url['FindBillion']),
                                           axis=1)
            if '相關性' in c:
                df_show[c] = df_show[c].apply(lambda x: x.split(' ')[-1])

        df_show['股票代碼'] = df_show['sid'].apply(fn_make_clickable)
        df_show['股票名稱'] = df_show.apply(lambda x: fn_click_name(x["sid"], x["sid_name"], dic_url['Yahoo']), axis=1)
        df_show['股價'] = df_show.apply(
            lambda x: fn_click_name(x["sid"] + '/technical-analysis', x["股價"], dic_url['Yahoo']), axis=1)

        show_cols_order = ['股票名稱', '股票代碼', 'date', '股價', '大盤領先指標', '產業領先指標',
                           '勝率(%)_營收', '相關性_營收', '勝率(%)_EPS', '相關性_EPS',
                           '勝率(%)_殖利率', '相關性_殖利率']

        df_show = df_show[[c for c in show_cols_order if c in df_show.columns]]
        # ➡
        show_cols_rename = {'date': '日期',
                            '股票名稱': '名稱',
                            '股票代碼': '代碼',
                            '大盤領先指標': '大盤<br>領先指標',
                            '產業領先指標': '產業<br>領先指標',
                            '勝率(%)_營收': '營收<br>勝率',
                            '相關性_營收': '營收<br>相關性',
                            '勝率(%)_EPS': 'EPS<br>勝率',
                            '相關性_EPS': 'EPS<br>相關性',
                            '勝率(%)_殖利率': '殖利率<br>勝率',
                            '相關性_殖利率': '殖利率<br>相關性'}

        df_show.rename(columns=show_cols_rename, inplace=True)
        fn_st_add_space(1)
        st.write(df_show.to_html(escape=False, index=True), unsafe_allow_html=True)


def fn_show_bar(df, stg=None, x='策略選股', y=None, num=None, title=False):
    # fn_st_add_space(3)
    df_win = df[df["績效(%)"] > 0]
    win_rate = round(10 * df_win.shape[0] / df.shape[0], 1)
    # st.markdown(f'#### 依{stg}選股 勝率: {win_rate}成, {df_win.shape[0]}/{df.shape[0]}')
    if title:
        st.markdown(f'#### 📊 {num}檔個股的 績效 v.s. "{stg}" 策略指標')
    st.bar_chart(data=df, x=x, y=y,
                 width=0, height=500,
                 use_container_width=True)


def fn_st_chart_bar(df):
    df_pick = fn_pick_date(df, '代碼', '日期')
    df_pick['日期'] = pd.to_datetime(df_pick['日期'])
    df_pick['股價'] = df_pick['股價'].astype(float)

    # st.write(df_pick)

    for c in df_pick.columns:
        if '勝率' in c or '合理價差' in c:
            df_pick[c] = df_pick[c].apply(lambda x: 0 if x == '' else round(float(x.replace('%', '')) / 10, 1))

    dic_sid = defaultdict(list)
    for sid in df_pick['代碼'].unique():
        df_sid = df_pick[df_pick['代碼'] == sid]
        df_sid.reset_index(drop=True, inplace=True)

        gain = (df_sid['股價'].values[0] - df_sid['股價'].values[-1]) / df_sid['股價'].values[-1]
        gain = round(100 * gain, 2)
        gain_str = str(gain) + '%'

        dt = max(df_sid['日期']) - min(df_sid['日期'])

        dic_sid['績效(%)'].append(gain)
        dic_sid['績效_str'].append(gain_str)
        dic_sid['天數'].append(-1*dt.days)

        for c in df_sid.columns:
            df_sid_old = df_sid[df_sid['日期'] == min(df_sid['日期'])]
            df_sid_new = df_sid[df_sid['日期'] == max(df_sid['日期'])]
            dic_sid[c+'_old'].append(df_sid.loc[df_sid_old.index[0], c])

            dic_sid[c+'_new'].append(df_sid.loc[df_sid_new.index[0], c])

    df_sids = pd.DataFrame(dic_sid)

    st.markdown(f'#### 📊 {df_sids.shape[0]}檔個股的 績效 v.s. 策略指標')

    # ==========

    for c in [c for c in df_sids.columns if '相關性' in c]:
        df_sids[c] = df_sids[c].apply(lambda x: 0 if x == '' else float(x) * 10)

    watch = [c for c in df_sids.columns if '勝率' in c or '合理' in c or '相關性' in c]

    if 'stra' not in st.session_state.keys():
        st.session_state['stra'] = ['營收']

    kpis = ['績效(%)', '天數'] + [w for w in watch if w.split('_')[0] in st.session_state['stra']]
    if 'kpi' not in st.session_state.keys():
        st.session_state['kpi'] = [k for k in kpis if k != '天數']

    if 'order' not in st.session_state.keys():
        st.session_state['order'] = '績效(%)'

    # ==========

    with st.form(key='Form1'):

        cs = st.columns([2, 5])
        st.session_state['stra'] = cs[0].multiselect(f'選擇策略:', options=['營收', 'EPS', '殖利率'], default=st.session_state['stra'], key='straxx')

        dft_kpi = [k for k in st.session_state['kpi'] if k in kpis]
        st.session_state['kpi'] = cs[1].multiselect(f'選擇指標:', options=kpis, default=dft_kpi, key='kpixxx')

        fn_st_add_space(1)
        submit = st.form_submit_button('選擇')

    if len(st.session_state['kpi']) > 0:
        c1, c2 = st.columns([2, 5])
        dft_idx = st.session_state['kpi'].index(st.session_state['order']) if st.session_state['order'] in \
                                                                              st.session_state['kpi'] else 0
        st.session_state['order'] = c1.selectbox(f'選擇排序方式:', options=st.session_state['kpi'], index=dft_idx)

        df_sids.sort_values(by=[st.session_state['order']], inplace=True, ascending=False, ignore_index=True)
        df_sids.reset_index(inplace=True)

        def fn_add_digit(x):
            for i in range(3 - len(str(x))):
                x = '0' + str(x)
            return str(x)

        df_sids['index'] = df_sids['index'].apply(fn_add_digit)
        df_sids['策略選股'] = df_sids['index'] + ' ' + df_sids['名稱'] + ' ' + df_sids['代碼']
        df_sids['策略選股'] = df_sids['策略選股'].apply(lambda x: x + '⭐' if x.split(' ')[1] in dic_sel['pick'] else x)
        fn_st_add_space(2)
        fn_show_bar(df_sids[df_sids['績效(%)'] > 0], stg=','.join(st.session_state['stra']), y=st.session_state['kpi'], num=df_sids.shape[0], title=False)
        fn_show_bar(df_sids[df_sids['績效(%)'] <= 0], stg=','.join(st.session_state['stra']), y=st.session_state['kpi'], num=df_sids.shape[0])


def fn_st_stock_all(df_all):

    df_all = df_all[[c for c in df_all.columns if '耗時' not in c]]
    show_cols_rename = {'date': '日期',
                        'sid_name': '名稱',
                        'sid': '代碼',
                        '合理價差(%)_營收': '營收_合理價差',
                        '合理價差(%)_EPS': 'EPS_合理價差',
                        '合理價差(%)_殖利率': '殖利率_合理價差',

                        '勝率(%)_營收': '營收_勝率',
                        '勝率(%)_EPS': 'EPS_勝率',
                        '勝率(%)_殖利率': '殖利率_勝率',

                        '相關性_營收': '營收_相關性',
                        '相關性_EPS': 'EPS_相關性',
                        '相關性_殖利率': '殖利率_相關性',
                        }

    df_all.rename(columns=show_cols_rename, inplace=True)
    col_order = ['名稱', '代碼']
    col_order = col_order + [c for c in df_all.columns if c not in col_order]

    k_cols = {}
    for k in ['營收', 'EPS', '殖利率']:
        k_cols[k] = []
        for c in col_order:
            if k in c and '策略_' not in c:
                k_cols[k].append(c)

        for d in k_cols[k]:
            col_order.remove(d)

        col_order = col_order + k_cols[k]

    df_all = df_all[col_order]
    df_all.sort_values(by=['代碼', '日期'], ascending=[True, False], inplace=True, ignore_index=True)

    for c in df_all.columns:
        if '相關性' in c:
            df_all[c] = df_all[c].apply(lambda x: x.split(' ')[-1] if '相關' in x else x)

        if '勝率' in c:
            df_all[c] = df_all[c].apply(lambda x: x if x == '' else str(x) + '%')

        if '價差' in c:
            df_all[c] = df_all[c].apply(lambda x: x if x == '' else str(x) + '%')
            df_all[c] = df_all[c].apply(lambda x: x.replace('%%', '%'))

    def fn_rename(name, sid):
        return sid if name == '' else name

    df_all['名稱'] = df_all.apply(lambda x: fn_rename(x['名稱'], x['代碼']), axis=1)
    dic_sel['pick'] = [c for c in list(df_all[df_all['篩選'] == 1]['名稱'].unique()) if c != '']

    fn_st_chart_bar(df_all)

    df_all_show = df_all.style.applymap(fn_color_map, subset=[c for c in df_all.columns if '勝率' in c] + ['篩選', '名稱'])

    fn_st_add_space(3)
    st.markdown(f'#### 📡 {df_all["代碼"].nunique()}檔 台股的 "勝率" 與 "合理價" 分析:')
    st.dataframe(df_all_show, width=None, height=500)


def fn_st_stock_main():
    stock_file = dic_cfg['stock_file']
    if not os.path.exists(stock_file):
        st.error(f"{stock_file} NOT Exist !!!")
        return

    df_all = pd.read_csv(stock_file, na_filter=False, encoding='utf_8_sig', index_col=0, dtype=str)
    df_all["篩選"] = 0

    df_all['date_dt'] = pd.to_datetime(df_all['date'])
    fr = min(df_all['date'])
    to = max(df_all['date'])
    dl = max(df_all['date_dt']) - min(df_all['date_dt'])
    df_all.drop(columns=['date_dt'], inplace=True)
    #  👀 關注個股:

    txt = f'''
           #### 🎯 篩選條件:
           * 篩選 台股: __{df_all["sid"].nunique()}檔__ 
           * 篩選 股價: __低於 {dic_cfg["sel_price"]}元__
           * 篩選 期間: __{fr} ~ {to}, {dl.days}天__
           * 篩選 策略: 營收, EPS, 殖利率 __任一勝率大於 {dic_cfg["sel_rat"]}% 👍__
           * 篩選 策略: 歷史股價 與 所選策略之 __相關性大於 {dic_cfg["sel_corr"]} 📈__
           '''

    st.title(f'👨‍💻 傑克潘的爬蟲練習')
    fn_st_add_space(1)
    c1, c2 = st.columns([2.5, 1])
    c1.info(txt)

    fn_st_stock_sel(df_all)
    fn_st_add_space(3)
    fn_st_stock_all(df_all)


def fn_st_init():
    st.set_page_config(page_title='爬蟲練習', page_icon='🕷️', layout='wide', initial_sidebar_state="auto", menu_items=None)


def fn_main():
    # if fn_is_parsing():
    #     try:
    #         df = fn_fb_recommend_stock()
    #         fn_find_billion(df, dic_cfg["stocks"])
    #     except:
    #         pass

    fn_st_init()
    fn_st_stock_main()


if __name__ == '__main__':
    fn_main()
