import pandas as pd
import streamlit as st
import math
from collections import defaultdict
from twstock import Stock
from plotly.subplots import make_subplots
from app_stock_fb import *
from app_utils import *

dic_url = {
    'FindBillion': 'https://www.findbillion.com/twstock/',
    'Yahoo': 'https://tw.stock.yahoo.com/quote/',
    'Cmoney': 'https://www.cmoney.tw/forum/stock/',
    'FinLab': r'https://ai.finlab.tw/stock/?stock_id=',
    'WantRich': r'https://wantrich.chinatimes.com/tw-market/listed/stock/',
    'Yahoo_field': r'https://tw.stock.yahoo.com/t/nine.php?cat_id=%',
}

dic_sel = {
    'pick': []
}


dic_field_id = {
    '其他': '23024',
    '水泥': '',
    '食品': '23007',
    '觀光': '23022',
    '塑膠': '23012',
    '汽車': '23019',
    '化學': '23068',
    '光電': '23073',
    '橡膠': '23018',
    '造紙': '23009',
    '鋼鐵': '23017',
    '航運': '23021',
    '半導體': '23071',
    '農科技': '',
    '建材營造': '23006',
    '生技醫療': '23069',
    '紡織纖維': '23008',
    '貿易百貨': '23023',
    '電機機械': '23013',
    '通信網路': '23074',
    '電腦週邊': '23072',
    '電器電纜': '23014',
    '其他電子': '23078',
    '玻璃陶瓷': '23016',
    '電子通路': '',
    '資訊服務': '23077',
    '油電燃氣': '',
    '金融保險': '23010',
    '電子商務': '23076',
    '文化創意': '',
    '電子零組件': '23075',
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


def fn_kpi_plt(kpis, df_sids):
    dis = [k for k in kpis if 'new' in k]
    dis = dis + ['績效(%)', '天數']

    rows = 4
    cols = math.ceil(len(dis) / rows)  # int(round(len(dis) / rows, 0))
    titles = [f'{d} 👉 {round(df_sids[d].min(), 2) if "差" in d or "天數" in d else round(df_sids[d].max(), 2)}' for d in dis]

    dis = dis + ['產業別']
    titles = titles + ['產業別']

    watch = ''
    subplot_titles = []
    for t in titles:
        sub_t = t
        if "勝率" in t:
            if float(t.split('👉')[-1]) > 5.0:
                sub_t = sub_t + '💎'
                watch = watch+'💎'
        if "價差" in t:
            if float(t.split('👉')[-1]) < -5.5:
                sub_t = sub_t + '💎'
                watch = watch+'💎'

        subplot_titles.append(sub_t)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for d in dis:
        i = dis.index(d)
        r = int(i / cols) + 1
        c = i - cols * (r - 1) + 1
        # c = c + 1 if '績效' in d else c
        x = [_ for _ in df_sids[d].values if _ != 0]
        fig.add_trace(
            go.Histogram(x=x, nbinsx=50, showlegend=False,
                         marker=dict(opacity=1, line=dict(color='white', width=0.4)),
                         ),
            row=r, col=c
        )

    margin = {'t': 30, 'b': 0, 'l': 0, 'r': 0}
    fig.update_layout(margin=margin, height=700, width=600)

    return fig, watch


def fn_twstock(sid):
    stock = Stock(sid)  # 擷取台積電股價
    # ma_p = stock.moving_average(stock.price, 5)  # 計算五日均價
    # ma_c = stock.moving_average(stock.capacity, 5)  # 計算五日均量
    # ma_p_cont = stock.continuous(ma_p)  # 計算五日均價持續天數
    # ma_br = stock.ma_bias_ratio(5, 10)  # 計算五日、十日乖離值
    #
    # print(f'{sid} --> {stock.data[-1].date} {stock.price[-1]}元 {int(stock.capacity[-1]/1000)}張')

    price = stock.price[-1]
    amount = int(stock.capacity[-1] / 1000)

    return price, amount


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


def fn_get_field_id(x):
    field_id = ''
    if x in dic_field_id.keys():
        field_id = dic_field_id[x]

    return field_id


def fn_st_stock_sel(df_all):

    df_all['date_dt'] = pd.to_datetime(df_all['date'])
    fr = min(df_all['date'])
    to = max(df_all['date'])
    dl = max(df_all['date_dt']) - min(df_all['date_dt'])
    df_all.drop(columns=['date_dt'], inplace=True)

    fn_st_add_space(1)

    # c1, c2 = st.columns([2.5, 1])
    with st.form(key='sel'):
        st.markdown(f'#### 🎚️ 篩選條件設定:')
        sels = st.columns([1, 1, 2])

        dic_cfg["sel_rat"] = sels[0].slider('勝率門檻(%)', min_value=40, max_value=100, value=50)
        dic_cfg["sel_corr"] = sels[1].slider('相關性門檻', min_value=0.5, max_value=1.0, value=0.8)
        dic_cfg["sel_price"] = sels[2].slider('股價上限', min_value=0, max_value=500, value=200)

        # fn_st_add_space(1)
        submit = st.form_submit_button('選擇')

    txt = f'''
           #### 🎯 篩選條件:
           * 篩選 台股: __{df_all["sid"].nunique()}檔__ 
           * 篩選 股價: __低於 {dic_cfg["sel_price"]}元__
           * 篩選 期間: __{fr} ~ {to}, {dl.days}天__
           * 篩選 策略: 營收, EPS, 殖利率 __任一勝率大於 {dic_cfg["sel_rat"]}% 👍__
           * 篩選 策略: 歷史股價 與 所選策略之 __相關性大於 {dic_cfg["sel_corr"]} 📈__
           '''

    df_sel = fn_stock_sel(df_all)

    if df_sel.shape[0] == 0:
        st.error(f'#### 👉 篩選結果(0檔): 🙅‍♂️')
    else:
        def f(s, name):
            return s if name == '' else name

        df_sel['sid_name'] = df_sel.apply(lambda x: f(x.sid, x.sid_name), axis=1)
        df_sel['sid_name'] = df_sel['sid_name'].apply(lambda x: x.replace('0050', '元大台灣50'))
        df_sel['max'] = df_sel[[c for c in df_sel.columns if '勝率' in c]].max(axis=1)
        df_sel.sort_values(by=['max'], ascending=False, inplace=True, ignore_index=True)

        sel_sid = list(df_sel["sid_name"].unique())
        sel_num = df_sel["sid"].nunique()
        c1, c2 = st.columns([2.5, 1])
        st.info(txt)
        st.error(f'#### 👉 篩選結果({sel_num}檔): {", ".join(sel_sid)}')
        fn_st_add_space(1)

        sel_num_metric = min(sel_num, 8)

        cs = st.columns(sel_num_metric + 1)
        # cs[0].markdown('# 👀')
        cs[0].metric('關注個股', '👀', '績效/天數', delta_color='inverse')
        # j = 1
        profs = []
        metrics = []
        for i in range(sel_num_metric):
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
                sign = '' if prof < 10 else ' 🚀'
                metrics.append([f'⭐{sid_name} {sid}', f'{price_new}{sign}', f'{prof}% / {days}天'])

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

        df_show['field_id'] = df_show['產業別'].apply(fn_get_field_id)
        df_show['產業別'] = df_show.apply(lambda x: fn_click_name(x['field_id'], x['產業別'], dic_url['Yahoo_field']), axis=1)

        show_cols_order = ['股票名稱', '股票代碼', 'date', '股價', '大盤領先指標', '產業領先指標',
                           '勝率(%)_營收', '相關性_營收', '勝率(%)_EPS', '相關性_EPS',
                           '勝率(%)_殖利率', '相關性_殖利率', '產業別', '市場別']

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


def fn_show_bar_h(df, x, y, title=None, barmode='relative', col=None):
    margin = {'t': 40, 'b': 0, 'l': 0, 'r': 0}

    width_full = 1200
    width_max = 600
    height = 650
    bars = 30

    col_max = 3
    col_end = math.ceil(df.shape[0] / bars)
    width = min(int(width_full / col_max), width_max)
    cs = st.columns(col_max)
    fr = 0

    df['min'] = 0
    df['max'] = 0

    for idx in df.index:
        for c in y:
            if df.loc[idx, c] > 0:
                df.at[idx, 'max'] = df.loc[idx, 'max'] + df.loc[idx, c]
            else:
                df.at[idx, 'min'] = df.loc[idx, 'min'] + df.loc[idx, c]

    m, M = df['min'].min(), df['max'].max()
    x_range = [m+min(m/8, -1), M+max(M/8, 1)]

    if col is None:
        for c in range(col_max):
            if c < col_end and fr < df.shape[0]:
                to = min(df.shape[0], fr + bars)
                df_c = df.loc[fr: to].reset_index(drop=True)
                fr = to+1

                fig = fn_gen_plotly_bar(df_c, x_col=y, y_col=x, v_h='h', margin=margin, op=0.9, barmode=barmode,
                                        lg_pos='h', lg_x=0.8, lg_title='指標:', width=width, height=height,
                                        title=title, x_range=x_range)

                cs[col_end - c - 1].plotly_chart(fig, use_container_width=True)

    else:
        st.write(df)
        fig = fn_gen_plotly_bar(df, x_col=y, y_col=x, v_h='h', margin=margin, op=0.9, barmode=barmode,
                                lg_pos='h', lg_x=0.8, lg_title='指標:', width=width, height=height,
                                title=title, x_range=x_range)

        col.plotly_chart(fig, use_container_width=True)


def fn_show_bar(df, x='策略選股', y=None, v_h='h', col=None):
    if v_h == 'v':
        st.bar_chart(data=df, x=x, y=y,
                     width=0, height=500,
                     use_container_width=True)
    else:
        df = df.loc[::-1].reset_index(drop=True)
        fn_show_bar_h(df, x, y, col=col)


def fn_stock_filter(df, stra, col):

    with col.form(key=f'Form2_{stra}'):
        corr = st.slider('相關性 大於', min_value=5.0, max_value=10.0, value=7.0, step=0.5)
        win = st.slider('勝率 大於', min_value=1.0, max_value=10.0, value=4.5, step=0.5)
        margin = st.slider('預估價差 大於', min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        win_diff = st.slider('勝率變化 大於', min_value=0.0, max_value=10.0, value=0.0, step=0.5)
        st.form_submit_button('選擇')

    flts = [f'{stra}_相關性_new', f'{stra}_勝率_new', f'{stra}_合理價差_new', f'{stra}_勝率_diff']

    df_f = df[df[flts[0]].apply(lambda x: x > corr)]
    df_f = df_f[df_f[flts[1]].apply(lambda x: x > win)]
    df_f = df_f[df_f[flts[2]].apply(lambda x: x < -1*margin)]
    df_f = df_f[df_f[flts[3]].apply(lambda x: x > win_diff)]

    return df_f, flts


def fn_st_chart_bar(df):
    df_pick = fn_pick_date(df, '代碼', '日期')
    df_pick['日期'] = pd.to_datetime(df_pick['日期'])

    # st.write(df_pick['股價'])

    df_pick['股價'] = df_pick['股價'].astype(float)

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
        dic_sid['天數'].append(-1 * dt.days)

        for c in df_sid.columns:
            df_sid_old = df_sid[df_sid['日期'] == min(df_sid['日期'])]
            df_sid_new = df_sid[df_sid['日期'] == max(df_sid['日期'])]
            dic_sid[c].append(df_sid.loc[df_sid_old.index[0], c])

            dic_sid[c + '_new'].append(df_sid.loc[df_sid_new.index[0], c])

            if '勝率' in c:
                dic_sid[c + '_diff'].append(df_sid.loc[df_sid_new.index[0], c] - df_sid.loc[df_sid_old.index[0], c])

    df_sids = pd.DataFrame(dic_sid)

    st.markdown(f'#### 📊 {df_sids.shape[0]}檔個股的 績效 v.s. 策略指標')

    # ==========

    for c in [c for c in df_sids.columns if '相關性' in c]:
        df_sids[c] = df_sids[c].apply(lambda x: 0 if x == '' else float(x) * 10)

    for s in ['kpi', 'order', 'order_typ', 'bar']:
        if s not in st.session_state.keys():
            st.session_state[s] = []

    # ==========

    cs = st.columns([3, 1, 1, 1])
    kpis = ['績效(%)', '天數'] + [c for c in df_sids.columns if '勝率' in c or '合理' in c or '相關性' in c]
    with cs[0].form(key='Form1'):
        st.session_state['kpi'] = st.multiselect(f'策略指標:', options=kpis, default=['績效(%)', '營收_勝率_new', '營收_合理價差_new'],
                                                 key='kpixxx')
        fn_st_add_space(1)
        submit = st.form_submit_button('選擇')

    if len(st.session_state['kpi']) > 0:
        st.session_state['order_typ'] = cs[1].selectbox(f'排序方向:', options=['大 --> 小', '小 --> 大'], index=0)
        st.session_state['order'] = cs[1].selectbox(f'排序指標:', options=st.session_state['kpi'], index=0)
        st.session_state['bar'] = cs[2].selectbox(f'柱狀圖方向:', options=['水平', '垂直'], index=0)
        v_h = 'v' if '垂直' in st.session_state['bar'] else 'h'
        st.session_state['kpi'] = [st.session_state['order']]+[k for k in st.session_state['kpi'] if k != st.session_state['order']]

        ascending = st.session_state['order_typ'] == '小 --> 大'
        df_sids.sort_values(by=[st.session_state['order']], inplace=True, ascending=ascending, ignore_index=True)
        df_sids.reset_index(inplace=True)

        def fn_add_digit(x):
            for i in range(3 - len(str(x))):
                x = '0' + str(x)
            return str(x)

        df_sids['index'] = df_sids['index'].apply(fn_add_digit)
        df_sids['策略選股'] = df_sids['index'] + ' ' + df_sids['名稱'] + ' ' + df_sids['代碼']
        df_sids['策略選股'] = df_sids['策略選股'].apply(lambda x: x + '⭐' if x.split(' ')[1] in dic_sel['pick'] else x)
        fn_st_add_space(2)

        df_p = df_sids[df_sids['績效(%)'].apply(lambda x: 1 < x < 5)]
        df_p5 = df_sids[df_sids['績效(%)'].apply(lambda x: x >= 5)]
        df_n = df_sids[df_sids['績效(%)'] < -1]
        df_e = df_sids[df_sids['績效(%)'].apply(lambda x: -1 <= x <= 1)]

        fig, watch = fn_kpi_plt(kpis, df_sids)

        tab_d, tab_p5, tab_p, tab_n, tab_e, tab_f = st.tabs(
            [f'指標分布{watch}', f'正報酬( > 5% ): {df_p5.shape[0]}檔', f'正報酬( 1% ~ 5% ): {df_p.shape[0]}檔',
             f'負報酬( < -1% ): {df_n.shape[0]}檔', f'持平( -1% ~ 1% ): {df_e.shape[0]}檔', '策略選股'])

        with tab_p:
            fn_show_bar(df_p, y=st.session_state['kpi'], v_h=v_h)

        with tab_p5:
            fn_show_bar(df_p5, y=st.session_state['kpi'], v_h=v_h)

        with tab_n:
            fn_show_bar(df_n, y=st.session_state['kpi'], v_h=v_h)

        with tab_e:
            fn_show_bar(df_e, y=st.session_state['kpi'], v_h=v_h)

        with tab_d:
            cs = st.columns([1, 7, 1])
            cs[1].plotly_chart(fig, use_container_width=True)

        with tab_f:
            tab1, tab2, tab3 = st.tabs(['依營收', '依EPS', '依殖利率'])
            with tab1:
                cols = st.columns([1, 2, 1])
                df_f, flts = fn_stock_filter(df_sids, '營收', cols[0])
                fn_show_bar(df_f, y=flts, v_h=v_h, col=cols[1])

            with tab2:
                cols = st.columns([1, 2, 1])
                df_f, flts = fn_stock_filter(df_sids, 'EPS', cols[0])
                fn_show_bar(df_f, y=flts, v_h=v_h, col=cols[1])

            with tab3:
                cols = st.columns([1, 2, 1])
                df_f, flts = fn_stock_filter(df_sids, '殖利率', cols[0])
                fn_show_bar(df_f, y=flts, v_h=v_h, col=cols[1])


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

    cols = [c for c in df_all.columns if '策略_' not in c]
    df_all = df_all[cols]

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
    df_field = pd.read_csv('stock_field.csv', na_filter=False, encoding='utf_8_sig', index_col=0, dtype=str)
    df_all["篩選"] = 0

    for idx in df_all.index:
        sid = df_all.loc[idx, 'sid']
        df_all.at[idx, '產業別'] = '未分類'
        df_all.at[idx, '市場別'] = '未分類'
        if sid in df_field['sid'].values:
            field = df_field[df_field['sid'] == sid]['產業別'].values[0]
            market = df_field[df_field['sid'] == sid]['市場別'].values[0]
            df_all.at[idx, '產業別'] = field
            df_all.at[idx, '市場別'] = market

    st.title(f'👨‍💻 傑克潘的爬蟲練習')
    fn_st_stock_sel(df_all)
    fn_st_add_space(3)
    fn_st_stock_all(df_all)


def fn_st_init():
    st.set_page_config(page_title='爬蟲練習', page_icon='🕷️', layout='wide', initial_sidebar_state="auto", menu_items=None)


def fn_main():
    fn_st_init()
    fn_st_stock_main()


if __name__ == '__main__':
    fn_main()
