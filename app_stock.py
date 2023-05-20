import datetime
import random

import pandas as pd
import streamlit as st
import math
from collections import defaultdict
from pandas_datareader import data
import yfinance as yf
from twstock import Stock
from plotly.subplots import make_subplots
from app_stock_fb import *
from app_utils import *
from workalendar.asia import Taiwan
# from streamlit_player import st_player
from platform import python_version
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


dic_sel = {
    'pick': []
}

dic_my_stock = {'my_stock': ['2851 中再保', '4562 穎漢', '3426 台興', '2404 漢唐']}

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

dic_mops = {'per_date': '0519'}

dic_fin_name = {
    'ROE': '權益報酬率',
    'ROA': '資產報酬率',
    'OPM': '營業利益率',
    'DR': '負債佔資產比率',
    'OCF': '營業現金對負債比',
}

dic_mkd = {
    '1sp': "&nbsp;",
    '2sp': "&ensp;",
    '4sp': "&emsp;",
}

dic_book_img = {
    '我的職業是股東': r'https://im2.book.com.tw/image/getImage?i=https://www.books.com.tw/img/001/080/05/0010800551.jpg&v=5baa0e38k&w=348&h=348',
    '大會計師教你從財報數字看懂經營本質': r'https://im1.book.com.tw/image/getImage?i=https://www.books.com.tw/img/001/082/53/0010825332.jpg&v=5d038542k&w=348&h=348',
    '掌握市場週期': r'https://im2.book.com.tw/image/getImage?i=https://www.books.com.tw/img/001/081/02/0010810203.jpg&v=5c235887k&w=348&h=348',
    '投資最重要的事': r'https://im2.book.com.tw/image/getImage?i=https://www.books.com.tw/img/001/074/49/0010744933.jpg&v=58a6d0d2k&w=348&h=348',
}

dic_book_lnk = {
    '我的職業是股東': r'https://www.books.com.tw/products/0010800551?utm_source=chiay0327&utm_medium=ap-books&utm_content=recommend&utm_campaign=ap-201809',
    '大會計師教你從財報數字看懂經營本質': r'https://www.books.com.tw/products/0010825332?utm_source=chiay0327&utm_medium=ap-books&utm_content=recommend&utm_campaign=ap-202205',
    '掌握市場週期': r'https://www.books.com.tw/products/0010810203?sloc=main',
    '投資最重要的事': r'https://www.books.com.tw/products/0010744933?sloc=main',
}

dic_book_cmt = {
    '我的職業是股東': '''這本書概念性的介紹各種投資理論，並比較各門派投資方法的優缺點，  
                       沒有論及太多技術細節，讀起來相對輕鬆，適合當投資小白的入門書。''',

    '大會計師教你從財報數字看懂經營本質': '''這本書介紹三大財務報表(資產負債表、損益表、現金流量表)的各項指標，  
                                       以及如何由這些指標判斷公司的經營體質。  
                                       個人覺得若非財會背景還是不容易消化，需反芻多次，方可內化成自身武功。''',

    '掌握市場週期': '''順勢而為''',
    '投資最重要的事': '''巴菲特，讀兩遍''',
}

dic_colors = {
    "c1": "rgba(35, 146, 255, 1)",
    "c2": "rgba(0, 72, 142, 1)",
}

dic_df = {}


def fn_make_clickable(x):
    name = x
    sid = x if str(x).isnumeric() else x.split(" ")[0]
    url = rf'{dic_url["WantRich"]}{sid}'

    return '<a href="{}">{}</a>'.format(url, name)


def fn_make_clickable_report(sid):
    if sid == '':
        return sid
    else:
        df_report = dic_df['report']
        df_rp_sid = df_report[df_report['sid'] == sid]
        url = df_rp_sid['report'].values[0]
        name = url.split('M')[0].split(sid)[-1]
        if name == 'NA':
            url = dic_url['Cnyes'] + f'{sid}/company/profile'

        return '<a href="{}">{}</a>'.format(url, name)


def fn_make_clickable_tdcc(x):
    name = x
    url = rf'{dic_url["tdcc"]}'

    if x == '':
        return x
    else:
        return '<a href="{}">{}</a>'.format(url, name)


def fn_click_name(sid, name, url):
    url = rf'{url}{sid}'

    return '<a href="{}">{}</a>'.format(url, name)


def fn_color_map(x):
    css = ""  # 'background-color: white; color: black'
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
    titles = [f'{d} 👉 {round(df_sids[d].min(), 2) if "差" in d or "天數" in d else round(df_sids[d].max(), 2)}' for d in
              dis]

    dis = dis + ['產業別']
    titles = titles + ['產業別']

    watch = ''
    subplot_titles = []
    for t in titles:
        sub_t = t
        if "勝率" in t:
            if float(t.split('👉')[-1]) > 5.0:
                sub_t = sub_t + '💎'
                watch = watch + '💎'
        if "價差" in t:
            if float(t.split('👉')[-1]) < -5.5:
                sub_t = sub_t + '💎'
                watch = watch + '💎'

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
        lead = df_all.loc[idx, '產業領先指標']
        market = df_all.loc[idx, '市場別']
        for c in df_all.columns:
            if '勝率' in c:
                v = df_all.loc[idx, c]
                corr = df_all.loc[idx, '相關性_' + c.split('_')[-1]].split(' ')[-1]
                if v != '' and corr != '':
                    if int(v) >= dic_cfg["sel_rat"] and float(corr) > dic_cfg["sel_corr"]:
                        if dic_cfg["sel_lead"] == '極佳':
                            if lead == '極佳':
                                if '櫃' in dic_cfg['sel_market'] or market == '上市':
                                    df_all.at[idx, "篩選"] = 1
                                    break
                        elif dic_cfg["sel_lead"] == '佳':
                            if lead == '佳' or lead == '極佳':
                                if '櫃' in dic_cfg['sel_market'] or market == '上市':
                                    df_all.at[idx, "篩選"] = 1
                                    break
                        elif dic_cfg["sel_lead"] == '中等':
                            if lead == '佳' or lead == '極佳' or lead == '中等':
                                if '櫃' in dic_cfg['sel_market'] or market == '上市':
                                    df_all.at[idx, "篩選"] = 1
                                    break
                        else:
                            df_all.at[idx, "篩選"] = 1
                            break
                    elif int(v) >= dic_cfg["sel_rat_h"]:
                        # df_all.at[idx, "篩選"] = 1
                        pass
                        break

    # for s in df_all[df_all['篩選'] == 1]['sid'].unique():
    sids = df_all[df_all['篩選'] == 1]['sid'].unique().tolist()
    # st.write(sids)
    sids = [s1.split(" ")[0] for s1 in dic_my_stock['my_stock']] + sids
    # st.write(sids)
    for s in sids:
        df_sid = df_all[df_all['sid'] == s]
        s_date = df_sid[df_sid['篩選'] == 1]['date'].min()
        s_date = df_sid['date'].min() if str(s_date) == 'nan' else s_date
        for idx in df_all.index:
            if df_all.loc[idx, 'sid'] == s and df_all.loc[idx, 'date'] > s_date:
                df_all.at[idx, "篩選"] = 1

    df_sel = df_all[df_all["篩選"] == 1]
    # df_sel = df_sel[df_sel["股價"].apply(lambda x: float(x) < dic_cfg["sel_price"] if x != '' else True)]
    # df_sel = df_sel[[c for c in df_sel.columns if '篩選' not in c and
    #                  '耗時' not in c and
    #                  '合理價差' not in c]]

    df_sel = df_sel[[c for c in df_sel.columns if '篩選' not in c and
                     '耗時' not in c]]

    df_sel.reset_index(drop=True, inplace=True)
    # st.write(df_all[df_all['sid']=='3023'])
    # st.write(df_sel)
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


def fn_get_stock_price(sid, days=30):
    sid_tw = sid + '.TW'
    df_sid = pd.DataFrame()

    end = datetime.datetime.today().date() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=days)
    yf.pdr_override()
    try:
        df_sid = data.get_data_yahoo([sid_tw], start, end)
    except:
        df_sid = yf.download([sid_tw], start, end)

    if df_sid.shape[0] == 0:
        df_sid = data.get_data_yahoo([sid_tw + 'O'], start, end)

    return df_sid


def fn_get_stock_price_plt(df, df_p=None, days_ago=None, watch=None, height=120, showlegend=False, title=None, op=0.5):
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(go.Bar(x=df.index,
                         y=df['Volume'].apply(lambda x: int(x / 1000)),
                         name='交易量',
                         opacity=0.4,
                         ),
                  secondary_y=False)

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='漲跌價',
                                 increasing={'line_color': 'red'},
                                 decreasing={'line_color': 'green'}),
                  secondary_y=True)

    if df_p is None:
        pass
    else:
        df1 = df_p[[c for c in df_p.columns if 'date' in c or '合理價_' in c or 'sid' in c or '股價' in c or '相關' in c]]

        for c in df1.columns:
            if '合理價_' in c:
                df_plt = df1[df1[c].apply(lambda x: len(str(x)) > 0)]

                try:
                    corr = float(df_plt['相關性_'+c.split('_')[-1]].values[-1].split('相關')[-1])
                    visible = 'legendonly' if corr < 0.65 else None
                    op = op if corr < 0.65 else 0.8
                except:
                    visible = 'legendonly'

                fig.add_trace(go.Scatter(x=df_plt['date'], y=df_plt[c],
                                         mode='lines+markers', name=c,
                                         opacity=op, visible=visible),
                              secondary_y=True)

    if title is None:
        margin = {'t': 0, 'b': 0, 'l': 10, 'r': 10}
        title_dic = dict(text='', font_size=22, font_family='Times New Roman')
    else:
        margin = {'t': 50, 'b': 0, 'l': 10, 'r': 10}
        title_dic = dict(text=title, font_size=22, font_family='Times New Roman')

    fig.update_layout(xaxis_rangeslider_visible=False, margin=margin, height=height, showlegend=showlegend,
                      title=title_dic)

    fig.update_xaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikemode="across", spikethickness=1,
                     spikedash='solid', rangebreaks=[dict(bounds=["sat", "mon"])])

    fig.update_yaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikemode="across", spikethickness=1,
                     spikedash='solid')

    # if days_ago is not None:
    #     days_ago = days_ago - int(days_ago / 7) * 2 - 1
    #     color = "pink" if df["Close"][-1] >= df["Close"][days_ago] else "lightgreen"
    #     fig.add_vrect(x0=df.index[days_ago], x1=df.index[-1],
    #                   fillcolor=color, opacity=0.45, line_width=0)

    if watch is not None:
        fr, to = watch[0], watch[-1]

        day_fr = datetime.date.fromisoformat(fr)

        is_working_day = Taiwan().is_working_day(day_fr)
        # st.write(f'{fr} --> {is_working_day}')

        # day_to = datetime.date.fromisoformat(to)
        # to = str(day_to + datetime.timedelta(days=-1))

        if is_working_day:
            pass
        else:
            fr = str(day_fr + datetime.timedelta(days=-2))
            # st.write(f'{fr} --> {fr in df.index}')

        if fr in df.index:
            to = to if to in df.index else str(df.index.values[-1])
            p_fr = df[df.index == fr]["Close"].values[0]
            p_to = df[df.index == to]["Close"].values[0]
            color = "pink" if p_to >= p_fr else "lightgreen"
            fig.add_vrect(x0=fr, x1=to,
                          fillcolor=color, opacity=0.2, line_width=0)

        else:
            st.write(f'{fr} --> {fr in df.index}')
            st.write(f'{to} --> {to in df.index}')

    return fig


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
        fn_st_add_space(1)
        sels = st.columns([1, 1, 1, 0.1, 0.45, 0.45, 0.45])

        sid_2_watch = sels[0].text_input('手動篩選:', value='3093, 4562, 3426, 5871, 2330', key='sid_2_watch')
        dic_cfg["sel_rat"] = sels[1].slider('勝率門檻(%)', min_value=40, max_value=100, value=50)
        dic_cfg["sel_corr"] = sels[2].slider('相關性門檻', min_value=0.5, max_value=1.0, value=0.9)
        # dic_cfg["sel_price"] = sels[2].slider('股價上限', min_value=0, max_value=500, value=500)
        dic_cfg["sel_lead"] = sels[4].radio('產業領先指標', ('極佳', '極佳/佳'), index=0, horizontal=False)
        dic_cfg["sel_market"] = sels[5].radio('市場別', ('上市', '上市/櫃'), index=1, horizontal=False)
        is_latest_only = sels[6].radio('表格顯示', ('最新', '歷史'), index=0, horizontal=False)
        show_latest_only = True if is_latest_only == '最新' else False

        dic_my_stock['my_stock'] = list(sid_2_watch.replace(' ', '').split(','))

        fn_st_add_space(1)
        submit = st.form_submit_button('選擇')

    txt = f'''
           ##### 🎯 篩選條件:
           * 篩選 台股: __{df_all["sid"].nunique()}檔__ 
           * 篩選 股價: __低於 {dic_cfg["sel_price"]}元__
           * 篩選 期間: __{fr} ~ {to}, {dl.days}天__
           * 篩選 策略: 營收, EPS, 殖利率 __任一勝率大於 {dic_cfg["sel_rat"]}% 👍__
           * 篩選 策略: 歷史股價 與 所選策略之 __相關性大於 {dic_cfg["sel_corr"]} 📈__
           '''

    df_sel = fn_stock_sel(df_all)

    # st.write(df_sel)

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
        # c1, c2 = st.columns([2.5, 1])
        cols = st.columns([1, 1])
        cols[0].info(txt)
        cols[1].error(f'##### 👉 篩選結果(:red[{sel_num}檔]): :blue[{", ".join(sel_sid)}]')
        fn_st_add_space(1)

        df_sel['sid_and_name'] = df_sel['sid'] + ' ' + df_sel['sid_name']
        with st.form(key='watch'):
            st.markdown(f'#### 👀 選擇關注個股:')
            option_all = df_sel['sid_and_name'].unique().tolist()
            option_dft = option_all[0: 1 + min(len(option_all) - 1, 7)]
            cols = st.columns([6, 0.5, 1])
            option_sel = cols[0].multiselect('', option_all, option_dft, key='watch_sids', label_visibility='collapsed')
            cols[2].form_submit_button('選擇')
            fn_st_add_space(1)

        fn_st_add_space(1)

        watchs = [s in option_sel for s in df_sel['sid_and_name'].values]

        df_sel = df_sel[watchs]
        df_sel.reset_index(inplace=True, drop=True)

        sel_sid = list(df_sel["sid_name"].unique())

        # st.write(df_sel)

        sel_num_metric = sel_num  # min(sel_num, 8)
        # cs = st.columns(sel_num_metric + 1)
        metric_cols = 8
        cs = st.columns(metric_cols)
        # cs[0].markdown('# 👀')
        cs[0].metric('關注個股', '👀', '績效/天數', delta_color='inverse')
        # j = 1
        profs = []
        metrics = []
        # for i in range(sel_num_metric):
        for i in range(len(sel_sid)):
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
                sign = '' if prof < 20 else ' 🚀'
                metrics.append([f'⭐{sid_name} {sid}', f'{price_new}{sign}', f'{prof}% / {days}天'])

                # cs[j].metric(f'{sid_name} {sid}', f'{price_new}', f'{prof}% / {days}天', delta_color='inverse')
                # j = j + 1

        profs_sort = sorted(profs, reverse=True)

        # j = 0
        # for p in profs:
        #     i = profs_sort.index(p)
        #     cs[i + 1].metric(*metrics[j], delta_color='inverse')
        #     j += 1

        for p in profs_sort:
            i = profs_sort.index(p)
            if i < metric_cols - 1:
                cs[i + 1].metric(*metrics[profs.index(p)], delta_color='inverse')

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
                return str(x) + '%'

        watch_list = []
        for idx in df_show.index:
            df_tdcc = dic_df['tdcc']
            # date_tdcc = df_tdcc['資料日期'].values[0]
            # date_show = df_show.loc[idx, 'date'].replace('-', '')
            idx_sid = df_show.loc[idx, 'sid']

            df_tdcc_sid = df_tdcc[df_tdcc['證券代號'] == idx_sid]

            # if int(date_show) >= int(date_tdcc) and idx_sid not in watch_list:
            if idx_sid in watch_list:
                big = ''
                num = ''
                rp = ''

            else:
                big = df_tdcc_sid[df_tdcc_sid['持股分級'] == '15']['占集保庫存數比例%'].values[0] + '%'
                num = int(df_tdcc_sid[df_tdcc_sid['持股分級'] == '17']['人數'].values[0])
                num = str(num) + '人' if num < 10000 else str(round(num / 10000, 1)) + '萬人'
                rp = idx_sid

                watch_list.append(idx_sid)

            df_show.loc[idx, '大戶比'] = big
            df_show.loc[idx, '股東數'] = num
            df_show.loc[idx, '法說會'] = rp

        df_show['大戶比'] = df_show['大戶比'].apply(fn_make_clickable_tdcc)
        df_show['股東數'] = df_show['股東數'].apply(fn_make_clickable_tdcc)
        df_show['法說會'] = df_show['法說會'].apply(fn_make_clickable_report)

        for c in df_show.columns:
            if '勝率' in c:
                df_show[c] = df_show[c].apply(fn_sel)
                page = dic_page[c.split('_')[-1]]
                df_show[c] = df_show.apply(lambda x: fn_click_name(x['sid'] + page, x[c], dic_url['FindBillion']),
                                           axis=1)
            if '相關性' in c:
                df_show[c] = df_show[c].apply(lambda x: x.split(' ')[-1])

        df_show['股票代碼'] = df_show['sid'].apply(fn_make_clickable)
        df_show['股票名稱'] = df_show.apply(lambda x: fn_click_name(x["sid"], x["sid_name"], dic_url['dog']), axis=1)
        df_show['股價'] = df_show.apply(
            lambda x: fn_click_name(x["sid"] + '/technical-analysis', x["股價"], dic_url['Yahoo']), axis=1)

        df_show['field_id'] = df_show['產業別'].apply(fn_get_field_id)
        df_show['產業別'] = df_show.apply(lambda x: fn_click_name(x['field_id'], x['產業別'], dic_url['Yahoo_field']),
                                          axis=1)
        df_show['勝率(%)_營收'] = df_show['勝率(%)_營收'] + ' , ' + df_show['合理價差(%)_營收'] + '%' + ' , ' + df_show[
            '相關性_營收']
        df_show['勝率(%)_EPS'] = df_show['勝率(%)_EPS'] + ' , ' + df_show['合理價差(%)_EPS'] + '%' + ' , ' + df_show[
            '相關性_EPS']
        df_show['勝率(%)_殖利率'] = df_show['勝率(%)_殖利率'] + ' , ' + df_show['合理價差(%)_殖利率'] + '%' + ' , ' + \
                                    df_show['相關性_殖利率']
        df_show['領先指標'] = df_show['大盤領先指標'] + ' , ' + df_show['產業領先指標']
        df_show['領先指標'] = df_show['領先指標'].apply(
            lambda x: x.replace('佳 ,', '佳等 ,') if str(x).startswith('佳 ,') else x)

        show_cols_order = ['股票名稱', '股票代碼', 'date', '股價',
                           '勝率(%)_營收', '勝率(%)_EPS', '勝率(%)_殖利率',
                           '領先指標', '產業別', '市場別', '大戶比', '股東數', '法說會']

        df_show = df_show[[c for c in show_cols_order if c in df_show.columns]]

        show_cols_rename = {'date': '日期',
                            '股票名稱': '名稱',
                            '股票代碼': '代碼',
                            '領先指標': '領先指標<br>大盤, 產業',
                            '勝率(%)_營收': '營收<br>勝率, 價差, 相關',
                            '相關性_營收': '營收<br>相關性',
                            '勝率(%)_EPS': 'EPS<br>勝率, 價差, 相關',
                            '相關性_EPS': 'EPS<br>相關性',
                            '勝率(%)_殖利率': '殖利率<br>勝率, 價差, 相關',
                            '相關性_殖利率': '殖利率<br>相關性'}

        df_show.rename(columns=show_cols_rename, inplace=True)

        # show_latest_only = True
        if show_latest_only:
            df_show = df_show[df_show['大戶比'] != '']
            df_show.reset_index(drop=True, inplace=True)

        fn_st_add_space(1)
        st.write(df_show.to_html(escape=False, index=True), unsafe_allow_html=True)

        fn_st_add_space(3)

        p, s, d, sid_order, days = [], [], [], [], []

        for m in metrics:
            p.append(float(m[2].split('%')[0]))
            s.append(m[0])
            d.append(int(m[-1].split(' /')[-1].replace('天', '')))

        p_sort = sorted(p, reverse=True)

        for ps in p_sort:
            sid = s[p.index(ps)]
            if sid not in sid_order:
                sid_order.append(sid)
                days.append(d[p.index(ps)])

        is_price_got = False
        df_per = dic_mops['per']
        # st.write(df_per)
        for n_s in sid_order:
            sid = n_s.split(' ')[-1]
            df_per_sid = df_per[df_per['股票代號'] == sid]
            if df_per_sid.shape[0] > 0:
                per = df_per_sid['本益比'].values[-1]
                p2 = df_per_sid['殖利率(%)'].values[-1]
                mk = df_per_sid['市場別'].values[-1]
            else:
                per, p2, mk = 'NA', 'NA', 'NA'

            df = fn_get_stock_price(sid, days=300)
            if df.shape[0] > 0:
                is_price_got = True
                days_ago = -1 * days[sid_order.index(n_s)]
                fr = df_sel[df_sel['sid'] == sid]['date'].min()
                to = df_sel[df_sel['sid'] == sid]['date'].max()
                df_p = df_all[df_all['sid'] == sid]
                fig = fn_get_stock_price_plt(df, df_p=df_p, days_ago=days_ago, watch=[fr, to], height=150)
                # st.write(f'{sid} {fr} {to}')

                fig.update_layout(
                    yaxis={'showticklabels': True,
                           'showgrid': False,
                           'showspikes': True,
                           'spikethickness': 1,
                           'spikecolor': "grey",
                           'spikedash': 'solid',
                           'spikemode': "across",
                           'spikesnap': "cursor",
                           'tickfont_color': 'dodgerblue',
                           },
                    yaxis2={'showticklabels': True,
                            'showgrid': True,
                            'tickfont_color': 'red',
                            },
                )

                c1, c2, c3, c4 = st.columns([1.3, 5, 1, 1])
                n = n_s.split(' ')[0].replace("⭐", "").replace('-', '')
                s = n_s.split(' ')[-1].replace("0050", "")
                if mk == 'NA':
                    c1.markdown(f'##### [${n}\ {s}$]({dic_url["dog"] + s})')
                else:
                    c1.markdown(f'##### [${n}\ {s}$]({dic_url["dog"] + s})$\ ({mk})$')
                lnk1 = r'https://www.twse.com.tw/zh/page/trading/exchange/BWIBBU.html'
                lnk2 = r'https://www.tpex.org.tw/web/stock/aftertrading/peratio_stk/pera.php?l=zh-tw'
                link = lnk1 if mk == '市' else lnk2
                c1.markdown(f'[$本益比:\ {per}$]({link})')
                # c1.markdown(f'[$殖利率:\ {p2}\ \%$]({link})')
                try:
                    color = 'red' if float(p2) >= 5 else 'green'
                except:
                    color = 'blue'
                c1.markdown(f'[:{color}[$殖利率:\ {p2}\ \%$]]({link})')
                c2.plotly_chart(fig, use_container_width=True)

                for m in metrics:
                    if sid in m[0]:
                        c3.metric(*metrics[metrics.index(m)], delta_color='inverse')

        if is_price_got is False:
            st.error(f'get stock price fail !')


def fn_show_bar_h(df, x, y, title=None, barmode='relative', col=None, lg_pos='h', margin=None, showtick_y=True,
                  text=None, tick_color=None, height=500):
    margin = {'t': 40, 'b': 0, 'l': 0, 'r': 0} if margin is None else margin

    width_full = 1000
    width_max = 600
    col_max = 3

    bars = math.ceil(df.shape[0] / col_max)
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
    x_range = [m + min(m / 8, -1), M + max(M / 8, 1)]

    if col is None:
        for c in range(col_max):
            if c < col_end and fr < df.shape[0]:
                to = min(df.shape[0], fr + bars)
                df_c = df.loc[fr: to].reset_index(drop=True)
                fr = to + 1

                fig = fn_gen_plotly_bar(df_c, x_col=y, y_col=x, v_h='h', margin=margin, op=0.9, barmode=barmode,
                                        lg_pos=lg_pos, lg_x=0.8, lg_title='', width=width, height=height,
                                        title=title, x_range=x_range, showtick_y=showtick_y, txt_col=text)

                fig.update_xaxes(tickfont_size=16)
                fig.update_yaxes(tickfont_size=16, tickfont_color=tick_color)

                if col_end - c - 1 < col_max:
                    cs[col_end - c - 1].plotly_chart(fig, use_container_width=True)
                else:
                    cs[col_max - 1].error(f'{col_end} - {c} - 1 out of max col {col_max}')

    else:
        fig = fn_gen_plotly_bar(df, x_col=y, y_col=x, v_h='h', margin=margin, op=0.9, barmode=barmode,
                                lg_pos=lg_pos, lg_x=0.8, lg_title='', lg_top=False, width=width, height=height,
                                title=title, x_range=x_range, showtick_y=showtick_y, txt_col=text)

        fig.update_xaxes(tickfont_size=16)
        fig.update_yaxes(tickfont_size=16)
        col.plotly_chart(fig, use_container_width=True)


def fn_show_bar(df, x='策略選股', y=None, text=None, v_h='h', col=None, lg_pos='h', margin=None, showtick_y=True, tick_color=None, height=500):
    if v_h == 'v':
        if col is None:
            st.bar_chart(data=df, x=x, y=y,
                         width=0, height=500,
                         use_container_width=True)
        else:
            col.write('')
            col.bar_chart(data=df, x=x, y=y,
                          width=0, height=500,
                          use_container_width=True)
    else:
        df = df.loc[::-1].reset_index(drop=True)
        fn_show_bar_h(df, x, y, col=col, lg_pos=lg_pos, margin=margin, showtick_y=showtick_y, text=text, tick_color=tick_color, height=height)


def fn_stock_filter(df, stra, col, fr=''):
    for _ in range(1):
        col.write('')
    with col.form(key=f'Form2_{stra}_{fr}'):
        dft_win = round(float(df[f'{stra}_勝率_new'].max() - 0.5), 1)
        win = st.slider(f'{stra} 勝率 大於', min_value=1.0, max_value=10.0, value=dft_win, step=0.5)
        v = 2.0 if '營收' in stra else -1.0
        margin = st.slider(f'{stra} 預估價差 大於', min_value=-1.0, max_value=10.0, value=v, step=0.5)
        corr = st.slider(f'{stra} 相關性 大於', min_value=5.0, max_value=10.0, value=6.0, step=0.5)
        win_diff = st.slider(f'{stra} 勝率變化 大於', min_value=-1.0, max_value=10.0, value=-1.0, step=0.5)
        fn_st_add_space(3)
        st.form_submit_button('選擇')

    flts = [f'{stra}_勝率_new', f'{stra}_合理價差_new', f'{stra}_相關性_new', f'{stra}_勝率_diff']

    df_f = df[df[flts[0]].apply(lambda x: x > win)]
    df_f = df_f[df_f[flts[1]].apply(lambda x: x < -1 * margin)] if df_f.shape[0] > 0 else df_f
    df_f = df_f[df_f[flts[2]].apply(lambda x: x > corr)] if df_f.shape[0] > 0 else df_f
    df_f = df_f[df_f[flts[3]].apply(lambda x: x > win_diff)] if df_f.shape[0] > 0 else df_f

    df_f.sort_values(by=flts, ascending=[False, True, False, False], inplace=True, ignore_index=True)

    return df_f, flts


def fn_basic_rule(sid, df_mops, years=5):
    chk_fr = int(df_mops['year'].values[-1]) - years
    df_mops = df_mops[df_mops['year'].apply(lambda x: int(x) > chk_fr)]

    df_sm = df_mops[df_mops['公司代號'] == sid]
    ROE = [float(r) for r in df_sm['獲利能力-權益報酬率(%)'].values]
    basic = '⭕'
    # if len(ROE) > 1:
    #     basic = '❌' if ROE[-1] < ROE[-2] else '⭕'
    basic = '❌' if min(ROE) < 8 else basic
    basic = '✔️' if basic == '⭕' and ROE[-1] > 15 else basic

    return basic


def fn_stock_basic(df, df_mops, y, col):
    txt = f'''
           ##### 🎯 [$基本面指標$](https://youtu.be/ShNI41_rFv4?list=PLySGbWJPNLA8D17qZx0KVkJaXd3qxncGr&t=69):✔️ ⭕  ❌ 
           1. ROE: __> 8%__ (公司錢滾錢的能力)
           2. 營業利益率: __> 0%__ (本業有沒有賺錢)
           3. 本業收入率: __> 80%__ (本業收入的比例)
           4. 負債佔資產比率: __< 60%__ (舉債經營壓力)
           5. 營運現金流量: __> 0__ (確認有現金流入)
           '''

    col.info(txt)

    for idx in df.index:
        sid = df.loc[idx, '代碼']
        basic = fn_basic_rule(sid, df_mops)

        df.at[idx, 'basic'] = f'基本面: {basic}'

    return df, y


def fn_get_mops(df_mops, sid):
    # 公司代號, 公司簡稱, year, market,
    #
    # 財務結構-負債佔資產比率(%),
    # 財務結構-長期資金佔不動產、廠房及設備比率(%),
    #
    # 償債能力-流動比率(%),
    # 償債能力-速動比率(%),
    # 償債能力-利息保障倍數(%),
    #
    # 經營能力-應收款項週轉率(次),
    # 經營能力-平均收現日數,
    # 經營能力-存貨週轉率(次),
    # 經營能力-平均售貨日數,
    # 經營能力-不動產、廠房及設備週轉率(次),
    # 經營能力-總資產週轉率(次),
    #
    # 獲利能力-資產報酬率(%),
    # 獲利能力-權益報酬率(%),
    # 獲利能力-稅前純益佔實收資本比率(%),
    # 獲利能力-純益率(%),獲利能力-每股盈餘(元),
    #
    # 現金流量-現金流量比率(%),
    # 現金流量-現金流量允當比率(%),
    # 現金流量-現金再投<br>資比率(%),

    df_mops_sid = df_mops[df_mops['公司代號'] == str(sid)].reset_index(drop=True)
    df_mops_sid = df_mops_sid[['公司代號', '公司簡稱', 'market', 'year',
                               '獲利能力-資產報酬率(%)', '獲利能力-權益報酬率(%)',
                               '財務結構-負債佔資產比率(%)', '現金流量-現金流量比率(%)']]

    return df_mops_sid


def fn_show_mops(df_mops, df):
    for sid in df['代碼'].values:
        df_mops_sid = fn_get_mops(df_mops, sid)
        st.write(df_mops_sid)


def fn_add_digit(x):
    for i in range(3 - len(str(x))):
        x = '0' + str(x)
    return str(x)


def fn_get_sids(df):
    df_pick = fn_pick_date(df, '代碼', '日期')
    df_pick['日期'] = pd.to_datetime(df_pick['日期'])
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

    # ==========

    for c in [c for c in df_sids.columns if '相關性' in c]:
        df_sids[c] = df_sids[c].apply(lambda x: 0 if x == '' else float(x) * 10)

    return df_sids


def fn_pick_stock(df, df_mops):
    df_sids = fn_get_sids(df)

    # df_sids['index'] = df_sids['index'].apply(fn_add_digit)
    # df_sids['策略選股'] = df_sids['index'] + ' ' + df_sids['名稱'] + ' ' + df_sids['代碼']
    df_sids['策略選股'] = df_sids['名稱'] + ' ' + df_sids['代碼']
    df_sids['策略選股'] = df_sids['策略選股'].apply(lambda x: x + '⭐' if x.split(' ')[1] in dic_sel['pick'] else x)

    # df_mops = pd.read_csv('mops.csv', na_filter=False, dtype=str)

    tab1, tab2, tab3 = st.tabs(['依營收', '依EPS', '依殖利率'])
    margin = {'t': 15, 'b': 110, 'l': 0, 'r': 0}
    col_width = [0.8, 1.6, 0.8]
    with tab1:
        cols = st.columns(col_width)
        df, y = fn_stock_filter(df_sids, '營收', cols[0], fr='pick')
        if df.shape[0] > 0:
            cols[2].write('')
            df, y = fn_stock_basic(df.copy(), df_mops, y.copy(), cols[2])
            fn_show_bar(df, y=y, text='basic', col=cols[1], margin=margin, height=650)
            # fn_show_hist_price(df, df_mops, key='income')
        else:
            cols[1].write('')
            cols[1].markdown('# 🙅‍♂️')

    with tab2:
        cols = st.columns(col_width)
        df, y = fn_stock_filter(df_sids, 'EPS', cols[0], fr='pick')
        if df.shape[0] > 0:
            cols[2].write('')
            df, y = fn_stock_basic(df.copy(), df_mops, y.copy(), cols[2])
            fn_show_bar(df, y=y, text='basic', col=cols[1], margin=margin, height=650)
            # fn_show_hist_price(df, df_mops, key='eps')
        else:
            cols[1].write('')
            cols[1].markdown('# 🙅‍♂️')

    with tab3:
        cols = st.columns(col_width)
        df, y = fn_stock_filter(df_sids, '殖利率', cols[0], fr='pick')
        if df.shape[0] > 0:
            cols[2].write('')
            df, y = fn_stock_basic(df.copy(), df_mops, y.copy(), cols[2])
            fn_show_bar(df, y=y, text='basic', col=cols[1], margin=margin, height=650)
            # fn_show_hist_price(df, df_mops, key='cash')
        else:
            cols[1].write('')
            cols[1].markdown('# 🙅‍♂️')


def fn_get_mops_fin(fin, sid, years=None):
    df_fin = dic_mops[fin]

    df_mops_fin = df_fin[df_fin['sid'] == sid]

    assert df_mops_fin.shape[0] > 0, f'{sid} not in df_fin {fin}'

    df_mops_fin = df_mops_fin[[c for c in df_mops_fin.columns if 'Q' in c]]
    df_mops_fin = df_mops_fin.transpose()
    df_mops_fin.rename(columns={df_mops_fin.columns[0]: f'{dic_fin_name[fin]}({fin})'}, inplace=True)
    df_mops_fin['year'] = df_mops_fin.index
    df_mops_fin['year'] = df_mops_fin['year'].apply(lambda x: x.split('Q')[0])
    df_mops_fin = df_mops_fin.sort_values(by='year', ascending=False)

    for c in df_mops_fin.columns:
        if fin in str(c):
            pass
        else:
            del df_mops_fin[c]

    # st.write(df_mops_fin)

    return df_mops_fin


def fn_life():
    fn_st_add_space(1)
    st.markdown(f'### 👨‍🌾 :green[$小佃農$] $ 與 $ :blue[$老碼農$] $ 的耕讀生活$')
    tab_0, tab_1, tab_2, tab_3 = st.tabs(['薑', '芥菜', '白蘿蔔', '程式碼'])
    head_sp = 5 * dic_mkd["4sp"]

    tit0 = f'#### {head_sp}$教學參考$'
    tit1 = f'#### {head_sp}:red[$慎選$]$標的$'
    tit2 = f'#### {head_sp}$耐心等待$'
    tit3 = f'#### {head_sp}$期盼收穫$'

    with tab_0:
        cols = st.columns(4)
        video = r'https://www.youtube.com/watch?v=jQtHilLwA44'
        img1 = r'https://scontent.ftpe7-3.fna.fbcdn.net/v/t39.30808-6/325940478_458998859777175_4053406779201999787_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=108&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=NB91HH9uGvoAX83YAhY&_nc_ht=scontent.ftpe7-3.fna&oh=00_AfDN0P69QOnB0jemRIOF9xeXLz4RqYhZwqrgCu31e_Nvdg&oe=64072CB7'
        img2 = r'https://scontent.ftpe7-3.fna.fbcdn.net/v/t39.30808-6/326954716_849046239540921_5946960737469138547_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=102&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=CxnHm3_PRpcAX9FNNIv&_nc_ht=scontent.ftpe7-3.fna&oh=00_AfDXWYIfBn-YPudUreAxTDU8UZuVt_bp0HdhT40LkHN_pQ&oe=6406D70A'
        img3 = r'https://scontent.ftpe7-1.fna.fbcdn.net/v/t39.30808-6/326904370_3285754411675268_7387608385564380001_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=110&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=6waBTE6LhpoAX-iE8I5&_nc_ht=scontent.ftpe7-1.fna&oh=00_AfCEbcWPfx4Jvv0-MGO-Tp5LdC6zI4VMTomjO3sG18DY5A&oe=6406F40C'

        with cols[0]:
            st.markdown(tit0)
            st_player(video, key='video_tab0', playing=False, loop=False, volume=1, height=440, light=True)

        cols[1].markdown(tit1)
        cols[1].image(img1, caption='很快就發芽了')
        cols[2].markdown(tit2)
        cols[2].image(img2, caption='薑黃開的白色花朵，美麗優雅 ~')
        cols[3].markdown(tit3)
        cols[3].image(img3, caption='小農婦自己種的薑')

    with tab_1:
        cols = st.columns(4)
        video = r'https://www.youtube.com/watch?v=yKAUqklC5Hs'
        img1 = r'https://scontent.ftpe8-4.fna.fbcdn.net/v/t39.30808-6/326890989_728929498577116_8254747758524523208_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=102&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=PucGpTuS3VIAX-ToMs-&_nc_ht=scontent.ftpe8-4.fna&oh=00_AfDZxy5SjJqFSbmMPh0FwcMURPlLdCCuPNq67wrVFYFqMQ&oe=63DA93D0'
        img2 = r'https://scontent.ftpe8-3.fna.fbcdn.net/v/t39.30808-6/326730771_737113820959445_2047346049108884382_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=107&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=9Tco-lY_tWcAX_bACUB&_nc_ht=scontent.ftpe8-3.fna&oh=00_AfBLl7cdnP0zjzLdbrkDURuZjIrRGt0DO8kCVSphOcGbQg&oe=63DA9AD0'
        img3 = r'https://scontent.ftpe8-1.fna.fbcdn.net/v/t39.30808-6/325782354_779450000284400_3666154961436129569_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=108&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=u8xNKaRXB_QAX8PaG4M&tn=IlHWvw90GUJy8pGM&_nc_ht=scontent.ftpe8-1.fna&oh=00_AfDnfrqPFIS41R93XXVpVowtFOYyiKEoyMoJ1XH7imZw9Q&oe=63DAD7AC'

        with cols[0]:
            st.markdown(tit0)
            st_player(video, key='video_tab2', playing=False, loop=False, volume=1, height=440, light=True)

        cols[1].markdown(tit1)
        cols[1].image(img1, caption='')
        cols[2].markdown(tit2)
        cols[2].image(img2, caption='')
        cols[3].markdown(tit3)
        cols[3].image(img3, caption='長年菜')

    with tab_2:
        cols = st.columns(4)
        video = r'https://www.youtube.com/watch?v=hlQTmmhMuQ4&t=14s'
        img1 = r'https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.30808-6/309235737_10222161831940357_319357518375648256_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=103&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=M1qUSec2jl0AX9SVEOs&_nc_ht=scontent.ftpe8-2.fna&oh=00_AfCV59IjWkagZUPqnIud3Tu1GuTHWjUtRPuohxeIhjYUnQ&oe=63DA3B64'
        img2 = r'https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.30808-6/314891381_10222426458635859_5303105120234812499_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=103&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=ML4ZjQFxLmUAX-IGcd6&_nc_ht=scontent.ftpe8-2.fna&oh=00_AfC6OF_gAYD9uVWFE_OL6h9l-zA23aIUd6Kqj9MYSPwqIg&oe=63DAFEC9'
        img3 = r'https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.30808-6/320433018_557649275872527_1374980607348320756_n.jpg?stp=cp6_dst-jpg_p843x403&_nc_cat=101&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=21mNspDlvKYAX-Wqp5C&tn=IlHWvw90GUJy8pGM&_nc_ht=scontent.ftpe8-2.fna&oh=00_AfBfKVBF53lQl-KjPz85DxFcHC6rUBWvUe1kyOVMXIZnzg&oe=63DB74FB'
        with cols[0]:
            st.markdown(tit0)
            st_player(video, key='video_tab1', playing=False, loop=False, volume=1, height=440, light=True)

        cols[1].markdown(tit1)
        cols[1].image(img1, caption='')
        cols[2].markdown(tit2)
        cols[2].image(img2, caption='')
        cols[3].markdown(tit3)
        cols[3].image(img3, caption='')

    with tab_3:
        st.markdown(f'#### {dic_mkd["1sp"]} $碼園也是一片綠油油$ ~')
        st.image('coder.png', use_column_width=False, caption='2022年的碼園耕耘 ~')

    img1 = r'https://scontent.ftpe7-1.fna.fbcdn.net/v/t39.30808-6/272767780_10220980105757941_4447844687755244925_n.jpg?stp=c0.88.692.692a_dst-jpg_s851x315&_nc_cat=110&ccb=1-7&_nc_sid=da31f3&_nc_ohc=dPYpb8mBIJsAX-jPQNM&_nc_ht=scontent.ftpe7-1.fna&oh=00_AfB9G24Jon3jyLYcNop9oReEXwk7K3xqewtbbfmm9ygYIw&oe=6406E360'
    img2 = r'https://scontent.ftpe7-1.fna.fbcdn.net/v/t39.30808-6/278861386_10221365854641422_403041763089540585_n.jpg?_nc_cat=110&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=iPc5i4Pa3PEAX-RyANj&_nc_ht=scontent.ftpe7-1.fna&oh=00_AfAKBeCpDS7mcH9B83bmHm0BduodyYjHh2DhAve7uXTLZA&oe=6406967B'
    img3 = r'https://scontent.ftpe7-3.fna.fbcdn.net/v/t39.30808-6/291827655_10221745319487806_9084714485075851384_n.jpg?stp=cp6_dst-jpg&_nc_cat=103&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=Qdssl4eIjh8AX-IdQoi&_nc_ht=scontent.ftpe7-3.fna&oh=00_AfATdonQQYSgsMqxFSxm3C641XtlUWRvVrBHBJCw4yQNmg&oe=6407B9DA'

    cols = st.columns([1.5, 1.5, 2])
    # cols[0].image(img1, width=550)
    cols[0].image(img1, caption='很有愛 ~', use_column_width=True)
    cols[1].image(img2, caption='綠油油 ~', use_column_width=True)
    cols[2].image(img3, caption='一起玩 ~', use_column_width=True)


def fn_idea():
    fig = go.Figure(go.Funnelarea(
        # text=["搜尋網站選股", "基本面分析", "擬訂策略", "觀察驗證"],
        text=["搜尋網站選股", "基本面分析", "擬訂策略", "驗證"],
        values=[5, 4, 3, 2],
        textinfo='text',
        textfont={'size': [20, 20, 20, 20]},
        showlegend=False,
    ))

    fig.update_layout(
        autosize=False,
        width=600,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=140,
            t=40,
            pad=4
        ),
    )
    cols = st.columns([1.7, 3, 1.7])
    cols[1].image('save.png')

    cols = st.columns([0.01, 1.19, 2.6, 0.9, 0.3])
    cols[2].plotly_chart(fig, use_container_width=True)

    for _ in range(4):
        cols[1].write('')
        cols[3].write('')

    cols[1].image('sign.png')
    cols[1].image('word1.png')

    cols[3].image('NoPenCmt.png')
    cols[3].image('word2.png')
    #
    # st.markdown(f'### 👨‍🌾 :green[$小佃農$] $ 與 $ :blue[$老碼農$] $ 的耕讀生活$')
    # tab_0, tab_1, tab_2, tab_3 = st.tabs(['薑', '芥菜', '白蘿蔔', '程式碼'])
    # head_sp = 5*dic_mkd["4sp"]
    #
    # tit0 = f'#### {head_sp}$教學參考$'
    # tit1 = f'#### {head_sp}:red[$慎選$]$標的$'
    # tit2 = f'#### {head_sp}$耐心等待$'
    # tit3 = f'#### {head_sp}$期盼收穫$'
    #
    # with tab_0:
    #     cols = st.columns(4)
    #     video = r'https://www.youtube.com/watch?v=jQtHilLwA44'
    #     img1 = r'https://scontent.ftpe8-1.fna.fbcdn.net/v/t39.30808-6/325940478_458998859777175_4053406779201999787_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=108&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=51ohNMr5BlsAX_MQAdk&tn=IlHWvw90GUJy8pGM&_nc_ht=scontent.ftpe8-1.fna&oh=00_AfBVKtoYQkUMgfSF95Fd_sABHjg8QgjElGYiYGDcxy6mOA&oe=63DBAB37'
    #     img2 = r'https://scontent.ftpe8-4.fna.fbcdn.net/v/t39.30808-6/326954716_849046239540921_5946960737469138547_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=102&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=EOmsGrdZbSIAX8KeoOQ&_nc_ht=scontent.ftpe8-4.fna&oh=00_AfDSTeRwLxIunhy2mor8T7lWowtLtamgHCximWtGo7xk9Q&oe=63DB558A'
    #     img3 = r'https://scontent.ftpe8-4.fna.fbcdn.net/v/t39.30808-6/326904370_3285754411675268_7387608385564380001_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=110&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=Ue4gNZR9BcwAX8O5duO&_nc_ht=scontent.ftpe8-4.fna&oh=00_AfDY1hourtNnXSBvXP7xgxgpkb8SQS1xbBIURMLLH_HSrg&oe=63DB728C'
    #
    #     with cols[0]:
    #         st.markdown(tit0)
    #         st_player(video, key='video_tab0', playing=False, loop=False, volume=1, height=440, light=True)
    #
    #     cols[1].markdown(tit1)
    #     cols[1].image(img1, caption='很快就發芽了')
    #     cols[2].markdown(tit2)
    #     cols[2].image(img2, caption='薑黃開的白色花朵，美麗優雅 ~')
    #     cols[3].markdown(tit3)
    #     cols[3].image(img3, caption='小農婦自己種的薑')
    #
    # with tab_1:
    #     cols = st.columns(4)
    #     video = r'https://www.youtube.com/watch?v=yKAUqklC5Hs'
    #     img1 = r'https://scontent.ftpe8-4.fna.fbcdn.net/v/t39.30808-6/326890989_728929498577116_8254747758524523208_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=102&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=PucGpTuS3VIAX-ToMs-&_nc_ht=scontent.ftpe8-4.fna&oh=00_AfDZxy5SjJqFSbmMPh0FwcMURPlLdCCuPNq67wrVFYFqMQ&oe=63DA93D0'
    #     img2 = r'https://scontent.ftpe8-3.fna.fbcdn.net/v/t39.30808-6/326730771_737113820959445_2047346049108884382_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=107&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=9Tco-lY_tWcAX_bACUB&_nc_ht=scontent.ftpe8-3.fna&oh=00_AfBLl7cdnP0zjzLdbrkDURuZjIrRGt0DO8kCVSphOcGbQg&oe=63DA9AD0'
    #     img3 = r'https://scontent.ftpe8-1.fna.fbcdn.net/v/t39.30808-6/325782354_779450000284400_3666154961436129569_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=108&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=u8xNKaRXB_QAX8PaG4M&tn=IlHWvw90GUJy8pGM&_nc_ht=scontent.ftpe8-1.fna&oh=00_AfDnfrqPFIS41R93XXVpVowtFOYyiKEoyMoJ1XH7imZw9Q&oe=63DAD7AC'
    #
    #     with cols[0]:
    #         st.markdown(tit0)
    #         st_player(video, key='video_tab2', playing=False, loop=False, volume=1, height=440, light=True)
    #
    #     cols[1].markdown(tit1)
    #     cols[1].image(img1, caption='')
    #     cols[2].markdown(tit2)
    #     cols[2].image(img2, caption='')
    #     cols[3].markdown(tit3)
    #     cols[3].image(img3, caption='長年菜')
    #
    # with tab_2:
    #     cols = st.columns(4)
    #     video = r'https://www.youtube.com/watch?v=hlQTmmhMuQ4&t=14s'
    #     img1 = r'https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.30808-6/309235737_10222161831940357_319357518375648256_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=103&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=M1qUSec2jl0AX9SVEOs&_nc_ht=scontent.ftpe8-2.fna&oh=00_AfCV59IjWkagZUPqnIud3Tu1GuTHWjUtRPuohxeIhjYUnQ&oe=63DA3B64'
    #     img2 = r'https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.30808-6/314891381_10222426458635859_5303105120234812499_n.jpg?stp=cp6_dst-jpg_p720x720&_nc_cat=103&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=ML4ZjQFxLmUAX-IGcd6&_nc_ht=scontent.ftpe8-2.fna&oh=00_AfC6OF_gAYD9uVWFE_OL6h9l-zA23aIUd6Kqj9MYSPwqIg&oe=63DAFEC9'
    #     img3 = r'https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.30808-6/320433018_557649275872527_1374980607348320756_n.jpg?stp=cp6_dst-jpg_p843x403&_nc_cat=101&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=21mNspDlvKYAX-Wqp5C&tn=IlHWvw90GUJy8pGM&_nc_ht=scontent.ftpe8-2.fna&oh=00_AfBfKVBF53lQl-KjPz85DxFcHC6rUBWvUe1kyOVMXIZnzg&oe=63DB74FB'
    #     with cols[0]:
    #         st.markdown(tit0)
    #         st_player(video, key='video_tab1', playing=False, loop=False, volume=1, height=440, light=True)
    #
    #     cols[1].markdown(tit1)
    #     cols[1].image(img1, caption='')
    #     cols[2].markdown(tit2)
    #     cols[2].image(img2, caption='')
    #     cols[3].markdown(tit3)
    #     cols[3].image(img3, caption='')
    #
    # with tab_3:
    #     st.markdown(f'#### {dic_mkd["1sp"]} $碼園也是一片綠油油$ ~')
    #     st.image('coder.png', use_column_width=False)
    #
    # img1 = r'https://scontent.ftpe8-4.fna.fbcdn.net/v/t39.30808-6/272767780_10220980105757941_4447844687755244925_n.jpg?stp=c0.88.692.692a_dst-jpg_s851x315&_nc_cat=110&ccb=1-7&_nc_sid=da31f3&_nc_ohc=Qj6mMXyy3r0AX9N2mFB&tn=IlHWvw90GUJy8pGM&_nc_ht=scontent.ftpe8-4.fna&oh=00_AfBu4fier5xgYx5hjMdN-iQc8_trlhkIIw4nazRTOtjXOA&oe=63D76D60'
    # img2 = r'https://scontent.ftpe8-4.fna.fbcdn.net/v/t39.30808-6/278861386_10221365854641422_403041763089540585_n.jpg?stp=c0.85.702.702a_dst-jpg_s851x315&_nc_cat=110&ccb=1-7&_nc_sid=da31f3&_nc_ohc=80sDHAkO4YgAX99LFOn&_nc_ht=scontent.ftpe8-4.fna&oh=00_AfCbYWGaZkIibmYf3iaypNLw8fgJi8nw2VyUXOWeqJPhKw&oe=63D91ABB'
    # img3 = r'https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.30808-6/291827655_10221745319487806_9084714485075851384_n.jpg?stp=cp6_dst-jpg&_nc_cat=103&ccb=1-7&_nc_sid=8bfeb9&_nc_ohc=yFFOiVTkM04AX8tc2TG&_nc_ht=scontent.ftpe8-2.fna&oh=00_AfB484X6Uf13k0doW6thK9RGETeeFGORYjS04_V7ohCKSA&oe=63D843DA'
    #
    # cols = st.columns([1.5, 1.5, 2])
    # # cols[0].image(img1, width=550)
    # cols[0].image(img1, caption='為愛耕耘', use_column_width=True)
    # cols[1].image(img2, caption='豆豆龍 🎵 ~', use_column_width=True)
    # cols[2].image(img3, caption='一起來玩', use_column_width=True)


def fn_color_roe_season(x):
    css = ''
    css_h = 'background-color: pink; color: black'
    css_m = 'background-color: lightyellow; color: black'
    css_l = 'background-color: lightgreen; color: black'

    if len(str(x)) > 0:
        v = float(x)
        if v >= 4.0:
            css = css_h
        elif v >= 2.0:
            css = css_m
        else:
            css = css_l

    return css


def fn_color_roe_year(x):
    css = ''
    css_h = 'background-color: pink; color: black'
    css_m = 'background-color: lightyellow; color: black'
    css_l = 'background-color: lightgreen; color: black'

    if len(str(x)) > 0:
        v = float(x)
        if v >= 16.0:
            css = css_h
        elif v >= 8.0:
            css = css_m
        else:
            css = css_l

    return css


def fn_get_color(income, eps, cash, th):
    try:
        color_income = 'red' if float(income) >= th else 'green'
    except:
        color_income = 'blue'

    try:
        color_eps = 'red' if float(eps) >= th else 'green'
    except:
        color_eps = 'blue'

    try:
        color_cash = 'red' if float(cash) >= th else 'green'
    except:
        color_cash = 'blue'

    return color_income, color_eps, color_cash


def fn_light_color(x):
    i = int(x)

    if i >= 38:
        c = 'red'
    elif i >= 32:
        c = 'orange'
    elif i >= 23:
        c = 'green'
    elif i >= 17:
        c = 'yellow'
    else:
        c = 'blue'

    return c


def fn_show_basic_idx(df, df_mops, key='hist_price'):
    sep = ' '
    df['sid_name'] = df['代碼'] + sep + df['名稱']
    cols = st.columns([1, 0.1, 2.9])
    df_all = dic_df['stock_all']

    with cols[0].form(key=f'form_{key}'):

        cols2 = st.columns([2, 0.1, 3, 1.8])

        # dft_sid = '2404' if key == 'basic_idx' else df['代碼'].values[0]

        dft_sid = dic_df['df_win']['代碼'].values[0] if key == 'basic_idx' else df['代碼'].values[0]

        sid = cols2[0].text_input('股票代碼:', value=dft_sid)

        df_sid = df_all[df_all['sid'] == sid]

        sid_name = df_sid['sid_name'].values[0] if df_sid.shape[0] > 0 else '不在資料庫'
        cols2[2].write('')
        cols2[2].write('')
        cols2[2].write(f':orange[${sid_name}$]')

        cols2[-1].write('')
        cols2[-1].write('')
        cols2[-1].form_submit_button('選擇')

    df_sid_p = df_sid.copy()

    if df_sid.shape[0] == 0:
        st.error(f'抱歉 股票代碼 {sid} 尚未收錄至本資料庫(目前只收錄了{df_all["sid"].nunique()}檔) ~')
        return None
        # assert False, f'Sorry 您輸入的股票代碼 {sid} 不在資料庫喔 ~'
    else:
        market = df_sid["市場別"].values[0]

    # sid_name = df_sid['sid_name'].values[0]

    url_WantRich = rf'{dic_url["WantRich"]}{sid}'
    url_FB = rf'{dic_url["FindBillion"]}{sid}'
    url_PC = rf'{dic_url["PChome"]}{sid}.html'
    url_CMoney = rf'{dic_url["CMoney"]}{sid}'
    url_Wg = rf'{dic_url["Wantgoo"]}{sid}/profitability/roe-roa'
    url_Cnyes = rf'{dic_url["Cnyes"]}{sid}'
    url_dog = rf'{dic_url["dog"]}{sid}/stock-health-check'
    url_Yahoo = rf'{dic_url["Yahoo"]}{sid}.TW{"O" if market == "上櫃" else ""}/health-check'
    try:
        df_mop = fn_get_mops(df_mops, sid)
        df_roe = fn_get_mops_fin("ROE", sid)
        # st.write(df_roe)
        df_roa = fn_get_mops_fin("ROA", sid)
        df_opm = fn_get_mops_fin("OPM", sid)
        df_dr = fn_get_mops_fin("DR", sid)
        dr_cf = fn_get_mops_fin("OCF", sid)
        df_fin = pd.concat([df_roe, df_roa, df_opm, df_dr, dr_cf], axis=1)
        # st.write(df_fin)
        df_fin.reset_index(names='年/季', inplace=True)
        df_fin['year'] = df_fin['年/季'].apply(lambda x: x.split('Q')[0])
        df_fin['season'] = df_fin['年/季'].apply(lambda x: x.split('Q')[-1])
        df_fin.sort_values(by=['year', 'season'], ascending=[False, False], inplace=True, ignore_index=True)
        del df_fin['year']
        del df_fin['season']
        basic = fn_basic_rule(sid, df_mops)
    except:
        df_fin = pd.DataFrame()
        basic = ''

    mkd_space = f'{9 * dic_mkd["2sp"]}'

    cols[0].write('')
    cols[0].write('')
    cols[0].markdown(f'$市場別:$ ${market}$ - ${df_sid["產業別"].values[0]}$')
    # cols[0].markdown(f'產業別: {df_sid["產業別"].values[0]}')

    df_report = dic_df['report']
    report_lnk = 'NA' if sid not in df_report['sid'].values else df_report[df_report['sid'] == sid]['report'].values[0]
    report_date = 'NA' if report_lnk == 'NA' else report_lnk.split('M00')[0].split(sid)[-1]

    today = datetime.datetime.today()
    color = 'blue'
    try:
        if int(today.year) == int(report_date[:4]) and int(today.month) == int(report_date[4:6]):
            color = 'red'
    except:
        pass

    report_date = f'$中文簡報-{report_date}$'
    cmp_report = '$NA$' if report_lnk == 'NA' else f'[:{color}[{report_date}]]({report_lnk})'

    df_tdcc = dic_df['tdcc']
    df_tdcc_sid = df_tdcc[df_tdcc['證券代號'] == sid]
    df_rank_15 = df_tdcc_sid[df_tdcc_sid['持股分級'] == '15']
    df_rank_17 = df_tdcc_sid[df_tdcc_sid['持股分級'] == '17']

    n_share = int(int(df_rank_17['股數'].values[0]) / 1000)
    n_share = n_share if int(n_share) < 10000 else '約 ' + str(int(int(n_share) / 10000)) + '萬'

    n_owner = df_rank_17['人數'].values[0]
    n_owner = n_owner if int(n_owner) < 10000 else '約 ' + str(int(int(n_owner) / 10000)) + '萬'

    r_big = df_rank_15['占集保庫存數比例%'].values[0]
    r_date = df_rank_15['資料日期'].values[0].replace(f'{datetime.datetime.today().year}', '')
    lnk_tdcc = dic_url['tdcc']  # r'https://www.tdcc.com.tw/portal/zh/smWeb/qryStock'

    lnk = f'https://www.cnyes.com/twstock/{sid}/company/profile'
    cols[0].markdown(f'[$法說會:$]({lnk}) {cmp_report}')
    cols[0].markdown(f'$基本面:$ {basic}')

    cols[0].markdown(f'$專業的:$ [$財報狗$]({url_dog})、[$旺得富$]({url_WantRich})、')
    cols[0].markdown(f'{mkd_space}[$玩股網$]({url_Wg})、[$鉅亨網$]({url_Cnyes})、')
    cols[0].markdown(f'{mkd_space}[$CMoney$]({url_CMoney})、[$Yahoo$]({url_Yahoo})、')
    cols[0].markdown(f'{mkd_space}[$FindBillion$]({url_FB})、')
    cols[0].markdown(f'{mkd_space}[$PChome$]({url_PC})、')

    df_sid = fn_get_stock_price(sid, days=200)
    sid_price = round(df_sid['Close'].values[-1], 1)

    if df_sid.shape[0] > 0:

        df_sn = dic_df['season']
        df_sn = df_sn[df_sn['sid'] == sid]
        df_sn['yr_sn'] = df_sn['year'] + '<br>' + df_sn['season']
        df_sn['yr_sn'] = df_sn['yr_sn'].apply(lambda x: x.replace(' ', ''))

        with cols[2]:
            df_per = dic_mops['per']
            df_yh = dic_df['Yahoo_Health']
            df_yh_sid = df_yh[df_yh['sid'] == sid]
            df_sid_l = df_sid_p.iloc[-1, :]
            br = dic_mkd["2sp"]

            # fn_st_add_space(1)

            cols = st.columns(3)
            font_size = '#####' if len(sid_name) < 3 else '######'
            cols[1].error(f'{font_size} '
                          f'{dic_mkd["1sp"]}${sid}\ {sid_name}${br}'
                          f'$股價: {sid_price} 元$')

            cols = st.columns(6)

            with cols[1]:

                if str(sid) in df_per['股票代號'].values:
                    df_per_sid = df_per[df_per['股票代號'] == str(sid)]
                    per = df_per_sid['本益比'].values[0]
                    yr = df_per_sid['殖利率(%)'].values[0]
                    eps = round(sid_price / float(per), 1)
                    date_info = df_per_sid['日期'].values[0]
                    market = df_per_sid["市場別"].values[0]
                    if market == '市':
                        link = r'https://www.twse.com.tw/zh/page/trading/exchange/BWIBBU.html'
                    else:
                        link = r'https://www.tpex.org.tw/web/stock/aftertrading/peratio_stk/pera.php?l=zh-tw'

                    color_per = 'red' if float(per) <= 12 else 'green'
                    color_cash = 'red' if float(yr) >= 5 else 'green'
                    color_eps = 'red'

                    st.markdown(f'[$基本資料$]({link})$({dic_mops["per_date"]})$')
                    st.markdown(f'$本益比:$ [:{color_per}[${per} 倍$]]({link})')
                    st.markdown(f'$EPS:$ {br}{br} [:{color_eps}[${eps}元$]]({link})')
                    st.markdown(f'$殖利率:$ [:{color_cash}[${yr}\%$]]({link})')

            with cols[0]:
                sid_grow = df_yh_sid['grow'].values[0].replace('%', '')
                sid_stable = df_yh_sid['stable'].values[0].replace('%', '')
                sid_yh_link = df_yh_sid['link'].values[0]
                try:
                    color_grow = 'red' if float(sid_grow) >= 60 else 'green'
                    color_stable = 'red' if float(sid_stable) >= 60 else 'green'
                except:
                    color_grow = 'blue'
                    color_stable = 'blue'

                st.markdown(f'[$營運健診$]({sid_yh_link})')
                st.markdown(f'$獲利成長:$ [:{color_grow}[${sid_grow}分$]]({sid_yh_link})')
                st.markdown(f'$財務穩健:$ [:{color_stable}[${sid_stable}分$]]({sid_yh_link}) ')

            with cols[2]:

                win_income = df_sid_l['勝率(%)_營收']
                win_eps = df_sid_l['勝率(%)_EPS']
                win_cash = df_sid_l['勝率(%)_殖利率']

                color_income, color_eps, color_cash = fn_get_color(win_income, win_eps, win_cash, 50)

                def fn_unit(v):
                    return "\ " if v == "" else v + '\%'

                st.markdown(f'[$勝率分析$]({sid_yh_link})')
                st.markdown(f'$依營收:$ [:{color_income}[${fn_unit(win_income)}$]]({sid_yh_link})')
                st.markdown(f'$依EPS:$ [:{color_eps}[${fn_unit(win_eps)}$]]({sid_yh_link})')
                st.markdown(f'$依殖率:$ [:{color_cash}[${fn_unit(win_cash)}$]]({sid_yh_link})')

            with cols[3]:

                p_income = df_sid_l['合理價_營收']
                p_eps = df_sid_l['合理價_EPS']
                p_cash = df_sid_l['合理價_殖利率']

                def fn_unit(v):
                    return "\ " if v == "" else v + '元'

                color_income, color_eps, color_cash = fn_get_color(p_income, p_eps, p_cash, sid_price)
                st.markdown(f'[$價格推估$]({sid_yh_link})')
                st.markdown(f'$依營收:$ [:{color_income}[${fn_unit(p_income)}$]]({sid_yh_link})')
                st.markdown(f'$依EPS:$ [:{color_eps}[${fn_unit(p_eps)}$]]({sid_yh_link})')
                st.markdown(f'$依殖率:$ [:{color_cash}[${fn_unit(p_cash)}$]]({sid_yh_link})')

            with cols[4]:

                c_income = df_sid_l['相關性_營收'].split(' ')[-1]
                c_eps = df_sid_l['相關性_EPS'].split(' ')[-1]
                c_cash = df_sid_l['相關性_殖利率'].split(' ')[-1]

                color_income, color_eps, color_cash = fn_get_color(c_income, c_eps, c_cash, 0.65)

                def fn_unit(v):
                    return "\ " if v == "" else v + ''

                st.markdown(f'[$相關性$]({sid_yh_link})')
                st.markdown(f'$依營收:$ [:{color_income}[${fn_unit(c_income)}$]]({sid_yh_link})')
                st.markdown(f'$依EPS:$ [:{color_eps}[${fn_unit(c_eps)}$]]({sid_yh_link})')
                st.markdown(f'$依殖率:$ [:{color_cash}[${fn_unit(c_cash)}$]]({sid_yh_link})')

            with cols[5]:
                st.markdown(f'[$股權分布$]({lnk_tdcc})$({r_date})$')
                st.markdown(f'$股票數:$ [:blue[${n_share} 張$]]({lnk_tdcc}) ')
                st.markdown(f'$股東數:$ [:blue[${n_owner} 人$]]({lnk_tdcc})')
                st.markdown(f'$大戶比:$ [:blue[${r_big} \%$]]({lnk_tdcc})')

            df_mop['年度'] = df_mop['year'].apply(lambda x: int(x) + 1911)
            cols = [c for c in df_mop.columns if '-' in c]
            df_mop = df_mop[['年度'] + [c for c in cols if '權益' in c] + [c for c in cols if '權益' not in c]]
            df_mop.sort_values(by=['年度'], ascending=[False], ignore_index=True, inplace=True)
            df_mop['年度'] = df_mop['年度'].apply(lambda x: str(x) + ' 年')

            df_mop_show = df_mop.style.applymap(fn_color_roe_year,
                                                subset=[c for c in df_mop.columns if '權益' in c])

            fn_st_add_space(1)
            tab_tech, tab_basic, tab_light, tab_raw, tab_src = st.tabs(
                ['技術指標', '基本指標', '景氣循環', '詳細數據', '資料來源'])
            y_fr = datetime.datetime.today().year - 5

            with tab_light:
                df_lt = dic_df['light']
                df_m = dic_df['month']
                df_m = df_m[df_m['sid'] == sid]
                df_m['month'] = df_m['month'].apply(lambda x: '0' + x if len(x) == 1 else x)
                df_m['y-m'] = df_m['year'] + '-' + df_m['month']
                c_bypass = ['y-m', '景氣對策信號(燈號)']
                for c in df_lt.columns:
                    if c == '景氣對策信號(分)':  # if c not in c_bypass:
                        cols = st.columns([4, 1])

                        df_lt_1 = df_lt[df_lt['y-m'] >= df_m['y-m'].values[0]]
                        df_m_1 = df_m[df_m['y-m'] <= df_lt_1['y-m'].values[-1]]
                        nmi = round(normalized_mutual_info_score(df_lt_1[c], df_m_1['ave']), 2)

                        fig1 = fn_gen_plotly_line(df_lt, 'y-m', c, op=0.8, color='dodgerblue')
                        fig2 = fn_gen_plotly_line(df_m, 'y-m', 'ave', op=0.6, color='red')

                        if '景氣對策' in c:

                            with cols[-1]:
                                fn_st_add_space(3)
                                b = '掌握市場週期'
                                st.markdown(f'{dic_mkd["4sp"]}《 [${b}$]({dic_book_lnk[b]}) 》')
                                st.image(dic_book_img[b], use_column_width=True)

                            df_lt['color'] = df_lt[c].apply(fn_light_color)
                            fig1.update_traces(
                                marker=dict(size=14, color=df_lt['color'].values, opacity=0.3,
                                            line=dict(width=1, color='blue')))
                            #
                            # cols[-1].write('')
                            # cols[-1].write('')
                            # light_on = cols[-1].radio('$景氣燈號$', ['ON', 'OFF'], index=0, key='light')
                            # if light_on == 'ON':
                            #     df_lt['color'] = df_lt[c].apply(fn_light_color)
                            #     fig1.update_traces(
                            #         marker=dict(size=14, color=df_lt['color'].values, opacity=0.3,
                            #                     line=dict(width=1, color='blue')))
                            #

                        subfig = make_subplots(specs=[[{'secondary_y': True}]])
                        subfig.add_traces(fig1.data + fig2.data, secondary_ys=[False, True])
                        subfig.update_layout(
                            title_text=f'  🔵 國發會 ' + c + f' v.s. 🔴 {sid} {sid_name} 股價(元)  MI = {nmi}',
                            title_font_size=18,
                            xaxis={'showgrid': True},
                            yaxis={'showticklabels': True,
                                   'showgrid': True,
                                   'showspikes': True,
                                   'spikethickness': 1,
                                   'spikecolor': "grey",
                                   'spikedash': 'solid',
                                   'spikemode': "across",
                                   'spikesnap': "cursor",
                                   'tickfont_color': 'dodgerblue',
                                   },
                            yaxis2={'showticklabels': True,
                                    'showgrid': False,
                                    'tickfont_color': 'red',
                                    },
                        )
                        subfig.update_xaxes(tickfont_size=16, range=['2015-1', '2024-1'])
                        subfig.update_yaxes(tickfont_size=16)

                        cols[0].plotly_chart(subfig, use_container_width=True)

            with tab_tech:
                fn_st_add_space(1)
                fr = df_sid_p['date'].min()
                to = df_sid_p['date'].max()
                mk = df_sid_p['市場別'].values[-1]
                mk = mk + '-' if len(str(mk)) > 0 else ''
                indu = df_sid_p['產業別'].values[-1]
                title = f'{sid} {sid_name} ({mk}{indu})'
                fig = fn_get_stock_price_plt(df_sid, df_p=df_sid_p, watch=[fr, to], height=350, showlegend=True,
                                             title=title, op=0.7)

                fig.update_layout(
                    title_font_size=18,
                    yaxis={'showticklabels': True,
                           'showgrid': False,
                           'showspikes': True,
                           'spikethickness': 1,
                           'spikecolor': "grey",
                           'spikedash': 'solid',
                           'spikemode': "across",
                           'spikesnap': "cursor",
                           'tickfont_color': 'dodgerblue',
                           },
                    yaxis2={'showticklabels': True,
                            'showgrid': True,
                            'tickfont_color': 'red',
                            },
                )
                fig.update_xaxes(tickfont_size=15)
                fig.update_yaxes(tickfont_size=15)
                st.plotly_chart(fig, use_container_width=True)

            with tab_basic:

                if df_fin.shape[0] == 0:
                    return

                df_month = df_sn[df_sn['year'].apply(lambda x: int(x) >= y_fr)]
                # st.write(df_fin)
                df_fin_b = df_fin.sort_index(ascending=False, ignore_index=True)
                df_fin_b = df_fin_b[df_fin_b['年/季'].apply(lambda x: int(x.split('Q')[0]) >= y_fr)]
                df_fin_b['color'] = df_fin_b['年/季'].apply(lambda x: 2 if int(x.split('Q')[0]) % 2 == 1 else 1)
                df_fin_b['年/季'] = df_fin_b['年/季'].apply(lambda x: str(x).replace('Q', '<br>Q'))

                df_fin_b = df_fin_b[df_fin_b['權益報酬率(ROE)'].apply(lambda x: len(str(x)) > 0)]

                df_fin_b.reset_index(inplace=True, drop=True)

                Q_last = df_fin_b['年/季'].values[-1].split('<br>')[-1]

                df_mop_b = df_mop.sort_index(ascending=False, ignore_index=True)
                df_mop_b.reset_index(inplace=True, drop=True)

                tab_year, tab_season, tab_income = st.tabs(['年度', '季度', '月營收'])

                with tab_season:

                    for f in df_fin_b.columns:
                        if f == 'color' or f == '年/季' or 'ROA' in f or 'DR' in f:
                            pass
                        else:

                            if df_month.shape[0] > 0:
                                title = f'{" "*8}{sid} {sid_name}   {f} vs 股價走勢'

                                # colors = [dic_colors["c1"] if c == 1 else dic_colors["c2"] for c in df_fin_b["color"]]

                                colors = ["orange" if Q_last in q else dic_colors["c1"] for q in df_fin_b['年/季']]

                                df_fin_b_q = df_fin_b[df_fin_b['年/季'].apply(lambda x: Q_last in x)]

                                color_last = 'pink' if float(df_fin_b_q[f].values[-1]) >= float(
                                    df_fin_b_q[f].values[-2]) else 'lightgreen'

                                colors = colors[:-1] + [color_last]

                                fig1 = fn_gen_plotly_bar(df_fin_b, '年/季', f,
                                                         v_h='v',
                                                         op=[0.5 for i in range(df_fin_b.shape[0] - 1)] + [1.0],
                                                         colors=colors, showscale=False,
                                                         textposition='outside', text_auto=True,
                                                         showspike=True)

                                fig2 = fn_gen_plotly_line(df_month, 'yr_sn', 'ave', op=0.4)

                                ticktext = [x if Q_last in x else '' for x in df_month['yr_sn']]
                                tickvals = list(range(0, len(ticktext)))

                                subfig = make_subplots(specs=[[{'secondary_y': True}]])
                                subfig.add_traces(fig1.data + fig2.data, secondary_ys=[False, True])
                                subfig.update_layout(coloraxis_showscale=False,
                                                     title_text=title,
                                                     title_font_size=18,
                                                     yaxis={'showticklabels': False,
                                                            'showgrid': False,
                                                            'showspikes': True,
                                                            'spikethickness': 1,
                                                            'spikecolor': "grey",
                                                            'spikedash': 'solid',
                                                            'spikemode': "across",
                                                            'spikesnap': "cursor",
                                                            },
                                                     yaxis2={'showticklabels': True,
                                                             'showgrid': False,
                                                             'tickfont_color': 'red',
                                                             'tickfont_size': 14,
                                                             },
                                                     )
                                # st.write(ticktext)
                                subfig.update_xaxes(ticktext=ticktext, tickvals=tickvals, tickmode='array',
                                                    tickfont_size=14)
                                st.plotly_chart(subfig, use_container_width=True)

                            else:
                                colors = [dic_colors["c1"] if c == 1 else dic_colors["c2"] for c in df_fin_b["color"]]
                                colors = colors[:-1] + ["orange"]
                                fig = fn_gen_plotly_bar(df_fin_b, '年/季', f, title=f'{sid} {sid_name}   {f}',
                                                        v_h='v',
                                                        op=[0.5 for i in range(df_fin_b.shape[0] - 1)] + [1.0],
                                                        colors=colors, showscale=False,
                                                        textposition='outside', text_auto=True)

                                fig.update_layout(coloraxis_showscale=False,
                                                  # title_text=title,
                                                  title_font_size=18,
                                                  yaxis={'showticklabels': False,
                                                         'showgrid': False,
                                                         'showspikes': True,
                                                         'spikethickness': 1,
                                                         'spikecolor': "grey",
                                                         'spikedash': 'solid',
                                                         'spikemode': "across",
                                                         'spikesnap': "cursor",
                                                         },
                                                  )

                                cols = st.columns([3.5, 1])
                                subfig.update_xaxes(tickfont_size=14)
                                cols[0].plotly_chart(fig, use_container_width=True)

                with tab_year:

                    # st.write(df_mop_b)

                    for f in df_mop_b.columns:
                        if f == '年度':
                            pass
                        else:
                            colors = [dic_colors["c1"] for _ in df_mop_b['年度']]
                            colors = colors[:-1] + ["orange"]
                            title = f'{" "*8}{sid} {sid_name}   {f.split("-")[-1]}'

                            fig = fn_gen_plotly_bar(df_mop_b, '年度', f, title=title,
                                                    v_h='v', op=[0.5 for i in range(df_mop_b.shape[0] - 1)] + [1.0],
                                                    colors=colors, showscale=False,
                                                    textposition='outside', text_auto=True, color_mid=None,
                                                    showspike=True)

                            if '權益報酬率' in f:
                                fig.add_hline(y=15, line_width=3, line_dash="dash", line_color="red", opacity=0.4,
                                              annotation_text="近3年ROE > 15%",
                                              annotation_font_size=17,
                                              annotation_font_color="red",
                                              annotation_position="top right")

                                fig.add_hline(y=8, line_width=3, line_dash="dash", line_color="red", opacity=0.6,
                                              annotation_text="歷年ROE > 8%",
                                              annotation_font_size=17,
                                              annotation_font_color="red",
                                              annotation_position="top left")

                            fig.update_layout(coloraxis_showscale=False,
                                              title_text=title,
                                              title_font_size=18,
                                              yaxis={'showticklabels': False,
                                                     'showgrid': False})

                            cols = st.columns([1, 1])
                            fig.update_xaxes(tickfont_size=16)
                            cols[0].plotly_chart(fig, use_container_width=True)

                with tab_income:
                    fn_st_add_space(1)
                    # 'https://www.cnyes.com/twstock/2404/financials/sales'
                    link = dic_url['Cnyes'] + f'{sid}/financials/sales'
                    st.markdown(f'##### '
                                f'[:orange[$月營收$]]({link})')

            with tab_raw:

                if df_fin.shape[0] == 0:
                    return

                df_fin_show = df_fin.style.applymap(fn_color_roe_season,
                                                    subset=[c for c in df_fin.columns if '權益' in c])

                tab_season, tab_year = st.tabs(['季度', '年度'])

                with tab_season:
                    fn_st_add_space(1)
                    st.markdown(f'##### $基本面指標 (季度):$')
                    st.dataframe(df_fin_show)

                with tab_year:
                    fn_st_add_space(1)
                    st.markdown(f'##### $基本面指標 (年度):$')
                    st.dataframe(df_mop_show)

            with tab_src:
                fn_st_add_space(1)
                url = r'https://mopsfin.twse.com.tw/'

                src1 = '臺灣證券交易所'
                lnk1 = r'https://www.twse.com.tw/zh/page/trading/exchange/BWIBBU.html'
                lnk11 = r'https://www.twse.com.tw/zh/trading/historical/fmsrfk.html'
                src2 = '證券櫃檯買賣中心'
                lnk2 = r'https://www.tpex.org.tw/web/stock/aftertrading/peratio_stk/pera.php?l=zh-tw'
                lnk21 = r'https://www.tpex.org.tw/web/stock/statistics/monthly/st44.php?l=zh-tw'

                st.markdown(f'###### $資料來源$:')
                st.markdown(f'$EPS:$ [${src1}$]({lnk1}) $(每日更新)$')
                st.markdown(f'$EPS:$ [${src2}$]({lnk2}) $(每日更新)$')
                st.markdown(f'$ROE:$ [$公開資訊觀測站 > 獲利能力 > 權益報酬率$]({url}) $(每季更新)$')
                st.markdown(f'$ROA:$ [$公開資訊觀測站 > 獲利能力 > 資產報酬率$]({url}) $(每季更新)$')
                st.markdown(f'$OPM:$ [$公開資訊觀測站 > 獲利能力 > 營業利益率$]({url}) $(每季更新)$')
                st.markdown(f'$DR:\ $ [$公開資訊觀測站 > 財務結構 > 負債佔資產比率$]({url}) $(每季更新)$')
                st.markdown(f'$OCF:$ [$公開資訊觀測站 > 現金流量 > 營業現金對負債比$]({url}) $(每季更新)$')
                st.markdown(
                    f'$ROE:$ [$公開資訊觀測站 > 彙總報表 > 營運概況 > 財務比率分析 > 採IFRSs後 > 財務分析資料查詢彙總表$](https://mops.twse.com.tw/mops/web/t51sb02_q1) $(每年 4 月 1 日更新)$')
                st.markdown(
                    f'$OPM:$ [$公開資訊觀測站 > 彙總報表 > 營運概況 > 財務比率分析 > 採IFRSs後 > 營益分析查詢彙總表$](https://mops.twse.com.tw/mops/web/t163sb06) $(每季更新)$')
                st.markdown(
                    f'$營收:$ [$公開資訊觀測站 > 彙總報表 > 營運概況 > 每月營收 > 採IFRSs後每月營業收入彙總表$](https://mops.twse.com.tw/mops/web/t21sc04_ifrs) $(每月11日更新)$')
                st.markdown(f'$月成交資訊:$ [${src1}$]({lnk11})')
                st.markdown(f'$月成交資訊:$ [${src2}$]({lnk21})')

        # with tab_tech:
        #     fn_st_add_space(1)
        #     # st.markdown(f'##### :red[{sid_name}] {dic_mkd["2sp"]} 技術面指標:')
        #
        #     # days_ago = -1 * days[sid_order.index(n_s)]
        #     fr = df_sid_p['date'].min()
        #     to = df_sid_p['date'].max()
        #     # df_p = df_sid[df_sid['sid'] == sid]
        #     mk = df_sid_p['市場別'].values[-1]
        #     mk = mk + '-' if len(str(mk)) > 0 else ''
        #     indu = df_sid_p['產業別'].values[-1]
        #     title = f'{sid} {sid_name} ({mk}{indu})'
        #     fig = fn_get_stock_price_plt(df_sid, df_p=df_sid_p, watch=[fr, to], height=350, showlegend=True,
        #                                  title=title, op=0.7)
        #
        #     st.plotly_chart(fig, use_container_width=True)


def fn_get_yh_grow(sid):
    df_yh = dic_df['Yahoo_Health']
    df_yf_sid = df_yh[df_yh['sid'] == sid]
    if df_yf_sid.shape[0] > 0:
        grow = df_yf_sid['grow'].values[0].replace('%', '')
    else:
        grow = 'NA'

    return grow


def fn_get_yh_stable(sid):
    df_yh = dic_df['Yahoo_Health']
    df_yf_sid = df_yh[df_yh['sid'] == sid]
    if df_yf_sid.shape[0] > 0:
        stable = df_yf_sid['stable'].values[0].replace('%', '')
    else:
        stable = 'NA'

    return stable


def fn_st_chart_bar(df):
    df_sids = fn_get_sids(df)

    for s in ['kpi', 'order', 'order_typ', 'bar']:
        if s not in st.session_state.keys():
            st.session_state[s] = []

    # ==========

    st.markdown(f'#### 📊 觀察 {df_sids.shape[0]} 檔個股')

    cs = st.columns([3, 1, 1, 1])
    kpis = ['績效(%)', '天數'] + [c for c in df_sids.columns if '勝率' in c or '合理' in c or '相關性' in c]

    if True:

        order = '績效(%)'
        v_h = 'h'

        df_sids.sort_values(by=[order], inplace=True, ascending=False, ignore_index=True)
        df_sids.reset_index(inplace=True)

        df_sids['index'] = df_sids['index'].apply(fn_add_digit)
        df_sids['策略選股'] = df_sids['index'] + ' ' + df_sids['名稱'] + ' ' + df_sids['代碼']
        df_sids['策略選股'] = df_sids['策略選股'].apply(lambda x: x + '⭐' if x.split(' ')[1] in dic_sel['pick'] else x)
        # fn_st_add_space(1)

        df_sids = df_sids[df_sids['代碼'] != '6411']
        df_p = df_sids[df_sids['績效(%)'].apply(lambda x: x >= 0)]
        df_n = df_sids[df_sids['績效(%)'].apply(lambda x: x < 0)]

        df_sids = df_sids[[c for c in df_sids.columns if '合理價_' not in c]]
        kpis = [k for k in kpis if '合理價_' not in k]

        fig, watch = fn_kpi_plt(kpis, df_sids)

        tab_w, tab_d, tab_f = st.tabs(['勝率分析', f'指標分布{watch}', f'績效追蹤'])

        with tab_w:

            cols = ['名稱', '代碼', '股價_new',
                    '營收_勝率_new', '營收_合理價差_new', '營收_相關性_new',
                    'EPS_勝率_new', 'EPS_合理價差_new', 'EPS_相關性_new',
                    '殖利率_勝率_new', '殖利率_合理價差_new', '殖利率_相關性_new',
                    '大盤領先指標_new', '產業領先指標_new', '產業別']

            df_show = df_sids[cols]
            df_show.rename(columns={c: c.replace('_new', '') for c in df_show.columns}, inplace=True)
            # df_show.rename(columns={c: c.split('_')[-1]+'_'+c.split('_')[0] if '_' in c else c for c in df_show.columns}, inplace=True)
            # df_show.sort_values(by=['勝率_營收', '勝率_EPS', '勝率_殖利率'], ascending=False, inplace=True, ignore_index=True)
            df_show.sort_values(
                by=['營收_勝率', 'EPS_勝率', '殖利率_勝率', '營收_合理價差', 'EPS_合理價差', '殖利率_合理價差'],
                ascending=[False, False, False, True, True, True],
                inplace=True, ignore_index=True)

            fn_st_add_space(1)

            # st.dataframe(df_show, height=500)

            def fn_color_df(x):
                css = ''
                css_p = 'background-color: pink; color: black'
                css_r = 'background-color: orangered; color: white'
                css_y = 'background-color: lightyellow; color: black'
                css_b = 'background-color: lightblue; color: black'
                css_g = 'background-color: lightgreen; color: black'
                css_gray = 'background-color: lightgray; color: black'
                try:
                    f = float(x)
                except:
                    f = 'NA'

                if '.' in str(x):
                    if 4.9 < f < 10.1:
                        css = css_r

                    if 4.5 < f < 4.91:
                        css = css_p

                    if 3.9 < f < 4.51:
                        css = css_y

                    if 1.0 <= f < 3.91:
                        css = css_g

                    if f < 1.0:
                        css = css_gray
                else:
                    if f == 'NA':
                        css = css_gray
                    else:
                        if f == 100:
                            css = css_r
                        elif f >= 80:
                            css = css_p
                        elif f >= 60:
                            css = css_y
                        else:
                            css = css_g

                return css

            # pd.options.display.float_format = "{:.2f}".format
            df_show['成長'] = df_show['代碼'].apply(fn_get_yh_grow)
            df_show['穩健'] = df_show['代碼'].apply(fn_get_yh_stable)
            df_show = df_show[[c for c in df_show.columns if '領先指標' not in c and '產業別' not in c] + ['產業別']]
            for c in df_show.columns:
                if '_' in c or '股價' in c:
                    df_show[c] = df_show[c].apply(lambda x: format(float(x), ".1f"))

            dic_df['df_win'] = df_show
            df_color = df_show.style.applymap(fn_color_df, subset=[c for c in df_show.columns if
                                                                   '勝率' in c or '成長' in c or '穩健' in c])
            st.dataframe(df_color, height=500)

        with tab_d:
            fn_st_add_space(1)
            cs = st.columns([1, 7, 1])
            cs[1].plotly_chart(fig, use_container_width=True)

        with tab_f:
            fn_st_add_space(1)

            col_f = st.columns([1, 6])

            col_f[0].markdown(f'{dic_mkd["4sp"]}')

            with col_f[0].form(key='Form1'):
                kpi_sel = st.multiselect(f':blue[$選擇觀察指標$]',
                                                         options=kpis,
                                                         default=['績效(%)', '天數'],
                                                         key='kpi_sel')

                kpi_sel = [order] + kpi_sel if order not in kpi_sel else kpi_sel
                fn_st_add_space(1)
                submit = st.form_submit_button('選擇')

            tab_p, tab_n = col_f[-1].tabs(
                [f'正報酬: {df_p.shape[0]}檔',
                 f'負報酬: {df_n.shape[0]}檔'])

            with tab_p:
                fn_st_add_space(1)
                fn_show_bar(df_p, y=kpi_sel, v_h=v_h, tick_color='red')

            with tab_n:
                fn_st_add_space(1)
                fn_show_bar(df_n, y=kpi_sel, v_h=v_h, tick_color='green')


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

    return df_all


def fn_st_reference():
    fn_st_add_space(1)
    st.markdown('### 📚 參考資料:')
    with st.form(key='ref'):
        cols = st.columns([1, 1, 1, 1, 0.1])
        cols[0].markdown('#### :orange[$數據來源$]')
        cols[0].markdown('- [$FindBillion$](https://www.findbillion.com/)')
        cols[0].markdown('- [$財務比較e點通$](https://mopsfin.twse.com.tw)')
        cols[0].markdown('- [$公開資訊觀測站$](https://mops.twse.com.tw)')
        cols[0].markdown('- [$景氣指標及燈號$](https://index.ndc.gov.tw/n/zh_tw/lightscore#/)')
        cols[0].markdown('- [$臺灣證券交易所$](https://www.twse.com.tw/zh/page/trading/exchange/BWIBBU_d.html)')
        cols[0].markdown(
            '- [$證券櫃檯買賣中心$](https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera.php?l=zh-tw)')
        cols[0].markdown('- [$臺灣集中保管結算所$](https://www.tdcc.com.tw/portal/zh/smWeb/qryStock)')

        cols[1].markdown('#### :orange[$基本概念$]')
        cols[1].markdown(
            '- [$下班經濟學-股魚$](https://www.youtube.com/watch?v=ShNI41_rFv4&list=PLySGbWJPNLA8D17qZx0KVkJaXd3qxncGr&index=96&t=1610s&ab_channel=%E9%A2%A8%E5%82%B3%E5%AA%92TheStormMedia)')
        cols[1].markdown(
            '- [$Mr. Market市場先生$](https://rich01.com/learn-stock-all/#%E8%B2%A1%E5%A0%B1%E8%88%87%E8%B2%A1%E5%8B%99%E6%8C%87%E6%A8%99)')
        cols[1].markdown('- [$財經AI與資料科學分析平台$](https://www.youtube.com/@findbillion-ai563)')

        cols[2].markdown('#### :orange[$專業網站$]')
        cols[2].markdown('- [$財報狗$](https://statementdog.com/)')

        cols[3].markdown('#### :orange[$研究報告$]')
        cols[3].markdown('- [$當前經濟情勢簡報$](https://www.ndc.gov.tw/News.aspx?n=8E8FA34452E8DBC2&sms=40C8FF59B01AC562)')

        cols[-1].form_submit_button('')


def fn_show_raw(df_all):
    cols = [c for c in df_all.columns if '策略_' not in c]
    df_all = df_all[cols]
    df_all_show = df_all.style.applymap(fn_color_map,
                                        subset=[c for c in df_all.columns if '勝率' in c] + ['篩選', '名稱'])
    fn_st_add_space(3)
    st.markdown(f'#### 📡 {df_all["代碼"].nunique()}檔 台股的 "勝率" 與 "合理價" 分析:')
    st.dataframe(df_all_show, width=None, height=500)


def fn_book():
    fn_st_add_space(1)
    for b in dic_book_img.keys():
        fn_st_add_space(1)
        cols = st.columns([0.7, 1, 2.5])
        cols[1].image(dic_book_img[b], use_column_width=True)
        cols[1].markdown('---')
        cols[2].markdown(f'《 [${b}$]({dic_book_lnk[b]}) 》')
        cols[2].markdown(dic_book_cmt[b])


@st.cache_data
def fn_read_mops(latest='0322'):
    dic_rename = {
        '證券代號': '股票代號',
        '證券名稱': '名稱',
    }

    df_per = pd.DataFrame()
    for root, dirs, files in os.walk(dic_cfg['per_latest_path']):
        for name in files:
            if latest in name:
                csv = os.path.join(root, name)
                header = 3 if 'pera' in name else 1
                market = '櫃' if 'pera' in name else '市'
                try:
                    df = pd.read_csv(csv, na_filter=False, encoding='ANSI', index_col=None, dtype=str, header=header)
                except:
                    df = pd.read_csv(csv, na_filter=False, encoding='cp950', index_col=None, dtype=str, header=header)
                df['市場別'] = market
                df['File'] = name
                df = df.rename(columns=dic_rename)
                df_per = pd.concat([df, df_per])

    df_per['日期'] = latest
    df_per = df_per[[c for c in df_per.columns if 'Unnamed' not in c and 'File' not in c] + ['File']]
    df_per = df_per[df_per['本益比'].apply(lambda x: str(x) != '' and str(x) != 'N/A' and str(x) != '-')]
    df_per = df_per.sort_values(by=['股票代號'], ignore_index=True)
    # st.write(df_per)
    # dic_mops['per'] = df_per

    df_mops = pd.read_csv('mops.csv', na_filter=False, dtype=str)
    df_roe = pd.read_csv('mops_fin_ROE.csv', na_filter=False, dtype=str)
    df_roa = pd.read_csv('mops_fin_ROA.csv', na_filter=False, dtype=str)
    df_opm = pd.read_csv('mops_fin_Operating_Margin.csv', na_filter=False, dtype=str)
    df_dr = pd.read_csv('mops_fin_Debt_Ratio.csv', na_filter=False, dtype=str)
    df_ocf = pd.read_csv('mops_fin_Cash_Flow.csv', na_filter=False, dtype=str)

    return df_per, df_mops, df_roe, df_roa, df_opm, df_dr, df_ocf


def fn_proj():
    fn_st_add_space(2)
    cols = st.columns([1.4, 3])
    cols[1].markdown('### 🗃️ 其它專案:')
    cols[1].markdown(f'#### 📌 $專案:$ 🏠 [$尋找夢想家$](https://taipei-house-price.streamlit.app/)')
    cols[1].markdown(
        f'#### 📌 $專案:$ 🌏 [$座標查詢行政區$](https://ssp6258-use-conda-env-geopandas-25ytkj.streamlit.app/)')
    cols[1].markdown(f'#### 📌 $專案:$ 🎲 [$離散事件模擬器$](https://ssp6258-des-app-app-qdgbyz.streamlit.app/)')


def fn_wef_global_risk():
    fn_st_add_space(1)
    cols = st.columns([0.7, 2, 0.5])
    cols[1].markdown(
        '#### [$世界經濟論壇$](https://www.weforum.org/) $在2023年1月11日發布了$[:red[$《2023年全球風險報告》$]](https://www.weforum.org/reports/global-risks-report-2023/in-full/1-global-risks-2023-today-s-crisis#1-global-risks-2023-today-s-crisis)')
    fn_st_add_space(1)
    st.image(r'https://tccip.ncdr.nat.gov.tw/upload/ckfinder/images/pic_2_chart1a.png',
             caption='摘自: 臺灣氣候變遷推估資訊與調適知識平台(TCCIP) ， https://tccip.ncdr.nat.gov.tw',
             use_column_width=True)


@st.cache_data
def fn_st_stock_init():
    stock_file = dic_cfg['stock_file']
    tdcc_file = os.path.join('TDCC', 'TDCC_OD_1-5.csv')
    if not os.path.exists(stock_file):
        st.error(f"{stock_file} NOT Exist !!!")
        return

    df_all = pd.read_csv(stock_file, na_filter=False, encoding='utf_8_sig', index_col=0, dtype=str)
    df_tdcc = pd.read_csv(tdcc_file, na_filter=False, encoding='utf_8_sig', index_col=None, dtype=str)
    df_field = pd.read_csv('stock_field.csv', na_filter=False, encoding='utf_8_sig', index_col=0, dtype=str)
    df_rp = pd.read_csv('Company_Report_link.csv', na_filter=False, encoding='utf_8_sig', index_col=None,
                        dtype=str)

    df_month = pd.read_csv('Month.csv', na_filter=False, encoding='utf_8_sig', index_col=0, dtype=str)
    df_season = pd.read_csv('Season.csv', na_filter=False, encoding='utf_8_sig', index_col=0, dtype=str)
    df_yh = pd.read_csv('Yahoo_Health.csv', na_filter=False, encoding='utf_8_sig', index_col=None, dtype=str)
    df_light = pd.read_csv('Light.csv', na_filter=False, encoding='utf_8_sig', index_col=None, dtype=str, skiprows=[1])
    df_light.rename(columns={'Unnamed: 0': 'y-m'}, inplace=True)

    return df_all, df_field, df_rp, df_tdcc, df_month, df_season, df_yh, df_light


def fn_st_stock_main():
    df_all, df_field, dic_df['report'], dic_df['tdcc'], dic_df['month'], dic_df['season'], dic_df['Yahoo_Health'], \
    dic_df['light'] = fn_st_stock_init()

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

    dic_df['stock_all'] = df_all

    cols = st.columns([7, 3.3])
    home = r'https://streamlit.io/'
    ver = r'https://docs.streamlit.io/library/changelog'

    py_ver = python_version()
    lnk_py = r'https://www.python.org/downloads/'
    cols[-1].markdown(
        f'##### $by\ 🐍\ $[:green[$v{py_ver}$]]({lnk_py})$\ with\ $ [:blue[$Streamlit$]]({home}) [:red[$\ v{st.__version__}$]]({ver})')

    cols = st.columns([1.7, 0.7, 0.5, 1.5])
    url = r'https://th.bing.com/th/id/OIP.kiUSNjrStSTNTzPRGLFvzwHaE8?w=286&h=190&c=7&r=0&o=5&dpr=1.4&pid=1.7'
    url = r'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQD77cetO5GgS7c2YGH7ai5ocF_ZGMC64Wdqg&usqp=CAU'
    img_plan = r'Plan.png'
    img_b = r'B.png'

    img = fn_show_img(url)

    img_Plan = fn_show_img(img_plan)
    img_B = fn_show_img(img_b)

    # 👨‍💻  🐰
    # cols[0].title(r'')
    cols[0].title(r'👨‍💻 [$傑克潘$](https://www.facebook.com/jack.pan.96) $的$ :red[${\bf B}$] $計劃$ ')
    # cols[1].image(img)
    cols[1].image(img_Plan)
    cols[2].image(img_B)
    cols[3].write('')
    cols[3].image('use_pc.png')

    df = fn_st_stock_all(df_all)
    df_rcmd = df[df['Recommend'] == '1']

    dic_mops['per'], dic_mops['MOPS'], dic_mops['ROE'], dic_mops['ROA'], dic_mops['OPM'], dic_mops['DR'], dic_mops[
        'OCF'] = fn_read_mops(latest=dic_mops['per_date'])

    tab_trend, tab_index, tab_pick, tab_basic_idx, tab_watch, tab_idea, tab_ref, tab_book, tab_proj = st.tabs(
        ['全球趨勢', '指標分布', '策略選股', '基本指標', '觀察驗證', '設計概念', '參考資料', '閱讀書單', '其它專案'])

    # with tab_life:
    #     fn_life()

    with tab_trend:
        fn_wef_global_risk()

    with tab_idea:
        fn_idea()

    with tab_index:
        fn_st_chart_bar(df_rcmd)
        # fn_show_raw(df)

    with tab_pick:
        fn_pick_stock(df_rcmd, dic_mops['MOPS'])

    with tab_basic_idx:
        fn_st_add_space(1)
        fn_show_basic_idx(df, dic_mops['MOPS'], key='basic_idx')

    with tab_watch:
        fn_st_stock_sel(df_all)

    with tab_ref:
        fn_st_reference()

    with tab_book:
        fn_book()

    with tab_proj:
        fn_proj()


def fn_st_init():
    st.set_page_config(page_title='B計劃', page_icon='🅱️', layout='wide', initial_sidebar_state="auto", menu_items=None)


def fn_main():
    fn_st_init()
    fn_st_stock_main()


if __name__ == '__main__':
    fn_main()
