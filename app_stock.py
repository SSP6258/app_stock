import pandas as pd
import streamlit as st
from app_stock_fb import *


dic_url = {
    'findbillion': 'https://www.findbillion.com/twstock/',
    'yahoo': 'https://tw.stock.yahoo.com/quote/',
    'cmoney': 'https://www.cmoney.tw/forum/stock/',
}


def fn_add_space(s):
    for _ in range(s):
        st.write('')


def fn_make_clickable(x):
    name = x
    sid = x if str(x).isnumeric() else x.split(" ")[0]
    url = rf'{dic_url["cmoney"]}{sid}'

    return '<a href="{}">{}</a>'.format(url, name)


def fn_click_name(sid, name, url):
    url = rf'{url}{sid}'

    return '<a href="{}">{}</a>'.format(url, name)


def fn_st_init():
    st.set_page_config(page_title='爬蟲練習', page_icon='🕷️', layout='wide', initial_sidebar_state="auto", menu_items=None)
    st.title(f'👨‍💻 傑克潘的爬蟲練習')


def fn_st_show_win_rate():
    stock_file = dic_cfg['stock_file']
    if not os.path.exists(stock_file):
        st.error(f"{stock_file} NOT Exist !!!")
        return

    df_all = pd.read_csv(stock_file, na_filter=False, encoding='utf_8_sig', index_col=0, dtype=str)

    df_all["篩選"] = 0
    for idx in df_all.index:
        for c in df_all.columns:
            if '勝率' in c:
                v = df_all.loc[idx, c]
                if v != '':
                    if int(v) >= dic_cfg["sel_rat"]:
                        df_all.at[idx, "篩選"] = 1
                        break

    df_sel = df_all[df_all["篩選"] == 1]
    df_sel = df_sel[df_sel["股價"].apply(lambda x: float(x) < dic_cfg["sel_price"] if x != '' else True)]
    df_sel = df_sel[[c for c in df_sel.columns if '篩選' not in c and
                     '耗時' not in c and
                     '合理價差' not in c]]
    df_sel.reset_index(drop=True, inplace=True)

    st.markdown(f'### 針對 {df_all["sid"].nunique()}檔推薦股票進行篩選 '
                f'- 任一策略之勝率大於 {dic_cfg["sel_rat"]}% '
                f'- 股價低於 {dic_cfg["sel_price"]}元')

    if df_sel.shape[0] > 0:
        def f(sid, name):
            return sid if name == '' else name

        df_sel['sid_name'] = df_sel.apply(lambda x: f(x.sid, x.sid_name), axis=1)
        df_sel['max'] = df_sel[[c for c in df_sel.columns if '勝率' in c]].max(axis=1)
        df_sel.sort_values(by=['max'], ascending=False, inplace=True, ignore_index=True)

        c1, c2 = st.columns([2.2, 1])
        c1.error(f'#### 👉 篩選出{df_sel["sid"].nunique()}檔: {", ".join(list(df_sel["sid_name"].unique()))}')
        fn_add_space(1)

        df_sel = df_sel[[c for c in df_sel.columns if 'max' not in c]]
        df_show = df_sel.copy()
        df_show.sort_values(by=['sid_name', 'date'], inplace=True, ignore_index=True)
        df_show = df_show[['date'] + [c for c in df_show.columns if c != 'date']]

        dic_page = {
            '營收': '/revenue',
            'EPS': '/eps',
            '殖利率': '/cash_dividend',
        }

        for c in df_show.columns:
            if '勝率' in c:
                df_show[c] = df_show[c].apply(lambda x: '不適用' if x == '' else str(x)+'%')
                page = dic_page[c.split('_')[-1]]
                df_show[c] = df_show.apply(lambda x: fn_click_name(x['sid']+page, x[c], dic_url['findbillion']), axis=1)
            if '相關性' in c:
                df_show[c] = df_show[c].apply(lambda x: x.split(' ')[-1])

        df_show['股票代碼'] = df_show['sid'].apply(fn_make_clickable)
        df_show['股票名稱'] = df_show.apply(lambda x: fn_click_name(x["sid"], x["sid_name"], dic_url['yahoo']), axis=1)

        show_cols_order = ['股票名稱', '股票代碼', 'date', '股價', '大盤領先指標', '產業領先指標',
                           '勝率(%)_營收', '相關性_營收', '勝率(%)_EPS', '相關性_EPS',
                           '勝率(%)_殖利率', '相關性_殖利率']

        df_show = df_show[[c for c in show_cols_order if c in df_show.columns]]
        # ➡
        show_cols_renmae = {'date': '日期',
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

        df_show.rename(columns=show_cols_renmae, inplace=True)
        fn_add_space(1)
        st.write(df_show.to_html(escape=False, index=True), unsafe_allow_html=True)

    fn_add_space(3)
    st.markdown(f'### {df_all["sid"].nunique()}檔 台股的 "勝率" 與 "合理價" 分析:')
    df_all = df_all[[c for c in df_all.columns if '耗時' not in c]]
    st.dataframe(df_all, width=1200)


def fn_main():
    if fn_is_parsing():
        try:
            df = fn_fb_recommend_stock()
            fn_find_billion(df, dic_cfg["stocks"])
        except:
            pass

    fn_st_init()
    fn_st_show_win_rate()


if __name__ == '__main__':
    fn_main()
