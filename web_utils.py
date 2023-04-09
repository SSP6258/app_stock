import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options


def fn_web_click(drv, val, slp=2, by=By.XPATH):
    time.sleep(slp)
    elm = drv.find_element(by=by, value=val)
    elm.click()
    time.sleep(slp)


def fn_web_move_to(drv, act, val, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    act.move_to_element(elm).perform()
    time.sleep(slp)


def fn_web_clear_txt(drv, val, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    elm.send_keys(Keys.CONTROL, 'a')
    elm.send_keys(Keys.DELETE)
    time.sleep(slp)


def fn_web_send_keys(drv, val, key, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    elm.send_keys(key)
    time.sleep(slp)


def fn_web_get_text(drv, val, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    txt = elm.text
    time.sleep(slp)
    return txt


def fn_web_switch(drv, val, slp=2, by=By.XPATH):
    elm = drv.find_element(by=by, value=val)
    drv.switch_to.frame(elm)
    time.sleep(slp)


def fn_web_sel_val(drv, val, slp=2, by=By.ID, sel_txt=''):
    s = drv.find_element(by=by, value=val)
    Select(s).select_by_visible_text(sel_txt)
    time.sleep(slp)


def fn_web_handle(drv, act, typ, slp, by, val, key=''):
    if typ == 'click':
        fn_web_click(drv, val, slp=slp, by=by)
    elif typ == 'move2':
        fn_web_move_to(drv, act, val, slp=slp, by=by)
    elif typ == 'keyin':
        fn_web_send_keys(drv, val, key, slp=slp, by=by)
    elif typ == 'getText':
        text = fn_web_get_text(drv, val, slp=slp, by=by)
        return text

    elif typ == 'clear_txt':
        fn_web_clear_txt(drv, val, slp=slp, by=by)

    elif typ == 'sel_val':
        fn_web_sel_val(drv, val, slp=slp, by=by, sel_txt=key)

    elif typ == 'iframe_switch':
        fn_web_switch(drv, val, slp=slp, by=by)
    else:
        assert False, f'Invalid web handle typ = {typ}'

    return 'NA'


def fn_web_get_hyperlink(drv, val="a", k1="M00", slp=2):
    fmts = [".pdf", "PDF", ".ppt", ".PPT"]
    lnks = drv.find_elements(by=By.TAG_NAME, value=val)
    urls = []
    for lnk in lnks:
        url = lnk.get_attribute("href")
        url_str = str(url)
        if k1 in url_str:
            for f in fmts:
                if f in url_str:
                    urls.append(url)
                    break

    time.sleep(slp)
    return urls


def fn_web_init(link, is_headless=True):
    """
    要注意用selenium進行爬蟲的時候，
    chrome 有時候會出現「自動軟體正在控制您的瀏覽器」，然後程式可能會跑不動。
    https://ithelp.ithome.com.tw/m/articles/10267172
    """
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_experimental_option("prefs",
                                    {"profile.password_manager_enabled": False, "credentials_enable_service": False})

    if is_headless:
        options.add_argument('--headless')

    driver = webdriver.Chrome(options=options)

    try:
        driver.implicitly_wait(5)
        driver.get(link)
        action = ActionChains(driver)
    except:
        driver.implicitly_wait(15)
        driver.get(link)
        action = ActionChains(driver)

    return driver, action

