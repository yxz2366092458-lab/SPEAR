from selenium import webdriver # 导入库
from selenium.common import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
import time

from selenium.webdriver.support.wait import WebDriverWait

#read me
'''
    pip install selenium
    
    
    
    需要下载一个浏览器驱动
    Firefox(火狐)浏览器驱动
    url = https://github.com/mozilla/geckodriver/releases/
    Chrome(google)浏览器驱动
    url = http://chromedriver.storage.googleapis.com/index.html
    Microsoft Edge (EdgeHTML)浏览器驱动
    url = https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/

    将下载的浏览器驱动放在环境变量中或自己创建路径，将创建的路径更新到环境变量里
'''



def scroll_shim(passed_in_driver, object):
        x = object.location['x']
        y = object.location['y']
        scroll_by_coord = 'window.scrollTo(%s,%s);' % (
        x,
        y
        )
        #scroll_nav_out_of_way = 'window.scrollBy(0, -120);'
        passed_in_driver.execute_script(scroll_by_coord)
        #passed_in_driver.execute_script(scroll_nav_out_of_way)



def main():
    #找到登录界面的超链接复制到 url
    url = "https://sso.dlut.edu.cn/cas/login?service=https%3A%2F%2Fportal.dlut.edu.cn%2Ftp%2F" 
    # 账户 密码
    account = "32309079"
    passward = "a//s//d//2000"
    
    #课程名称_老师_班级名称
    cource = '数值最优化方法'
    teacher = '郭方芳'
    num = '4'
    browser = webdriver.Edge() # 声明浏览器
    browser.implicitly_wait(30) # 隐性等待 在规定的时间内，最长等待S秒
    browser.get(url) # 打开设置的网址
    time.sleep(1)
    
    browser.find_element('id','un').send_keys(account)
    browser.find_element('id','pd').send_keys(passward)
    time.sleep(0.5)
    browser.find_element('id','pd').send_keys(Keys.ENTER)
    time.sleep(0.5)
    browser.get('https://dutgs.dlut.edu.cn/pyxx/default.aspx?u=cas')
    
    time.sleep(0.5)
    browser.switch_to.frame(browser.find_element('name','MenuFrame'))
    action = ActionChains(browser)
    
    time.sleep(0.5)
    above = browser.find_element(By.XPATH, '/html/body/div[2]/dl/dt[4]')
    action.move_to_element(above).click().perform()
    
    time.sleep(0.5)
    above = browser.find_element(By.XPATH, '/html/body/div[2]/dl/dd[4]/ul/li[6]/a')
    action.move_to_element(above).click().perform()
    
    time.sleep(0.5)
    browser.switch_to.default_content()
    browser.switch_to.frame(browser.find_element('name','PageFrame'))
    
    tbody = browser.find_element(By.XPATH, '//*[@id="MainWork_dgData"]/tbody')
    a1 = tbody.find_elements(By.XPATH, 'tr')
    i = 0
    index = []
    for a in a1:
        i = i+1
        if((teacher in a.text) and (teacher in a.text) and (num == a.text[0])):
                print(a.text)
                index.append(i)
    if(not index):
        raise Exception("不存在课程，请注意是否是课程名称_老师_班级名称编写错误！！！")
    index = index[0]
    
    
    source_element = browser.find_element(By.XPATH, '/html/body/div[1]/form/div[4]/div[1]/div[4]/div[2]/table/tbody/tr[%d]/td[15]/a'%index)
    if 'firefox' in browser.capabilities['browserName']:
        scroll_shim(browser, source_element)

    while (True):
        # 获取选课按钮
        above = browser.find_element(By.XPATH,
                                     '/html/body/div[1]/form/div[4]/div[1]/div[4]/div[2]/table/tbody/tr[%d]/td[15]/a' % index)

        # 检查按钮文本是否是“退选”
        above_text = above.text
        print(f"按钮文本: {above_text}")

        if '退选' == above_text:
            print('选课成功')
            break

        # 尝试点击选课按钮
        action.move_to_element(above).click().perform()

        try:
            # 等待第一个警告弹窗，超时则捕获异常
            WebDriverWait(browser, 10).until(EC.alert_is_present())
            browser.switch_to.alert.accept()

            # 等待第二个警告弹窗，超时则捕获异常
            WebDriverWait(browser, 10).until(EC.alert_is_present())
            browser.switch_to.alert.accept()

        except TimeoutException:
            # 如果超时没有弹出警告窗口，打印信息并重新点击选课按钮
            print("超时：未检测到警告弹窗，重新尝试点击选课按钮")

        # 日志输出当前时间，方便调试
        print("Start : %s" % time.ctime())
        time.sleep(2)  # 等待 2 秒
        print("End : %s" % time.ctime())
    
 
if __name__ == "__main__":
    main()