#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib
from matplotlib import font_manager, rc
import platform

try :
    if platform.system() == 'windows':
        # windows의 경우
        font_name = font_manager.FomntProperties(fname="c:/Windows/Font")
        rc('font', family = font_name)
    else:
        # mac의 경우
        rc('font', family = 'AppleGothic')
except :
    pass

matplotlib.rcParams['axes.unicode_minus'] = False


# #### api-football 데이터 수집

# In[3]:


from pandas.io.json import json_normalize

n = 0

headers= {
        'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
        'x-rapidapi-key': "aeda90b38bmsh068bc2e3a0bb552p19cf54jsnea18b745c09e"
}

payload = {}
data = pd.DataFrame()

for i in range(1, 100+1):
    response = requests.request("GET","https://api-football-v1.p.rapidapi.com/v2/players/player/{}".format(i),                                 headers=headers, data=payload)
    json_object = response.json()
    df = json_normalize(json_object['api']['players'])
    data = data.append(df, ignore_index = True)

    i += 1
# data['Name'] = data[['firstname','lastname']].apply(lambda x: ' '.join(x), axis=1)
data


# In[3]:


data.to_csv('apidata_147824_150000.csv')


# In[7]:


df = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/회귀분석 프로젝트/데이터/api-football(원본)/apidata_147824_150000.csv')
is_season = df['season'] == '2019-2020'
is_season2 = df['season'] == '2018-2019'
is_season3 = df['season'] == '2017-2018'

df_new = df[is_season | is_season2 | is_season3]

df_new.to_csv('apidata_147824_150000.csv')


# #### 트랜스퍼마켓 데이터 크롤링 (Data Crawling from transfer-market)

# In[4]:


get_ipython().system('pip install html_table_parser')


# In[7]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

dataFrame= pd.DataFrame(columns=['Name', 'Values'])

for i in range(1,20+1):
    url = 'https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop?ajax=yw1&page=' + str(i)
    
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    chrome_driver = '/Users/wglee/Desktop/DATA ANALYSIS/Chromedriver'
    driver = webdriver.Chrome(chrome_driver, options=options)
    driver.implicitly_wait(3)
    driver.get(url)

    src = driver.page_source

    driver.close()

    resp = BeautifulSoup(src, "html.parser")
    values_data = resp.select('table')
    table_html = str(values_data)
    num = 0
    name = ' '
    value = ' '
    for index, row in pd.read_html(table_html)[1].iterrows():
        if index%3 == 0:
            num = row['#']
            value = row['Market value']
        elif index%3 == 1:
            name = row['Player']
        else : 
            dataFrame.loc[num] = [name, value]
dataFrame


# #### 인스타그램 팔로워 크롤링

# In[ ]:


userList = dataFrame['Name'].tolist()
ul = userList[301:401]
ul

# with the list obtained from above, search instagram to get followers of players
from selenium import webdriver
import re
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import ElementNotInteractableException
from tqdm import tqdm

listUser = []
listFollower = []

def checkInstaFollowers(user):

    try:        
        driver.find_element_by_xpath('//*[@id="react-root"]/section/nav/div[2]/div/div/div[2]/input').send_keys(user)
        time.sleep(4)
        driver.find_element_by_xpath('//*[@id="react-root"]/section/nav/div[2]/div/div/div[2]/div[2]/div[2]/div/a[1]/div').click()

        r = requests.get(driver.current_url).text
        followers = re.search('"edge_followed_by":{"count":([0-9]+)}',r).group(1)

    except AttributeError:
        print("{}'s top search is returned as hashtag.".format(user))
        try:
            checkInstaFollowers(user)
        finally:
            listUser.append(user)
            listFollower.append('Hashtag')
    except StaleElementReferenceException:
        print("{} called StaleElementReferenceException".format(user))
        try:
            checkInstaFollowers(user)
        except AttributeError:
            listUser.append(user)
            listFollower.append('SERE/Hashtag')
    except NoSuchElementException:
        print("{} called NoSuchElementException".format(user))
        try:
            checkInstaFollowers(user)
        except AttributeError:
            listUser.append(user)
            listFollower.append('NSEE/Hashtag')
    except ElementNotInteractableException:
        print("{} called ElementNotInteractableException".format(user))
        try:
            checkInstaFollowers(user)
        except AttributeError:
            listUser.append(user)
            listFollower.append('ENIE/Hashtag')
    
    else:
        if (r.find('"is_verified":true')!=-1):
    #        print('{} : {}'.format(user, followers))
            listUser.append(user)
            listFollower.append(followers)
        else:
    #        print('{} : user not verified'.format(user))
            listUser.append(user)
            listFollower.append('not verified')
            
#    finally:
#        driver.quit()
        
        
for a in tqdm(range(int((len(ul)/10)))):
    
    driver = webdriver.Chrome('/Users/wglee/Desktop/DATA ANALYSIS/chromedriver')
    driver.get('https://www.instagram.com/')
    delay = 3
    driver.implicitly_wait(delay)

    id = 'bhcboy100@naver.com' #Instagram ID
    pw = 'lwglwk5120!' #Instagram PW

    driver.find_element_by_xpath('//*[@id="react-root"]/section/main/article/div[2]/div[1]/div/form/div[2]/div/label/input').send_keys(id)
    driver.find_element_by_xpath('//*[@id="react-root"]/section/main/article/div[2]/div[1]/div/form/div[3]/div/label/input').send_keys(pw)
    driver.find_element_by_xpath('//*[@id="react-root"]/section/main/article/div[2]/div[1]/div/form/div[4]/button').click()

    driver.implicitly_wait(delay)
    
    for b in range(10):
#        print('(a*10)+b = {}, a={}, b={}'.format(((a*10) + b), a, b))
        num = (a*10) + b
        userName = ul[num]
        print(userName)
        checkInstaFollowers(userName)
#    print('==============================================')
    driver.quit()
    
df_follower = pd.DataFrame(list(zip(listUser, listFollower)), columns=['name', 'follower'])
df_follower.to_csv('follower_301_400.csv', encoding='utf-8-sig')


#  #### 데이터베이스 > Pandas

# In[328]:


from sqlalchemy import create_engine
import pymysql

db_connection_str = 'mysql+pymysql://root:Lwglwk5120!@54.180.4.238/Linear_Regression'
db_connection = create_engine(db_connection_str)

df_original = pd.read_sql('SELECT * FROM api_football', con=db_connection)
df_original.tail(3)


# In[329]:


db_connection_str = 'mysql+pymysql://root:Lwglwk5120!@54.180.4.238/Linear_Regression'
db_connection = create_engine(db_connection_str)

df_market = pd.read_sql('SELECT * FROM market_instagram', con=db_connection)
df_market['value'] = df_market['value'].str.replace(pat=r'[â‚¬@m\r]', repl = r' ', regex=True).astype(np.float)
df_market.tail(3)


# ### domain based OLS

# ###### 데이터 전처리 및 코딩

# * feature selection with domain knowledge

# In[330]:


pd.options.display.max_columns = len(df_original)


# In[331]:


df_original = df_original[df_original.position != 'Goalkeeper']
# df_original = df_original[df_original.position == 'Attacker']


# In[332]:


df_personal_info = df_original[['player_name','position','age','nationality','height','weight','rating']].groupby('player_name').mean().reset_index()

df_original = df_original.groupby('player_name').sum()
df_original.drop(['captain','goals_conceded','penalty_saved','age','height','weight','rating'], axis=1, inplace=True)
df_original


# In[333]:


df_original = pd.merge(df_original, df_market, on='player_name', how='inner').set_index('player_name')
df_original['games_played'] = round(df_original['games_minutes_played'] / 90,2)


# In[334]:


df_copy = df_original.copy()
df_copy


# In[335]:


games_played =pd.DataFrame(df_copy['games_played'])
pi = round(df_copy[['shots_total', 'shots_on', 'goals_total', 'goals_assists', 'passes_total', 'passes_key',                     'passes_accuracy','tackles_total', 'tackles_blocks', 'tackles_interceptions','duels_total', 'duels_won', 'dribbles_attempts',                     'dribbles_success','fouls_drawn', 'fouls_committed', 'cards_yellow', 'cards_yellowred','cards_red', 'penalty_won',                     'penalty_commited', 'penalty_success','penalty_missed', 'games_appearences','games_lineups',                     'substitutes_in','substitutes_out', 'substitutes_bench']].div(df_copy['games_played'], axis=0),4)

df_copy = pd.concat([df_copy[['value','follower']], pi], axis=1)
df = pd.concat([df_copy, games_played], axis=1).reset_index()
df_copy


# In[336]:


df_copy = pd.merge(df_copy,df_personal_info, on='player_name', how='inner').set_index('player_name')
df_copy


# #### 결측치 확인

# In[337]:


import missingno as msno
msno.matrix(df_copy)


# In[338]:


df_copy.isnull().sum()
df_copy = df_copy.dropna()


# #### VIF 값 확인

# In[339]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.options.display.float_format = '{:.2f}'.format
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(df_copy.values, i) for i in range(df_copy.shape[1])]
vif["features"] = df_copy.columns
vif.sort_values(by='VIF Factor', ascending=False)


# #### 독립변수들간 상관관계 확인

# In[341]:


plt.figure(figsize = (6,6))
sns.heatmap(data = df_copy.corr(), annot=False, fmt = '.2f', linewidths=.5, cmap='Blues')

df_copy.corr()[df_copy.corr() > 0.7]


# In[369]:


N = len(df_copy)
ratio = 0.8
np.random.seed(0)
idx_train = np.random.choice(np.arange(N), np.int(ratio * N))
idx_test = list(set(np.arange(N)).difference(idx_train))

df_train = df_copy.iloc[idx_train]
df_test = df_copy.iloc[idx_test]


# In[370]:


# feature_names = list(df_train.columns)
# feature_names = ["scale({})".format(name) for name in feature_names]

# formula = "value ~ " + "+".join(feature_names)


# In[371]:


import statsmodels.api as sm
model = sm.OLS.from_formula('value ~ scale(follower)+scale(goals_total)+scale(I(goals_total**2))+scale(goals_assists)+scale(duels_won)+scale(dribbles_success)', data=df_train)

result = model.fit()
print(result.summary())

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
print('----------------------------------------')

from sklearn.model_selection import KFold
scores = np.zeros(5)
cv = KFold(5, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_copy)):
    df_train = df_copy.iloc[idx_train]
    df_test = df_copy.iloc[idx_test]
       
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
#     print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print('검증모델성능:',scores.mean())


# In[362]:


model_full = sm.OLS.from_formula('value ~ scale(follower)+scale(goals_total)+scale(I(goals_total**2))+scale(goals_assists)+scale(duels_won)+scale(dribbles_success)' , data=df_copy)

model_reduced = sm.OLS.from_formula('value ~ scale(goals_total)+scale(I(goals_total**2))+scale(goals_assists)+scale(duels_won)+scale(dribbles_success)' , data=df_copy)
sm.stats.anova_lm(model_reduced.fit(), model_full.fit())


# In[363]:


model_boston = sm.OLS.from_formula(
    'value ~ scale(follower)+\
scale(goals_total)+scale(I(goals_total**2))+scale(goals_assists)+scale(duels_won)+\
scale(dribbles_success)' , data=df_copy)
result_boston = model_boston.fit()
sm.stats.anova_lm(result_boston, typ=2)


# In[121]:


pred = result.predict(df_test)

rss = ((df_test.value - pred) ** 2).sum()
tss = ((df_test.value - df_test.value.mean()) ** 2).sum()
rsquared = 1 - rss / tss
round(rsquared,2)


# ### pca based OLS

# In[2]:


df_ols = pd.read_csv(r'C:\Users\Gk\Documents\dev\data\LinearRegression_Football_data\dataset_20200629.csv', encoding='utf-8-sig')
df_ols


# In[3]:


df_ols = df_ols.drop(['player_name', 'age_y', 'height_y', 'weight_y', 'rating_y'], axis=1)
df_ols


# In[4]:


df_ols.corr()[df_ols.corr() > 0.7]


# # 결측치 제거

# In[5]:


df_ols = df_ols.dropna()


# # 높은 상관관계를 보이는 Feature들
# 1. shots_total, shots_on, goals_total
# 2. goals_assists, passes_key
# 3. passes_accuracy, games_appearences, substitutes_in
# 4. duels_total, duels_won
# 5. dribbles_attempts, dribbles_success

# In[6]:


df_pos = df_ols
df_pos


# In[7]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# In[8]:


# 1. shots_on, shots_total and goals_total PCA
df_pos_sotgt = df_pos[['shots_on', 'shots_total', 'goals_total']]
df_pos_sotgt = StandardScaler().fit_transform(df_pos_sotgt)
df_pos_pca_sg = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_sotgt), columns=['shotsOnTotal_goalsTotal'])
df_pos_pca_sg['shotsOnTotal_goalsTotal'] = MinMaxScaler().fit_transform(df_pos_pca_sg)


# In[9]:


# 2. goals_assists, passes_key
df_pos_gapk = StandardScaler().fit_transform(df_pos[['goals_assists', 'passes_key']])
df_pos_pca_gapk = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_gapk), columns=['goalsAssist_passesKey'])
df_pos_pca_gapk['goalsAssist_passesKey'] = MinMaxScaler().fit_transform(df_pos_pca_gapk)


# In[10]:


# 3. passes_accuracy, games_appearences, substitutes_in
df_pos_pagasi = StandardScaler().fit_transform(df_pos[['passes_accuracy', 'games_appearences', 'substitutes_in']])
df_pos_pca_pagasi = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_pagasi), columns=['passesAcc_gamesApp_subIn'])
df_pos_pca_pagasi['passesAcc_gamesApp_subIn'] = MinMaxScaler().fit_transform(df_pos_pca_pagasi)


# In[11]:


# 4. duels_total, duels_won
df_pos_duels = df_pos[['duels_total', 'duels_won']]
df_pos_duels = StandardScaler().fit_transform(df_pos_duels)
df_pos_pca_duels = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_duels), columns=['duelsWonTotal'])
df_pos_pca_duels['duelsWonTotal'] = MinMaxScaler().fit_transform(df_pos_pca_duels)


# In[12]:


# 5. dribbles_attempts, dribbles_success
df_pos_dribbles = df_pos[['dribbles_attempts', 'dribbles_success']]
df_pos_dribbles = StandardScaler().fit_transform(df_pos_dribbles)
df_pos_pca_dribbles = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_dribbles), columns=['dribblesAtmptsSuc'])
df_pos_pca_dribbles['dribblesAtmptsSuc'] = MinMaxScaler().fit_transform(df_pos_pca_dribbles)


# # PCA Feature Table

# In[13]:


df_pca = pd.concat([df_pos_pca_sg, df_pos_pca_gapk, df_pos_pca_pagasi, 
                    df_pos_pca_duels, df_pos_pca_dribbles], axis=1)
df_pca


# In[14]:


df_pca.corr()[df_pca.corr() > 0.7]


# # PCA feature들과 그 외 feature들의 OLS 확인

# In[15]:


pca_cols = ['shots_total', 'shots_on', 'goals_total', 'goals_assists', 'passes_key', 'passes_accuracy', 'games_appearences', 'substitutes_in', 'duels_total', 'duels_won', 'dribbles_attempts', 'dribbles_success']
npca_cols = df_pos.columns.tolist()
npca_features = [item for item in npca_cols if item not in pca_cols]


# In[16]:


df_ols = pd.concat([df_pos[npca_features].reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)
df_ols_f = df_ols
df_ols_f


# # Follower feature 제거

# In[17]:


df_ols_nf = df_ols.drop('follower', axis=1)


# # OLS - Basic Model

# In[18]:


from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula = "value ~ " + "+".join(feature_names)

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())

##############################################################################
from sklearn.model_selection import KFold

scores = np.zeros(5)
cv = KFold(5, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
#    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores.mean()))


# # OLS - 큰 P값 Feature 제거

# In[22]:


df_ols_nf_1 = df_ols.drop(['goalsAssist_passesKey', 'tackles_blocks', 'penalty_missed', 'penalty_success', 
                              'passesAcc_gamesApp_subIn', 'tackles_interceptions', 'tackles_total', 'substitutes_bench',
                             'duelsWonTotal', 'height_x', 'cards_yellow', 'fouls_drawn', 'rating_x', 'penalty_commited',
                             'cards_yellowred', 'penalty_won', 'weight_x', ], axis=1)
len(df_ols_nf_1.columns)


# In[23]:


df_ols_nf_1


# In[24]:


from sklearn.model_selection import train_test_split

df_ols = df_ols_nf_1
dfX = df_ols.drop(['value', 'follower'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula = "value ~ " + "+".join(feature_names)

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())

##############################################################################
from sklearn.model_selection import KFold

scores = np.zeros(5)
cv = KFold(5, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
    #print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores.mean()))


# # 다항기저모형

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula = "value ~ scale(passes_total) +                 scale(fouls_committed) +                 scale(games_lineups) +                 scale(substitutes_out) +                 scale(age_x) +                 scale(shotsOnTotal_goalsTotal) +                 scale(dribblesAtmptsSuc)"

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())

##############################################################################
from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

scores_nf = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rsquared = r2_score(df_test.value, pred)
    
    
#     pred = result.predict(df_test)
#     rss = ((df_test.value - pred) ** 2).sum()
#     tss = ((df_test.value - df_test.value.mean())** 2).sum()
#     rsquared = 1 - rss / tss
    
    scores_nf[i] = rsquared
#     print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores_nf.mean()))


# In[57]:


from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula = "value ~ scale(passes_total) +                 scale(fouls_committed) +                 scale(games_lineups) +                 scale(substitutes_out) +                 scale(age_x) +                 scale(shotsOnTotal_goalsTotal) +                 scale(dribblesAtmptsSuc) +                 scale(follower)"

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())

##############################################################################
from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

scores_nf = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rsquared = r2_score(df_test.value, pred)
    
    
#     pred = result.predict(df_test)
#     rss = ((df_test.value - pred) ** 2).sum()
#     tss = ((df_test.value - df_test.value.mean())** 2).sum()
#     rsquared = 1 - rss / tss
    
    scores_nf[i] = rsquared
#     print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores_nf.mean()))


# # MSE 확인 결과

# In[42]:


data = [scores_f, scores_nf]
plt.boxplot(data)
plt.show()


# In[ ]:





# # Feature들 대상 ANOVA 확인

# In[27]:


model_full = sm.OLS.from_formula(
    "value ~ scale(passes_total) + \
                scale(fouls_committed) + \
                scale(cards_red) + \
                scale(games_lineups) + \
                scale(substitutes_out) + \
                scale(age_x) + \
                scale(shotsOnTotal_goalsTotal) + \
                scale(dribblesAtmptsSuc) + \
                scale(follower)", data=df_ols)
model_reduced = sm.OLS.from_formula(
    "value ~ scale(passes_total) + \
                scale(fouls_committed) + \
                scale(cards_red) + \
                scale(games_lineups) + \
                scale(substitutes_out) + \
                scale(age_x) + \
                scale(shotsOnTotal_goalsTotal) + \
                scale(dribblesAtmptsSuc)", data=df_ols)

sm.stats.anova_lm(model_reduced.fit(), model_full.fit())


# In[27]:


model_full = sm.OLS.from_formula(
    "value ~ scale(passes_total) + \
                scale(fouls_committed) + \
                scale(cards_red) + \
                scale(games_lineups) + \
                scale(substitutes_out) + \
                scale(age_x) + \
                scale(shotsOnTotal_goalsTotal) + \
                scale(dribblesAtmptsSuc) + \
                scale(follower)", data=df_ols)

result = model_full.fit()
sm.stats.anova_lm(result)


# In[26]:


print("TSS = ", result.uncentered_tss)
print("ESS = ", result.mse_model)
print("RSS = ", result.ssr)
print("ESS + RSS = ", result.mse_model + result.ssr)
print("R squared = ", result.rsquared)


# In[175]:


from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula = "value ~ scale(goals_total) +                 scale(goals_assists) +                 scale(passes_total) +                 scale(passes_accuracy) +                 scale(games_appearences) +                 scale(age_x)"

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())

##############################################################################
from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

scores_nf = np.zeros(5)
cv = KFold(5, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rsquared = r2_score(df_test.value, pred)
    
    
#     pred = result.predict(df_test)
#     rss = ((df_test.value - pred) ** 2).sum()
#     tss = ((df_test.value - df_test.value.mean())** 2).sum()
#     rsquared = 1 - rss / tss
    
    scores_nf[i] = rsquared
    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores_nf.mean()))


# In[176]:


from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula = "value ~ scale(goals_total) +                 scale(goals_assists) +                 scale(passes_total) +                 scale(passes_accuracy) +                 scale(games_appearences) +                 scale(age_x) +                 scale(follower)"

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())

##############################################################################
from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

scores_nf = np.zeros(5)
cv = KFold(5, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rsquared = r2_score(df_test.value, pred)
    
    
#     pred = result.predict(df_test)
#     rss = ((df_test.value - pred) ** 2).sum()
#     tss = ((df_test.value - df_test.value.mean())** 2).sum()
#     rsquared = 1 - rss / tss
    
    scores_nf[i] = rsquared
    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores_nf.mean()))


# # 공격수 데이터 분석 - Follower 제거 전

# # 작업 데이터 받기

# In[2]:


df_pos = pd.read_csv(r'C:\Users\Gk\Documents\dev\data\LinearRegression_Football_data\df_pos.csv', encoding='utf-8-sig', index_col=0)


# # Position Rounding

# In[3]:


df_pos.position = df_pos.position.round()


# In[4]:


df_pos.position.unique()


# In[5]:


df_atk = df_pos[df_pos.position == 4]


# In[6]:


df_atk.reset_index(drop=True)


# In[ ]:





# # 상관관계 확인

# In[7]:


df_atk.corr()[df_atk.corr() > 0.7].to_csv('df_atk_corr.csv', encoding='utf-8-sig')
df_atk.corr()[df_atk.corr() > 0.7]


# In[ ]:





# # 높은 상관관계를 보이는 feature들
# 1. height, weight
# 2. shots_total, shots_on, goals_total
# 3. passes_key, passes_total, goals_assists
# 4. duels_total, duels_won
# 5. dribbles_attempts, dribbles_success
# 6. games_appearences, substitutes_in, substitutes_bench

# In[8]:


df_pos = df_atk


# In[9]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# In[10]:


# 1. height, weight PCA
df_pos_hw = df_pos[['height', 'weight']]
df_pos_hw = StandardScaler().fit_transform(df_pos_hw)
df_pos_pca_hw = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_hw), columns=['hw'])
df_pos_pca_hw['hw'] = MinMaxScaler().fit_transform(df_pos_pca_hw)


# In[11]:


# 2. shots_on, shots_total and goals_total PCA
df_pos_sotgt = df_pos[['shots_on', 'shots_total', 'goals_total']]
df_pos_sotgt = StandardScaler().fit_transform(df_pos_sotgt)
df_pos_pca_sg = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_sotgt), columns=['shotsOnTotal_goalsTotal'])
df_pos_pca_sg['shotsOnTotal_goalsTotal'] = MinMaxScaler().fit_transform(df_pos_pca_sg)


# In[12]:


# 3. passes_key, passes_total, goals_assists PCA
df_pos_pktga = df_pos[['passes_key', 'passes_total', 'goals_assists']]
df_pos_pktga = StandardScaler().fit_transform(df_pos_pktga)
df_pos_pca_pktga = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_pktga), columns=['passesKeyTotal_goalsAssists'])
df_pos_pca_pktga['passesKeyTotal_goalsAssists'] = MinMaxScaler().fit_transform(df_pos_pca_pktga)


# In[13]:


# 4. duels_total, duels_won PCA
df_pos_duels = df_pos[['duels_total', 'duels_won']]
df_pos_duels = StandardScaler().fit_transform(df_pos_duels)
df_pos_pca_duels = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_duels), columns=['duelsWonTotal'])
df_pos_pca_duels['duelsWonTotal'] = MinMaxScaler().fit_transform(df_pos_pca_duels)


# In[14]:


# 5. dribbles_attempts, dribbles_success PCA
df_pos_dribbles = df_pos[['dribbles_attempts', 'dribbles_success']]
df_pos_dribbles = StandardScaler().fit_transform(df_pos_dribbles)
df_pos_pca_dribbles = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_dribbles), columns=['dribblesAtmptsSuc'])
df_pos_pca_dribbles['dribblesAtmptsSuc'] = MinMaxScaler().fit_transform(df_pos_pca_dribbles)


# In[15]:


# 6. games_appearences, substitutes_in, substitutes_bench PCA
df_pos_gasub = df_pos[['games_appearences', 'substitutes_in', 'substitutes_bench']]
df_pos_gasub = StandardScaler().fit_transform(df_pos_gasub)
df_pos_pca_gasub = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_gasub), columns=['gamesAppearance_sub'])
df_pos_pca_gasub['gamesAppearance_sub'] = MinMaxScaler().fit_transform(df_pos_pca_gasub)


# In[ ]:





# # PCA Feature Table

# In[16]:


df_pca = pd.concat([df_pos_pca_hw, df_pos_pca_sg, df_pos_pca_pktga, df_pos_pca_duels, 
                    df_pos_pca_dribbles, df_pos_pca_gasub], axis=1)
df_pca


# In[17]:


df_pca.corr()[df_pca.corr() > 0.7]


# In[ ]:





# # PCA feature들과 그 외 feature들의 OLS 확인

# In[18]:


pca_cols = ['height', 'weight', 'shots_total', 'shots_on', 'goals_total', 'passes_key', 'passes_total', 'goals_assists', 'duels_total', 'duels_won', 'dribbles_attempts', 'dribbles_success', 'games_appearences', 'substitutes_in', 'substitutes_bench']
npca_cols = df_pos.columns.tolist()
npca_features = [item for item in npca_cols if item not in pca_cols]


# In[19]:


df_ols = pd.concat([df_pos[npca_features].reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)
df_ols = df_ols.drop('player_name', axis=1)


# # Drop 0 array - goals_conceded, penalty_saved

# In[20]:


df_ols = df_ols.drop(['goals_conceded', 'penalty_saved', 'position'], axis=1)


# In[21]:


df_ols_nf = df_ols


# # OLS - Basis Model

# In[21]:


from sklearn.model_selection import train_test_split

dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

#formula = "value ~ " + "+".join(feature_names)

model = sm.OLS.from_formula("value ~ " + "+".join(feature_names), data=df_train)
result = model.fit()
print(result.summary())

##############################################################################
from sklearn.model_selection import KFold

scores = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula("value ~ " + "+".join(feature_names), data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
#    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores.mean()))


# In[ ]:





# # Follower Data 없이 OLS 진행

# In[92]:


from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula = "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) +             scale(passes_accuracy) +             scale(games_played) +             scale(shotsOnTotal_goalsTotal) +             scale(gamesAppearance_sub)"

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())


##############################################################################
from sklearn.model_selection import KFold

scores = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores.mean()))


# # Follower Data 포함 OLS 진행

# In[89]:


from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula =  "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) +             scale(passes_accuracy) +             scale(games_played) +             scale(shotsOnTotal_goalsTotal) +             scale(gamesAppearance_sub) +             scale(follower)"

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())


##############################################################################
from sklearn.model_selection import KFold

scores = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores.mean()))


# # Feature들에 대해서 ANOVA 확인

# In[63]:


model_full = sm.OLS.from_formula(
    "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) + \
            scale(passes_accuracy) + \
            scale(games_played) + \
            scale(shotsOnTotal_goalsTotal) + \
            scale(gamesAppearance_sub) + \
            scale(follower)", data=df_ols)
model_reduced = sm.OLS.from_formula(
    "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) + \
            scale(passes_accuracy) + \
            scale(games_played) + \
            scale(shotsOnTotal_goalsTotal) + \
            scale(gamesAppearance_sub)", data=df_ols)

sm.stats.anova_lm(model_reduced.fit(), model_full.fit())


# In[64]:


model_full = sm.OLS.from_formula(
    "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) + \
            scale(passes_accuracy) + \
            scale(games_played) + \
            scale(shotsOnTotal_goalsTotal) + \
            scale(gamesAppearance_sub) + \
            scale(follower)", data=df_ols)

result = model_full.fit()
sm.stats.anova_lm(result, typ=2)


# In[ ]:





# # gamesAppearance_sub 제거 후

# In[84]:


model_full = sm.OLS.from_formula(
    "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) + \
            scale(passes_accuracy) + \
            scale(games_played) + \
            scale(shotsOnTotal_goalsTotal) + \
            scale(follower)", data=df_ols)

result = model_full.fit()
sm.stats.anova_lm(result, typ=2)


# In[ ]:





# In[ ]:


# 공격수 데이터 분석 - Follower 제거 전

# 작업 데이터 받기

df_pos = pd.read_csv(r'C:\Users\Gk\Documents\dev\data\LinearRegression_Football_data\df_pos.csv', encoding='utf-8-sig', index_col=0)

# Position Rounding

df_pos.position = df_pos.position.round()

df_pos.position.unique()

df_atk = df_pos[df_pos.position == 4]

df_atk.reset_index(drop=True)



# 상관관계 확인

df_atk.corr()[df_atk.corr() > 0.7].to_csv('df_atk_corr.csv', encoding='utf-8-sig')
df_atk.corr()[df_atk.corr() > 0.7]



# 높은 상관관계를 보이는 feature들
1. height, weight
2. shots_total, shots_on, goals_total
3. passes_key, passes_total, goals_assists
4. duels_total, duels_won
5. dribbles_attempts, dribbles_success
6. games_appearences, substitutes_in, substitutes_bench

df_pos = df_atk

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# 1. height, weight PCA
df_pos_hw = df_pos[['height', 'weight']]
df_pos_hw = StandardScaler().fit_transform(df_pos_hw)
df_pos_pca_hw = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_hw), columns=['hw'])
df_pos_pca_hw['hw'] = MinMaxScaler().fit_transform(df_pos_pca_hw)

# 2. shots_on, shots_total and goals_total PCA
df_pos_sotgt = df_pos[['shots_on', 'shots_total', 'goals_total']]
df_pos_sotgt = StandardScaler().fit_transform(df_pos_sotgt)
df_pos_pca_sg = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_sotgt), columns=['shotsOnTotal_goalsTotal'])
df_pos_pca_sg['shotsOnTotal_goalsTotal'] = MinMaxScaler().fit_transform(df_pos_pca_sg)

# 3. passes_key, passes_total, goals_assists PCA
df_pos_pktga = df_pos[['passes_key', 'passes_total', 'goals_assists']]
df_pos_pktga = StandardScaler().fit_transform(df_pos_pktga)
df_pos_pca_pktga = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_pktga), columns=['passesKeyTotal_goalsAssists'])
df_pos_pca_pktga['passesKeyTotal_goalsAssists'] = MinMaxScaler().fit_transform(df_pos_pca_pktga)

# 4. duels_total, duels_won PCA
df_pos_duels = df_pos[['duels_total', 'duels_won']]
df_pos_duels = StandardScaler().fit_transform(df_pos_duels)
df_pos_pca_duels = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_duels), columns=['duelsWonTotal'])
df_pos_pca_duels['duelsWonTotal'] = MinMaxScaler().fit_transform(df_pos_pca_duels)

# 5. dribbles_attempts, dribbles_success PCA
df_pos_dribbles = df_pos[['dribbles_attempts', 'dribbles_success']]
df_pos_dribbles = StandardScaler().fit_transform(df_pos_dribbles)
df_pos_pca_dribbles = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_dribbles), columns=['dribblesAtmptsSuc'])
df_pos_pca_dribbles['dribblesAtmptsSuc'] = MinMaxScaler().fit_transform(df_pos_pca_dribbles)

# 6. games_appearences, substitutes_in, substitutes_bench PCA
df_pos_gasub = df_pos[['games_appearences', 'substitutes_in', 'substitutes_bench']]
df_pos_gasub = StandardScaler().fit_transform(df_pos_gasub)
df_pos_pca_gasub = pd.DataFrame(data = PCA(n_components=1).fit_transform(df_pos_gasub), columns=['gamesAppearance_sub'])
df_pos_pca_gasub['gamesAppearance_sub'] = MinMaxScaler().fit_transform(df_pos_pca_gasub)



# PCA Feature Table

df_pca = pd.concat([df_pos_pca_hw, df_pos_pca_sg, df_pos_pca_pktga, df_pos_pca_duels, 
                    df_pos_pca_dribbles, df_pos_pca_gasub], axis=1)
df_pca

df_pca.corr()[df_pca.corr() > 0.7]



# PCA feature들과 그 외 feature들의 OLS 확인

pca_cols = ['height', 'weight', 'shots_total', 'shots_on', 'goals_total', 'passes_key', 'passes_total', 'goals_assists', 'duels_total', 'duels_won', 'dribbles_attempts', 'dribbles_success', 'games_appearences', 'substitutes_in', 'substitutes_bench']
npca_cols = df_pos.columns.tolist()
npca_features = [item for item in npca_cols if item not in pca_cols]

df_ols = pd.concat([df_pos[npca_features].reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)
df_ols = df_ols.drop('player_name', axis=1)

# Drop 0 array - goals_conceded, penalty_saved

df_ols = df_ols.drop(['goals_conceded', 'penalty_saved', 'position'], axis=1)

df_ols_nf = df_ols

# OLS - Basis Model

from sklearn.model_selection import train_test_split

dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

#formula = "value ~ " + "+".join(feature_names)

model = sm.OLS.from_formula("value ~ " + "+".join(feature_names), data=df_train)
result = model.fit()
print(result.summary())

##############################################################################
from sklearn.model_selection import KFold

scores = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula("value ~ " + "+".join(feature_names), data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
#    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores.mean()))



# Follower Data 없이 OLS 진행

from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula = "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) +             scale(passes_accuracy) +             scale(games_played) +             scale(shotsOnTotal_goalsTotal) +             scale(gamesAppearance_sub)"

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())


##############################################################################
from sklearn.model_selection import KFold

scores = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores.mean()))

# Follower Data 포함 OLS 진행

from sklearn.model_selection import train_test_split

df_ols = df_ols
dfX = df_ols.drop(['value'], axis=1)
dfy = df_ols['value']
df = pd.concat([dfX, dfy], axis=1)
df_train, df_test = train_test_split(df, test_size=0.3, random_state=0)

feature_names = list(dfX.columns)
feature_names = ["scale({})".format(name) for name in feature_names]

formula =  "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) +             scale(passes_accuracy) +             scale(games_played) +             scale(shotsOnTotal_goalsTotal) +             scale(gamesAppearance_sub) +             scale(follower)"

model = sm.OLS.from_formula(formula, data=df_train)
result = model.fit()
print(result.summary())


##############################################################################
from sklearn.model_selection import KFold

scores = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_ols)):
    df_train = df_ols.iloc[idx_train]
    df_test = df_ols.iloc[idx_test]
    
    model = sm.OLS.from_formula(formula, data=df_train)
    result = model.fit()
    
    pred = result.predict(df_test)
    rss = ((df_test.value - pred) ** 2).sum()
    tss = ((df_test.value - df_test.value.mean())** 2).sum()
    rsquared = 1 - rss / tss
    
    scores[i] = rsquared
    print("학습 R2 = {:.8f}, 검증 R2 = {:.8f}".format(result.rsquared, rsquared))
print("모델 성능 : {}".format(scores.mean()))

# Feature들에 대해서 ANOVA 확인

model_full = sm.OLS.from_formula(
    "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) + \
            scale(passes_accuracy) + \
            scale(games_played) + \
            scale(shotsOnTotal_goalsTotal) + \
            scale(gamesAppearance_sub) + \
            scale(follower)", data=df_ols)
model_reduced = sm.OLS.from_formula(
    "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) + \
            scale(passes_accuracy) + \
            scale(games_played) + \
            scale(shotsOnTotal_goalsTotal) + \
            scale(gamesAppearance_sub)", data=df_ols)

sm.stats.anova_lm(model_reduced.fit(), model_full.fit())

model_full = sm.OLS.from_formula(
    "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) + \
            scale(passes_accuracy) + \
            scale(games_played) + \
            scale(shotsOnTotal_goalsTotal) + \
            scale(gamesAppearance_sub) + \
            scale(follower)", data=df_ols)

result = model_full.fit()
sm.stats.anova_lm(result, typ=2)



# gamesAppearance_sub 제거 후

model_full = sm.OLS.from_formula(
    "value ~ scale(age) + scale(I(age**2)) + scale(I(age**3)) + \
            scale(passes_accuracy) + \
            scale(games_played) + \
            scale(shotsOnTotal_goalsTotal) + \
            scale(follower)", data=df_ols)

result = model_full.fit()
sm.stats.anova_lm(result, typ=2)


# In[ ]:





# In[ ]:




