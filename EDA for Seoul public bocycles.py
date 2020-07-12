#!/usr/bin/env python
# coding: utf-8

# ## 라이브러리 import

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
COLORS = sns.color_palette()
import chart_studio.plotly as py
import cufflinks as cf
print(cf.__version__)
cf.go_offline()
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import plotly
plotly.offline.init_notebook_mode()


#  ## 그래프 시각화의 한글지원 코드

# In[2]:


import matplotlib
from matplotlib import font_manager, rc
import platform

try : 
    if platform.system() == 'Windows':
    # 윈도우인 경우
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
    else:    
    # Mac 인 경우
        rc('font', family='AppleGothic')
except : 
    pass
matplotlib.rcParams['axes.unicode_minus'] = False 


# # <데이터 소개>
#     - 서울시 공공자전거 이용현황(2019년 10월1일 ~ 11월30일)

# ### 연령대별로 EDA
#     - 연령대별로는 어떻게 봐야할까?
#         - 이용시간, 이동거리, 이용건수를 추출
#             - 이용시간 대비 이동거리와 이용건수 비교, 분석
#             - 운동량은 날려야할까?
#     

# #### 데이터 로드 및 concat
#     - 공공데이터 csv파일의 한글깨짐현상
#         - 공공데이터 파일의 Encoding은 utf-8방식으로 통일해 주었으면 좋겠지만, 거의 대부분 cp949나 euc-kr방식으로 인코딩 되어 있음
#         - 해당 서울시 공공자전거 csv파일의 cp949로 인코딩이 되어 있고 utf8 불러왔을 때, ???? 현상이 나타남
#         - utf8 로 변환 후 재로드

# In[4]:


df1 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트/EDA프로젝트데이터/서울특별시 공공자전거 이용정보(시간대별)_20190601_20191130(7).csv', encoding='utf-8')
df1 = df1.loc[458674:]
df2 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트/EDA프로젝트데이터/서울특별시 공공자전거 이용정보(시간대별)_20190601_20191130(8).csv', encoding='utf-8')
df3 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트/EDA프로젝트데이터/서울특별시 공공자전거 이용정보(시간대별)_20190601_20191130(9).csv', encoding='utf-8')
df4 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트/EDA프로젝트데이터/서울특별시 공공자전거 이용정보(시간대별)_20190601_20191130(10).csv', encoding='utf-8')

pb_df = pd.concat([df1, df2, df3, df4]).reset_index(drop=True)
pb_df.isnull().sum()
pb_df.dropna(inplace=True)
pb_df


# #### 정규표현식 활용
#     - 연령대코드에서 ~10대, ~70대로 표기가 되어 있음
#     - 연령대코드 groupby 시, 정렬되지 않는 모습을 보임
#     - 정규표현식을 활용하여 '~'를 제거

# In[5]:


pb_df['연령대코드'] = pb_df['연령대코드'].str.replace(pat=r'[~@]', repl = r' ', regex=True)
pb_df


# #### 성별 Columns 대문자로 변환

# In[8]:


pb_df['성별'] = pb_df['성별'].str.upper()
pb_df


# #### 특정row 0값 제거하기
#     - null값이 아닌 0제거

# In[9]:


pb_df = pb_df[pb_df.이동거리 != 0]
pb_df


# In[10]:


age_by_df = pb_df[['대여일자','대여구분코드','성별','연령대코드','이용건수','운동량','이동거리','사용시간']]
age_by_df = age_by_df.reset_index(drop=True)
age_by_df['건당 이동거리'] = age_by_df['이동거리']/age_by_df['이용건수']

age_by_df


# #### 성별
#     - 성별에 따른 평균 사용시간 탐색
#     - 성별에 따른 평균 이동거리 탐색

# In[11]:


resultbysex_0 = round(age_by_df.groupby('성별').mean()['사용시간'].reset_index(name='성별 사용시간'),2)
resultbysex_0


# In[12]:


resultbysex_1 = round(age_by_df.groupby('성별').mean()['이동거리'].reset_index(name='평균이동거리'),2)
resultbysex_1


# In[13]:


resultbysex_2 = round(age_by_df.groupby('성별').mean()['건당 이동거리'].reset_index(name='1건당 이동거리'),2)
resultbysex_2


# In[14]:


result_sex = pd.concat([resultbysex_1,resultbysex_2],axis=1)
result_sex.columns = ['성별', '평균이동거리', '123424', '1건당 이동거리']
result_sex = result_sex.drop('123424', axis=1)
result_sex = result_sex.set_index('성별')


# In[17]:


result_sex[['평균이동거리','1건당 이동거리']].iplot(kind='barh',title='Minute Average', xTitle='SEX', yTitle='VALUES')


# #### 연령대별 대여구분코드를 나누어 이동거리 및 사용시간을 확인
#     - 운동량과 탄소량 형변환 (string > float)
#     - apply(pd.to_numeric) 방식으로 특정 컬럼 형 변환
#     - 숫자e를 보기 편하게 

# In[18]:


age_by_df['운동량(float)'] = age_by_df['운동량'].apply(pd.to_numeric, errors = 'coerce')
age_by_df


# In[19]:


pd.options.display.float_format = '{:.2f}'.format

resultbyage_0 = round(age_by_df.groupby('연령대코드').mean()['사용시간'].reset_index(name='연령별 사용시간'),2)
resultbyage_0


# In[20]:


resultbyage_1 = round(age_by_df.groupby('연령대코드').mean()['이동거리'].reset_index(name='평균이동거리'),2)
resultbyage_1['평균이동거리(km)'] = resultbyage_1['평균이동거리'] / 1000
resultbyage_1 = resultbyage_1.drop('평균이동거리', axis=1)
resultbyage_1


# In[22]:


age_datas = pd.concat([resultbyage_0, resultbyage_1], axis=1)
age_datas.columns = ['연령대코드', '연령별 사용시간', '드랍', '평균이동거리(km)']
age_datas.drop('드랍',axis=1, inplace=True)
age_datas


# In[23]:


x = age_datas['연령대코드']
y1 = age_datas['연령별 사용시간']
y2 = age_datas['평균이동거리(km)']

fig, ax1 = plt.subplots(figsize=(16,8))
ax2 = ax1.twinx()

data_y1 = ax1.plot(x, y1, color='b', marker='o', label='평균사용시간')
data_y2 = ax2.plot(x, y2, color='r', marker='s', label='평균이동거리')

ax1.set_xlabel('AGE')
ax1.set_ylabel('평균사용시간')
ax2.set_ylabel('평균이동거리')
ax1.legend()
plt.show()


# #### 연령대별 분당 이동거리, 운동량 데이터 전처리

# In[24]:


age_by_df2 = age_by_df.groupby('연령대코드').sum()[['이동거리','사용시간','운동량(float)']]

y = age_by_df2['이동거리'] / age_by_df2['사용시간']
pd.DataFrame(y)
age_by_df2 = pd.concat([age_by_df2,y], axis=1)
age_by_df2
age_by_df2.columns = ['이동거리','사용시간','운동량(float)','분당이동거리']

age_by_df2['분당운동량'] = age_by_df2['운동량(float)'] / age_by_df2['사용시간']
age_by_df2 = age_by_df2.reset_index()
age_by_df2


# #### 데이터 정규화 작업
#     - 데이터 정규화 작업을 위한 데이터 깊은 복사

# In[26]:


age_by_df3 = age_by_df2.copy()
age_by_df3


# In[28]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(age_by_df3[['이동거리','사용시간','운동량(float)']])
X = sc.transform(age_by_df3[['이동거리','사용시간','운동량(float)']])
X


# In[55]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

preprocessing.minmax_scale(age_by_df3['이동거리'])

# age_by_df3['이동거리']
# scaler = StandardScaler()
# scaler.fit_transform(age_by_df3['이동거리'])

# min_max_scaler = preprocessing.MinMaxScaler()
# x = age_by_df3[['분당이동거리','분당운동량']].values
# x_scaled = min_max_scaler.fit(x)
# print(x_scaled.data_max_)
# output = min_max_scaler.transform(x)
# output = pd.DataFrame(output, columns=age_by_df3[['분당이동거리','분당운동량']].columns)
# output


# In[60]:


age_by_df3 = pd.concat([age_by_df3,output], axis=1)
age_by_df3.columns = ['연령대코드','이동거리','사용시간','운동량(float)','분당이동거리','분당운동량','분당이동거리(norm)','분당운동량(norm)']
age_by_df3 = age_by_df3.set_index('연령대코드')    
age_by_df3


# #### 연령대별 분당이동거리, 운동량 그래프
#     - 시각화의 특성을 더욱 더 살리기 위해 iplot 을 사용
#     - 칼로리소모만을 기준으로 볼 때, 10대, 20대의 경우 이동거리보다 운동량이 적은 것은 천천히 혹은 짧게 여러번 탔다는 것을 의미할 수 있음

# In[61]:


age_by_df3[['분당이동거리(norm)','분당운동량(norm)']].iplot(kind='bar',title='Data per minute', xTitle='AGE', yTitle='NORMALIZING VALUES')


# In[64]:


age_by_df
xy = age_by_df.groupby(['대여일자','연령대코드']).mean()[['운동량(float)','이동거리','사용시간']]
xy = xy.reset_index()
xy


# In[66]:


sns.set(style="white")

sns.relplot(x="이동거리", y="운동량(float)", hue="연령대코드", size='사용시간',
            sizes=(40, 400), alpha=.5, palette="muted",
            height=7, data=xy)


# ### 날짜별로 EDA
#     - 주 단위로 보면 데이터가 어떻게 나올까?
#         - 주 단위 시간별 이용시간, 이동거리 등 데이터 전처리

# In[67]:


age_by_df['대여일자'] = pd.to_datetime(age_by_df['대여일자'], infer_datetime_format=True)
age_by_df = age_by_df.set_index('대여일자')
age_by_df


# #### 주별 이동거리(km)
#     - km로 변환하여 계산
#     - 날씨가 추워지면 추워질수록 이동거리가 줄어드는 현상을 보임
#     - 운동량 역시 날씨가 추워지고 이동거리가 짧아지면서 칼로리 소모가 급격히 떨어지는 것을 볼 수 있음
#     - 10월 1주는 월요일이 9월에 포함이 되어 빠지게 되서 데이터양이 적게 보임

# In[68]:


time_by_df = age_by_df.resample('W'). sum()['이동거리']
time_by_df.astype(np.float) / 1000


# In[69]:


weekly_momentum = age_by_df.resample('W').sum()['운동량(float)']
weekly_momentum.astype(np.float)


# In[70]:


date_by_df = age_by_df.copy()
date_by_df['운동량'] = date_by_df['운동량'].astype(np.float)
date_by_df


# In[72]:


date1 = date_by_df.groupby('대여일자').sum()[['이동거리','사용시간','운동량']]
date1['이동거리(km)'] = date1['이동거리'] / 1000
date1['사용시간(hour)'] = round(date1['사용시간'] / 60,2)
date1.drop(['이동거리','사용시간'], axis=1, inplace=True)
date1


# #### 데이터 정규화 작업

# In[73]:


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x = date1.copy()
x[:] = min_max_scaler.fit_transform(x[:])
x


#     #####  10월2일, 7일, 11월13일, 17일은 전국에 비 엄청 옴
#     #####  10월에는 한달 평균 최저기온이 5도 였는데 11월에는 -5도로 급격히 떨어졌기 대문에 자전거 사용량이 현저히 줄어든 것
#     #####  10월 25일 ~ 27일 사이에 비가 오면서 온도가 급격히 떨어진 것을 볼 수 있음

# In[79]:


x1 = x.reset_index()
plt.figure(figsize=(18,12))
plt.plot(x1['대여일자'], x1['운동량'], label="운동량")
plt.plot(x1['대여일자'], x1['이동거리(km)'],label="이동거리(km)")
plt.plot(x1['대여일자'], x1['사용시간(hour)'],label="사용시간(hour)")
plt.xticks(rotation=45)
plt.grid(False)
plt.legend()
plt.title("사용시간, 이동거리, 운동량, 탄소량")

plt.show()


# #### 대여소별 이용량, 이용시간 EDA

# ##### 대여소 별 실제 지도위의 분포도 그리기
#     - 실제 지도에 표시하기 위해서는 위도와 경도 데이터가 필수적

# In[4]:


import requests
# import pprint
from pandas.io.json import json_normalize

#-*- coding:utf-8 -*-
url = "http://openapi.seoul.go.kr:8088/43797267456268633130315757616b41/json/bikeList/1/1000/"

payload = {}
headers= {}

response = requests.request("GET", url, headers=headers, data = payload)
# print(response.text.encode('utf-8'))
# pprint.pprint(response.json())
json_object = response.json()
json_object['rentBikeStatus']['row']

# df = json_normalize(json_object['rentBikeStatus']['row'])
# df


# In[1]:


#-*- coding:utf-8 -*-
url = "http://openapi.seoul.go.kr:8088/43797267456268633130315757616b41/json/bikeList/1001/2000/"

payload = {}
headers= {}

response = requests.request("GET", url, headers=headers, data = payload)
# print(response.text.encode('utf-8'))
# pprint.pprint(response.json())
json_object = response.json()
json_object['rentBikeStatus']['row']

df2 = json_normalize(json_object['rentBikeStatus']['row'])
df2


# In[122]:


df = pd.concat([df,df2]).reset_index(drop=True)
df.info()
df


# In[123]:


df_map = df[['stationName','stationLatitude','stationLongitude']]
df_map[['대여 대여소번호', '대여소명']] = df_map['stationName'].str.split('.', n=1, expand=True)
df_map

df_map['stationLatitude'] = df_map['stationLatitude'].astype(np.float)
df_map['stationLongitude'] = df_map['stationLongitude'].astype(np.float)
df_map['대여 대여소번호'] = df_map['대여 대여소번호'].astype(np.int)

df_map.info()
df_map

# df_map1['stationName'] = df_map1['stationName'].str.replace(pat=r'[0-9.]', repl = r'', regex=True)


# #### 대여소 경도,위도 데이터와 대여소 별 이용건수 데이터 merge 
#     - 각각의 다른 데이터를 '대여소' 기준으로 merge

# In[124]:


xxx1 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트데이터/무제 폴더/서울특별시 공공자전거 대여정보_201910_1.csv')
xxx1.columns = ['자전거번호','대여일시','대여 대여소번호','stationName','대여거치대','반납일시','반납대여소번호','반납대여소명','반납거치대','이용시간','이용거리']

xxx2 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트데이터/무제 폴더/서울특별시 공공자전거 대여정보_201910_2.csv')
xxx2.columns = ['자전거번호','대여일시','대여 대여소번호','stationName','대여거치대','반납일시','반납대여소번호','반납대여소명','반납거치대','이용시간','이용거리']

xxx3 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트데이터/무제 폴더/서울특별시 공공자전거 대여정보_201910_3.csv')
xxx3.columns = ['자전거번호','대여일시','대여 대여소번호','stationName','대여거치대','반납일시','반납대여소번호','반납대여소명','반납거치대','이용시간','이용거리']

xxx4 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트데이터/무제 폴더/서울특별시 공공자전거 대여정보_201911_1.csv')
xxx4.columns = ['자전거번호','대여일시','대여 대여소번호','stationName','대여거치대','반납일시','반납대여소번호','반납대여소명','반납거치대','이용시간','이용거리']

xxx5 = pd.read_csv('/Users/wglee/Desktop/DATA ANALYSIS/데이터사이언스school/EDA프로젝트데이터/무제 폴더/서울특별시 공공자전거 대여정보_201911_2.csv')
xxx5.columns = ['자전거번호','대여일시','대여 대여소번호','stationName','대여거치대','반납일시','반납대여소번호','반납대여소명','반납거치대','이용시간','이용거리']

xxx = pd.concat([xxx1,xxx2,xxx3,xxx4,xxx5], axis = 0)
xxx


# In[125]:


result = pd.merge(df_map, xxx, on='대여 대여소번호', how='outer')
result


# In[126]:


result.drop(['stationName_x','대여소명','대여거치대','반납일시','반납대여소번호','반납대여소명','반납거치대','자전거번호'],axis=1,inplace=True)
result


# In[127]:


result.isnull().sum()
result = result.dropna()
result


# In[128]:


result1 = result.groupby(['대여 대여소번호','stationName_y']).mean()[['stationLatitude','stationLongitude','이용시간']]
result1


# In[129]:


result1 = result1.reset_index()
result1


# ####  대여소 별 이용시간 top50 & bottom50 시각화
#     - 1551개의 정류소를 다 표현하기에는 너무 많아서 상위50개, 하위50개를 출력

# In[130]:


result1 = result1.sort_values('이용시간', ascending=False)
result2 = result1.head(50)
result3 = result1.tail(50)


# In[116]:


get_ipython().system('pip install folium')


# In[131]:


import folium

map_osm = folium.Map(location=[37.5539876, 126.983000], zoom_start=12)
map_osm


# In[132]:


for item in result2.index:
    lat = result2.loc[item, 'stationLatitude']
    long = result2.loc[item, 'stationLongitude']
    
    folium.CircleMarker([lat, long],
                       radius=result2.loc[item, '이용시간']/3,
                       popup=result2.loc[item, '대여 대여소번호'],
                       color = 'blue',
                       fill = True).add_to(map_osm)
map_osm


# In[133]:


for item in result3.index:
    lat = result3.loc[item, 'stationLatitude']
    long = result3.loc[item, 'stationLongitude']
    
    folium.CircleMarker([lat, long],
                       radius=result3.loc[item, '이용시간']/2,
                       popup=result3.loc[item, '대여 대여소번호'],
                       color = 'red',
                       fill = True).add_to(map_osm)
map_osm


# In[ ]:





# In[ ]:




