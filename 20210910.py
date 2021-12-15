import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import json

#ggplot 
# 남북한발전전력량 데이터
# 그래프는 북한 전력 발전량(제목)
# 수력, 화력(bar chart)
# x축은 연도, y축 발전량
# df_power = pd.read_excel('data/Datas/남북한발전전력량.xlsx', convert_float=True)


# df_north = df_power.loc[5:8]         #북한 발전량
# #df_north.T.plot()
# df_north.drop(columns='전력량 (억㎾h)', inplace=True)
# df_north.set_index('발전 전력별', inplace=True)
# df_north = df_north.T
# df_north.drop(columns='원자력', inplace=True)
# df_north.rename(columns={'합계':'power generation', '수력': 'water', '화력':'fire'}, inplace=True)
# ex1 = df_north['PG - 1 year'] = df_north['power generation'].shift(1)
# df_north['change_rate'] = (df_north['power generation'] / df_north['PG - 1 year'] -1) * 100


# plt.style.use(['ggplot'])
# fig = plt.figure(figsize=(24,10))
# ax1 = df_north[['water','fire']].plot(kind = 'bar', figsize=(20,10), width=0.7, stacked=True )
# ax2 = ax1.twinx()
# ax2.plot(df_north.index, df_north['change_rate'], ls='--', marker='.', markersize=20, color='green', label='change_rate')
# ax1.set_ylim(0,300)
# ax2.set_ylim(-100,100)
# ax1.set_xlabel('year',size=20)
# ax1.set_ylabel('power generation', size=20)
# ax2.set_ylabel('change rate (%)')
# plt.title('North Korea Power Generation (1990 - 2016)', size = 30)
# ax1.legend(loc='upper left')
# # plt.show()

#데이터 auto-mpg.csb 불러오기
#컬럼명 ['mpg'~'name']
# 연비에 대한 히스토그램 그리기
# 제목 : Histogram
# xlabel : mpg
# 2번째 그래프 제목 : Scatter plot - mpg vs. weight
# x축은 weight, y축은 mpg 로 산점도 그리기

# df_auto = pd.read_csv('data/Datas/auto-mpg.csv',header=None)
# df_auto.columns = ['mpg', 'cylinders', 'displacement', 'hp', 'weight', 'acceleration', 'model year', 'origin', 'name']
# plt.style.use(['default'])
# df_auto.plot(kind='scatter', x='weight',y='mpg', marker = '+', cmap='viridis', s=100, alpha = 0.3, figsize=(10,5))
# plt.title('Scatter plot - mpg vs. weight')

# plt.savefig('data/Datas/scatter_20210919.png')
# plt.savefig('data/Datas/scatter_20210919_transparent.png', transarent=True)

# df_auto.origin.value_counts().plot(kind='pie', figsize=(7,7), autopct='%1.1f%%', startangle=90, colors=['chocolate', 'bisque', 'cadetblue'])
# plt.title('Model origin', size=20)


# titanic 그래프
# 데이터는 타이타닉
# 스타일 씨본 darkgrid
# figsize 15,5
# 1행 3열
# 1,2,3 번 그래프 sns.distplot
# 데이터는 모두 타이타닉의 fare, 2번 hist=False, 3번 kde=False
# 차트 제목 : titanic fare-hist/kde, titanic fare - kde, titanic fare - hist
titanic = sns.load_dataset('titanic')
sns.set_style('darkgrid')
table = titanic.pivot_table(index=['sex'], columns=['class'], aggfunc='size')
sns.heatmap(table, annot=True, fmt='d', cmap='coolwarm', linewidths=.5, cbar=False)

# sns.set_style('whitegrid')
# g = sns.FacetGrid(data = titanic, col='who', row='survived')
# g = g.map(plt.hist, 'age')
# plt.show()


fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

sns.regplot(x='age', y='fare', data=titanic, ax=ax1)
sns.regplot(x='age', y='fare', data=titanic, ax=ax2, fit_reg=False)

# plt.show()
#지도 호출
# seoul_map1 = folium.Map(location=[37.55, 126.98], tiles='Stamen Terrain', zoom_start=12)
# seoul_map1.save('data/Datas/seoul1')

# df_map = pd.read_excel('data/Datas/서울지역 대학교 위치.xlsx')
# print(df_map)
# seoul_map = folium.Map(location=[37.55, 126.98], titles = 'Stamen Terrian', zoom_start=12)
# df_map.rename(columns = {'Unnamed: 0'  : 'universities'}, inplace=True)

# for name, lat, lng in zip(df_map['universities'], df_map.위도, df_map.경도):
#     folium.CircleMarker([lat, lng],radius=10,color='brown', fill=True,fill_color='coral',fill_opacity=0.7, popup=name).add_to(seoul_map)

# seoul_map.save('data/Datas/seoul_colleages_20210910_1.html')

df_gg = pd.read_excel('data/Datas/경기도인구데이터.xlsx', index_col='구분')
df_gg.info()
df_gg.colunms = df_gg.columns.map(str)
try:
    get_data = json.load(open('data/Datas/경기도행정구역경계.json', encoding='utf-8'))
except:
    get_data = json.load(open('data/Datas/경기도행정구역경계.json', encoding='utf-8-sig'))

g_map = folium.Map(location=[37.5502, 126.982], tiles='stamen Terrain', zoom_start=9)

year = '2017'

folium.Choropleth(geo_data = get_data, data = df_gg[year],
 columns=[df_gg.index, df_gg[year]], fill_color='YlOrRd', 
 fill_opacity=0.7, line_opacity=0.3, threshold_scale=[100000,100000,300000,500000, 700000],
 key_on='feature.properties.name').add_to(g_map)

g_map.save('data/Datas/20210910_gyonggi_population_'+year+'.html')