import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# print(pd.read_csv('data/Datas/read_csv_sample.csv', index_col = 'c0'))
df1 = pd.read_excel('data/Datas/남북한발전전력량.xlsx')
df2 = pd.read_excel('data/Datas/남북한발전전력량.xlsx', header=None)
# print(df2)
df = pd.read_json('data/Datas/read_json_sample.json')
df.index
html_df = pd.read_html('data/Datas/sample.html')
# print(html_df[0])
# print(html_df[1].set_index('name'))

data = {'name':['Jerry', 'Riah', 'Paul'], 'algol':['A','A+','B']}
df = pd.DataFrame(data)
# print(df)
df.to_json('data/Datas/df_sample20210909.json')
# print(pd.read_json('data/Datas/df_sample20210909.json'))
df.to_excel('data/Datas/df_sample20210909.xlsx')
# print(pd.read_excel('data/Datas/df_sample20210909.xlsx'))
writer = pd.ExcelWriter('./df_excelwriter.xlsx')
df.to_excel(writer, sheet_name= 'sheet1')
df.to_excel(writer, sheet_name= 'sheet2')
writer.save()
# print(pd.read_excel('data/Datas/df_excelwriter.xlsx', sheet_name='sheet1'))
df = pd.read_csv('data/Datas/auto-mpg.csv', header=None)
df.columns = ['mpg', 'cylinders', 'displacement', 'hp', 'weight', 'accleration', 'model year', 'origin', 'name']
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.info())
# print(df.dtypes)
# print(df.describe())
# print(df.mean())
# print(df.median())
# print(df.corr())        #상호간의 관계성
# print(df[['mpg','weight']].corr())      #원하는 값만 찾아서

df = pd.read_excel('data/Datas/남북한발전전력량.xlsx')

#문제 -> 남한, 북한 발전량 함계 데이터만 추출
df_ns = df.iloc[[0,5], 2:]
df_ns.index = ['south', 'north']
df_ns.columns = df_ns.columns.map(int)  #정수형으로 변경
# print(df_ns.columns)
# df_ns.T.plot()
#산점도 그리기
df_auto = pd.read_csv('data/Datas/auto-mpg.csv',header=None)
df_auto.columns = ['mpg', 'cylinders', 'displacement', 'hp', 'weight', 'acceleration', 'model year', 'origin', 'name']
# df_auto.plot(x='weight', y='mpg', kind='scatter').show()
#box ,mpg, cylinders
# df_auto[['mpg','cylinders']].plot(kind='box')

df_city = pd.read_excel('data/Datas/시도별 전출입 인구수.xlsx')
df_city.fillna(method='ffill', inplace=True)        #빈칸 앞에 있는 걸로 채우기

#print(df_city.전출지별.unique())

#전출지별 중에서 서울시 데이터만 추출 + 전출지는 서울, 전입지는 서울이 아닌것
df_seoul = df_city[(df_city.전출지별 == '서울특별시') & (df_city.전입지별 != '서울특별시')]
#1. 전출지별 컬럼 삭제, 2. 전입지별 이름을 전입지로, 3. 전입지 컬럼을 인덱스컬럼으로 변경
df_seoul.drop('전출지별', inplace=True, axis=1)
df_seoul.rename(columns = {'전입지별':'전입지'}, inplace=True)
df_seoul.set_index('전입지', inplace=True)
# print(df_seoul)

#1. 경기도만 선택, 2.plot 그래프 그리기,
# ex1 = df_seoul.loc['경기도']
# ex1.plot()

#1개의 df, 서울 -> 충청남도, 경상북도, 강원도, 전라남돔 --> df4 저장
df4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도']]
#2행 2열의 서브플럿 4개 생성
plt.style.use('ggplot')

df4.T.plot(kind='area')

fig = plt.figure(figsize=(24,10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
# 그래프
ax1.plot(df4.loc['충청남도'], marker='.', markersize=10, markerfacecolor='green', color='olive', linewidth = 3, label = 'Seoul to Chungcheongnam-do')
ax1.legend(loc='best')
ax2.plot(df4.loc['경상북도'], marker='.', markersize=10, markerfacecolor='blue', color='skyblue', linewidth = 3, label = 'Seoul to Gyongsanbuk-do')
ax2.legend(loc='best')
ax3.plot(df4.loc['강원도'], marker='.', markersize=10, markerfacecolor='red', color='magenta', linewidth = 3, label = 'Seoul to Gangwon-do')
ax3.legend(loc='best')
ax4.plot(df4.loc['전라남도'], marker='.', markersize=10, markerfacecolor='orange', color='yellow', linewidth = 3, label = 'Seoul to Jeollanam-do')
ax4.legend(loc='best')

ax1.set_title('Seoul to Chungcheongnam-do', size=20)
ax2.set_title('Seoul to Gyeongsangbuk-do', size=20)
ax3.set_title('Seoul to Gangwon-do', size=20)
ax4.set_title('Seoul to Jeollanam-do', size=20)

#ylimit
# ax1.set_ylim(50000,800000)
# ax2.set_ylim(50000,800000)
# ax3.set_ylim(50000,800000)
# ax4.set_ylim(50000,800000)

ax1.set_xticklabels(df4.columns, rotation='45')
ax2.set_xticklabels(df4.columns, rotation='45')
ax3.set_xticklabels(df4.columns, rotation='45')
ax4.set_xticklabels(df4.columns, rotation='45')

ax1.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='x', labelsize=8)
ax3.tick_params(axis='x', labelsize=8)
ax4.tick_params(axis='x', labelsize=8)
# ax1.set_xlabel('Period', size = 12)
# ax1.set_ylabel('Migration numbers', size = 10)

print(df)

