from re import X
import pandas as pd
from pandas._libs.tslibs.timedeltas import ints_to_pytimedelta
import seaborn as sns

#1. stock prick.xlsx, stock valuation.xlsx 불러오기
#2. 데이터를 확인
#3. 데이터 프레임 합치기
df_pr = pd.read_excel('Datas/stock price.xlsx')
df_va = pd.read_excel('Datas/stock valuation.xlsx')

# pd.set_option('display.max_row',100)

a = pd.concat([df_pr, df_va])       #일단 합침
b = pd.merge(df_pr,df_va ,how='left', left_on ='stock_name', right_on='name')
c = pd.merge(df_va,df_pr ,how='right', left_on ='name', right_on='stock_name')            #같은 키를 맞춰서 합침, 없는 것은 생략(디폴트) 

# stock price 에서 pricerk 50000 미만인 것
# 위의 결과값과 
price_50000 = df_pr.loc[df_pr['price'] < 50000]
d = pd.merge(price_50000, df_va)
df_pr.set_index('id', inplace= True)
df_va.set_index('id', inplace= True)
e = df_pr.join(df_va)                   #인덱스 값이 동일한 것 끼리

#groupby , 같은 값 끼리 묶어서 집계

titanic = sns.load_dataset('titanic')
t5 = titanic[['age', 'sex', 'class', 'fare', 'survived']]
grouped = t5.groupby(['class'])
grouped_2 = t5.groupby(['class', 'sex'])
grouped.get_group('First')
for key, group in grouped:
    print('key:', key)
    print('index number:', len(group))
    print('group:', group.head())

# 최대값 - 최소값을 뺀 결과값을 반환하는 함수를 만들고 해당 함수를 이용해서 grouped에 적용합니다.
def max_min(x):
    return x.max() - x.min()

mm = grouped.agg(max_min)
f = grouped.agg(['min', 'max', 'mean', 'std'])
g = grouped.agg({'fare':['min','max'], 'age':'mean'})
grouped.filter(lambda x: len(x) >= 200)

# 1. z-score 함수를 생성. (x- 평균)/표준편차 (정규화)
# 2. grouped에 적용
# 3. grouped에서 나이의 평균이 30 미만인 데이터프레임 선택, 이 조건에 맞는 데이터 프레임을 각각 출력 (head())
# 4. grouped_2에서 평균으로 집계를 한 결과값에서 First인 경우만 출력
# 4. grouped_2에서 평균으로 집계를 한 결과값에서 First이고, sex가 female인 경우만 출력

def z_score(x):
    return (x - x.mean()) / x.std()

grouped.age.apply(z_score)

age_filter = grouped.apply(lambda x: x.mean() < 30)
grouped.get_group('Second').head()
grouped.get_group('Third').head()

for x in age_filter.index:
    if age_filter.loc[x, 'age'] == True:
        print(grouped.get_group(x).head())

df = grouped_2.mean()
df.loc['First']