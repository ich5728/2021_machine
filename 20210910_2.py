import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import json

titanic = sns.load_dataset('titanic')
sns.set_style('darkgrid')
table = titanic.pivot_table(index=['sex'], columns=['class'], aggfunc='size')

titanic.deck.value_counts(dropna =False)
titanic.embark_town.fillna(method='ffill', inplace=True)
a = titanic.isnull().sum()
titanic.drop_duplicates()

# 1.60934 / 3.78541
# kpl 컬럼생성, 데이터는 위의 결과값을 곱한다.
# 소수점 둘째자리에서 반올림, round()사용 
df = titanic[['age','fare']]
def add_10(x):
    return x+10

def add_two(a, b):
    return a + b
df['ten'] = 10
df.age.apply(add_10)
df.age.apply(add_two, b=10)
df.age.apply(lambda x: add_10(x))

def missing_value(series):
    return series.isnull()
def min_max(x):
    return x.max() - x.min()
df.apply(min_max)

df['add'] = df.apply( lambda x: add_two(x['age'], x['ten']), axis=1)
def missing_count(x):
    return missing_value(x).sum()

def total_number_missing(x):
    return missing_count(x).sum()
titanic.columns =  titanic.columns.sort_values()


df_stock = pd.read_excel('Datas/주가데이터.xlsx')
df_stock['ymd'] = df_stock.연월일.astype('str')
dates = df_stock.ymd.str.split('-')
df_stock['y'] = dates.str.get(0)
df_stock['m'] = dates.str.get(1)
df_stock['d'] = dates.str.get(2)

titanic = sns.load_dataset('titanic')
a= titanic.loc[(titanic.sibsp == 3) | (titanic.sibsp == 4 ) | (titanic.sibsp == 5)]
b= titanic.loc[titanic.sibsp.isin([3,4,5])]
print(a)
print(b)

