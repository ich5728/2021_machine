import pandas as pd
# dict_data = {'a':1, 'b':2, 'c': 3}
# sr = pd.Series(dict_data)
# list_data = ['2019-09-09', 3.14, 'ABC', 100, True]
# li = pd.Series(list_data)
# tup_data = ('schol', '2021-09-09', 'man', True)
# tup = pd.Series(tup_data, index = ['기관', '날짜', '성별', '학생여부'])

# 데이터 프레임
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9],'c3':[10,11,12], 'c4':[13,14,15]}
df = pd.DataFrame(dict_data)
df1 = pd.DataFrame([[15, '남', '서울중'],[17, '여', '수리중']], columns=['나이', '성별', '학교'], index=['준서','예은'])
# print(df1)
# df1.index = ['학생1', '학생2']
# print(df1)
# print(df1.rename(columns={'나이' : '연령'}))
exam_data = {'수학':[90,80,70], '영어':[98,89,95], '음악':[85,95,100], '체육':[100,90,90],}
df2 = pd.DataFrame(exam_data, index=['서준','우현', '인아'])
df3 = df2.drop('우현')
# print(df2.loc[['서준', '인아']])
df4 = df2.reset_index()
df4.rename(columns={'index': 'name'}, inplace=True)
df4.set_index('name', inplace=True)
df4.loc['인아','체육'] = 70
df4['국어'] = 80
df4.loc['동규'] = 0
# print(df4.T)
df.rename(index={0 : 'r0', 1:'r1', 2:'r2'}, inplace= True)
df6 = df.reindex(['r0', 'r1', 'r2', 'r3','r4', 'r5'], fill_value=0)
# print(df6.sort_index(ascending=False))
# print(df6.sort_values('c2', ascending=False))
import numpy as np

student1 = pd.Series({'국어':np.nan, '영어':80, '수학':90})
student2 = pd.Series({'국어':80, '수학':80})
add = student1+student2
sub = student1-student2
mul = student1*student2
div = student1/student2
result = pd.DataFrame([add,sub,mul,div], index=['덧셈','뺄셈','곱셈','나눗셈'])
# print(student1.add(student2, fill_value=0))
# print(student1.sub(student2, fill_value=0))
# print(student1.mul(student2, fill_value=0))
# print(student1.div(student2, fill_value=0))

import seaborn as sns
titanic = sns.load_dataset('titanic')
# print(titanic)
#인덱스 추출
df8 = titanic[['age','fare']]
# print(df8)
addtion = df8 + 10
#마지막 5개 컬럼 추출
# print(titanic.loc[886:], ['age','fare'])
# print(df8.iloc[-5:])
# print(df8.tail())
