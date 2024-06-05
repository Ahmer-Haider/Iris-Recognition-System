import pandas as pd

df = pd.read_excel('excelDataSet.xlsx')
df['CASTED VOTE'] = 0
df.to_excel('excelDataSet.xlsx', index=False)
print(df)
