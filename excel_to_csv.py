import pandas as pd

xlsx_path = r'D:\Downloads\DUCSE Documents\4-1\Mine\CSE-4125 Introduction to Machine Learning\Lab\Final Project\taining files\Real ML_DataSet (2).xlsx'
csv_path = r'D:\Downloads\DUCSE Documents\4-1\Mine\CSE-4125 Introduction to Machine Learning\Lab\Final Project\taining files\Real ML_DataSet (2).csv'

df = pd.read_excel(xlsx_path, engine='openpyxl')
df.rename(columns={'Time': 'Day'}, inplace=True)
df.insert(1, 'Image', '')

col_order = [
    'Image Name',
    # 'Image',
    'Distance between  building (feet)',
    'Distance from Camera (feet)',
    'level (positive, moderate, negative)',
    'Area',
    'Day'
]
df = df[col_order]

df.to_csv(csv_path, index=False)
print(f'Saved {len(df)} rows to {csv_path}')
