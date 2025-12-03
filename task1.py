import pandas as pd
data = pd.read_csv(r'C:\Users\bunny\Downloads\cardata.csv',names= ['buying','maint','doors','persons','lug_boot','safety','class'])
print(data)

from pandas import DataFrame
Cars = {'Brand':['Honda civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price':[32000,35000,37000,45000]}
df = DataFrame(Cars, columns = ['Brand','Price'])
export_excel = df.to_excel('export_data.xlsx')