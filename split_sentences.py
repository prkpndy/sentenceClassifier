import pandas as pd
import xlrd
import re

file = pd.read_excel("data/test.xlsx")

ls = []
for string in file['posts']:
    lst = re.split('\.|\?|\n|\!', string)

    for strg in lst:
        ls.append(strg)

ans = pd.DataFrame(ls, columns = ['post'])
ans.to_csv('data/spli_test.csv')
