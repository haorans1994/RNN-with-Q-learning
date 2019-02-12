

import tushare as ts


df = ts.get_hist_data('000001')
df.to_csv('000001.csv')
