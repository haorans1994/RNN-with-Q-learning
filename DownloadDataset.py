

import tushare as ts


df = ts.get_hist_data('000003')
df.to_csv('000003.csv')
