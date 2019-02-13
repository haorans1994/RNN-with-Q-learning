import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


stock1 = pd.read_csv('000001.csv')
stock1['date'] = pd.to_datetime(stock1['date'])

stock2 = pd.read_csv('000002.csv')
stock2['date'] = pd.to_datetime(stock2['date'])

plt.plot(stock1['date'], stock1['close'])
plt.plot(stock2['date'],stock2['close'])
plt.xticks(rotation=30)
plt.xlabel('day')
plt.ylabel('close price')
plt.title('stock-figure for two stocks')

plt.show()