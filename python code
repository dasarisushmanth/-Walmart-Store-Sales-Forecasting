import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
sns.set_style("whitegrid")
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
/anaconda/lib/python3.6/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
train=pd.read_csv('....../Walmart/train.csv')
test=pd.read_csv('......./Walmart/test.csv')
store=pd.read_csv('....../Walmart/stores.csv')
feature=pd.read_csv('...../Walmart/features.csv')
train.head()
Store	Dept	Date	Weekly_Sales	IsHoliday
0	1	1	2010-02-05	24924.50	False
1	1	1	2010-02-12	46039.49	True
2	1	1	2010-02-19	41595.55	False
3	1	1	2010-02-26	19403.54	False
4	1	1	2010-03-05	21827.90	False
feature.head()
Store	Date	Temperature	Fuel_Price	MarkDown1	MarkDown2	MarkDown3	MarkDown4	MarkDown5	CPI	Unemployment	IsHoliday
0	1	2010-02-05	42.31	2.572	NaN	NaN	NaN	NaN	NaN	211.096358	8.106	False
1	1	2010-02-12	38.51	2.548	NaN	NaN	NaN	NaN	NaN	211.242170	8.106	True
2	1	2010-02-19	39.93	2.514	NaN	NaN	NaN	NaN	NaN	211.289143	8.106	False
3	1	2010-02-26	46.63	2.561	NaN	NaN	NaN	NaN	NaN	211.319643	8.106	False
4	1	2010-03-05	46.50	2.625	NaN	NaN	NaN	NaN	NaN	211.350143	8.106	False
merge_df=pd.merge(train,feature, on=['Store','Date'], how='inner')
merge_df.head()
Store	Dept	Date	Weekly_Sales	IsHoliday_x	Temperature	Fuel_Price	MarkDown1	MarkDown2	MarkDown3	MarkDown4	MarkDown5	CPI	Unemployment	IsHoliday_y
0	1	1	2010-02-05	24924.50	False	42.31	2.572	NaN	NaN	NaN	NaN	NaN	211.096358	8.106	False
1	1	2	2010-02-05	50605.27	False	42.31	2.572	NaN	NaN	NaN	NaN	NaN	211.096358	8.106	False
2	1	3	2010-02-05	13740.12	False	42.31	2.572	NaN	NaN	NaN	NaN	NaN	211.096358	8.106	False
3	1	4	2010-02-05	39954.04	False	42.31	2.572	NaN	NaN	NaN	NaN	NaN	211.096358	8.106	False
4	1	5	2010-02-05	32229.38	False	42.31	2.572	NaN	NaN	NaN	NaN	NaN	211.096358	8.106	False
merge_df.describe().transpose()
count	mean	std	min	25%	50%	75%	max
Store	421570.0	22.200546	12.785297	1.000	11.000000	22.00000	33.000000	45.000000
Dept	421570.0	44.260317	30.492054	1.000	18.000000	37.00000	74.000000	99.000000
Weekly_Sales	421570.0	15981.258123	22711.183519	-4988.940	2079.650000	7612.03000	20205.852500	693099.360000
Temperature	421570.0	60.090059	18.447931	-2.060	46.680000	62.09000	74.280000	100.140000
Fuel_Price	421570.0	3.361027	0.458515	2.472	2.933000	3.45200	3.738000	4.468000
MarkDown1	150681.0	7246.420196	8291.221345	0.270	2240.270000	5347.45000	9210.900000	88646.760000
MarkDown2	111248.0	3334.628621	9475.357325	-265.760	41.600000	192.00000	1926.940000	104519.540000
MarkDown3	137091.0	1439.421384	9623.078290	-29.100	5.080000	24.60000	103.990000	141630.610000
MarkDown4	134967.0	3383.168256	6292.384031	0.220	504.220000	1481.31000	3595.040000	67474.850000
MarkDown5	151432.0	4628.975079	5962.887455	135.160	1878.440000	3359.45000	5563.800000	108519.280000
CPI	421570.0	171.201947	39.159276	126.064	132.022667	182.31878	212.416993	227.232807
Unemployment	421570.0	7.960289	1.863296	3.879	6.891000	7.86600	8.572000	14.313000
from datetime import datetime as dt
merge_df['DateTimeObj']=[dt.strptime(x,'%Y-%m-%d') for x in list(merge_df['Date'])]
merge_df['DateTimeObj'].head()
0   2010-02-05
1   2010-02-05
2   2010-02-05
3   2010-02-05
4   2010-02-05
Name: DateTimeObj, dtype: datetime64[ns]
plt.plot(merge_df[(merge_df.Store==1)].DateTimeObj, merge_df[(merge_df.Store==1)].Weekly_Sales, 'ro')
plt.show()

weeklysales=merge_df.groupby(['Store','Date'])['Weekly_Sales'].apply(lambda x:np.sum(x))
weeklysales[0:5]
Store  Date      
1      2010-02-05    1643690.90
       2010-02-12    1641957.44
       2010-02-19    1611968.17
       2010-02-26    1409727.59
       2010-03-05    1554806.68
Name: Weekly_Sales, dtype: float64
weeklysaledept=merge_df.groupby(['Store','Dept'])['Weekly_Sales'].apply(lambda x:np.sum(x))
weeklysaledept[0:5]
Store  Dept
1      1       3219405.18
       2       6592598.93
       3       1880518.36
       4       5285874.09
       5       3468885.58
Name: Weekly_Sales, dtype: float64
weeklyscale=weeklysales.reset_index()
weeklyscale[0:5]
Store	Date	Weekly_Sales
0	1	2010-02-05	1643690.90
1	1	2010-02-12	1641957.44
2	1	2010-02-19	1611968.17
3	1	2010-02-26	1409727.59
4	1	2010-03-05	1554806.68
walmartstore=pd.merge(weeklyscale, feature, on=['Store', 'Date'], how='inner')
walmartstore.head()
Store	Date	Weekly_Sales	Temperature	Fuel_Price	MarkDown1	MarkDown2	MarkDown3	MarkDown4	MarkDown5	CPI	Unemployment	IsHoliday
0	1	2010-02-05	1643690.90	42.31	2.572	NaN	NaN	NaN	NaN	NaN	211.096358	8.106	False
1	1	2010-02-12	1641957.44	38.51	2.548	NaN	NaN	NaN	NaN	NaN	211.242170	8.106	True
2	1	2010-02-19	1611968.17	39.93	2.514	NaN	NaN	NaN	NaN	NaN	211.289143	8.106	False
3	1	2010-02-26	1409727.59	46.63	2.561	NaN	NaN	NaN	NaN	NaN	211.319643	8.106	False
4	1	2010-03-05	1554806.68	46.50	2.625	NaN	NaN	NaN	NaN	NaN	211.350143	8.106	False
walmartstoredf = walmartstore.iloc[:, list(range(5)) + list(range(10,13))]
walmartstoredf.head()
Store	Date	Weekly_Sales	Temperature	Fuel_Price	CPI	Unemployment	IsHoliday
0	1	2010-02-05	1643690.90	42.31	2.572	211.096358	8.106	False
1	1	2010-02-12	1641957.44	38.51	2.548	211.242170	8.106	True
2	1	2010-02-19	1611968.17	39.93	2.514	211.289143	8.106	False
3	1	2010-02-26	1409727.59	46.63	2.561	211.319643	8.106	False
4	1	2010-03-05	1554806.68	46.50	2.625	211.350143	8.106	False
walmartstoredf['DateTimeObj'] = [dt.strptime(x, '%Y-%m-%d') for x in list(walmartstoredf['Date'])]
weekNo=walmartstoredf.reset_index()
weekNo = [(x - walmartstoredf['DateTimeObj'][0]) for x in list(walmartstoredf['DateTimeObj'])]
walmartstoredf['Week'] = [np.timedelta64(x, 'D').astype(int)/7 for x in weekNo]
walmartstoredf.head()
Store	Date	Weekly_Sales	Temperature	Fuel_Price	CPI	Unemployment	IsHoliday	DateTimeObj	Week
0	1	2010-02-05	1643690.90	42.31	2.572	211.096358	8.106	False	2010-02-05	0.0
1	1	2010-02-12	1641957.44	38.51	2.548	211.242170	8.106	True	2010-02-12	1.0
2	1	2010-02-19	1611968.17	39.93	2.514	211.289143	8.106	False	2010-02-19	2.0
3	1	2010-02-26	1409727.59	46.63	2.561	211.319643	8.106	False	2010-02-26	3.0
4	1	2010-03-05	1554806.68	46.50	2.625	211.350143	8.106	False	2010-03-05	4.0
plt.plot(walmartstoredf.DateTimeObj, walmartstoredf.Weekly_Sales, 'ro')
plt.show()

walmartstoredf['IsHolidayInt'] = [int(x) for x in list(walmartstoredf.IsHoliday)]
walmartstoredf.head()
Store	Date	Weekly_Sales	Temperature	Fuel_Price	CPI	Unemployment	IsHoliday	DateTimeObj	Week	IsHolidayInt
0	1	2010-02-05	1643690.90	42.31	2.572	211.096358	8.106	False	2010-02-05	0.0	0
1	1	2010-02-12	1641957.44	38.51	2.548	211.242170	8.106	True	2010-02-12	1.0	1
2	1	2010-02-19	1611968.17	39.93	2.514	211.289143	8.106	False	2010-02-19	2.0	0
3	1	2010-02-26	1409727.59	46.63	2.561	211.319643	8.106	False	2010-02-26	3.0	0
4	1	2010-03-05	1554806.68	46.50	2.625	211.350143	8.106	False	2010-03-05	4.0	0
walmartstoredf.Store.unique()
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45])
train_WM, test_WM = train_test_split(walmartstoredf, test_size=0.3,random_state=42)
plt.plot(walmartstoredf[(walmartstoredf.Store==1)].Week, walmartstoredf[(walmartstoredf.Store==1)].Weekly_Sales, 'ro')
plt.show()

XTrain = train_WM[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Week', 'IsHolidayInt']]
YTrain = train_WM['Weekly_Sales']
XTest = test_WM[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Week', 'IsHolidayInt']]
YTest = test_WM['Weekly_Sales']
wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear.fit(XTrain, YTrain)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
wmLinear.coef_
array([  -308.15266303,  -8747.78318227,  -1668.0479916 , -45058.54212884,
         -148.87164691,  95666.79893475])
#Performance on the test data sets
YHatTest = wmLinear.predict(XTest)
plt.plot(YTest, YHatTest,'ro')
plt.plot(YTest, YTest,'b-')
plt.show()

walmartstoredf['Store'].unique()
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45])
Store_Dummies = pd.get_dummies(walmartstoredf.Store, prefix='Store').iloc[:,1:]
walmartstoredf = pd.concat([walmartstoredf, Store_Dummies], axis=1)
walmartstoredf.head()
Store	Date	Weekly_Sales	Temperature	Fuel_Price	CPI	Unemployment	IsHoliday	DateTimeObj	Week	...	Store_36	Store_37	Store_38	Store_39	Store_40	Store_41	Store_42	Store_43	Store_44	Store_45
0	1	2010-02-05	1643690.90	42.31	2.572	211.096358	8.106	False	2010-02-05	0.0	...	0	0	0	0	0	0	0	0	0	0
1	1	2010-02-12	1641957.44	38.51	2.548	211.242170	8.106	True	2010-02-12	1.0	...	0	0	0	0	0	0	0	0	0	0
2	1	2010-02-19	1611968.17	39.93	2.514	211.289143	8.106	False	2010-02-19	2.0	...	0	0	0	0	0	0	0	0	0	0
3	1	2010-02-26	1409727.59	46.63	2.561	211.319643	8.106	False	2010-02-26	3.0	...	0	0	0	0	0	0	0	0	0	0
4	1	2010-03-05	1554806.68	46.50	2.625	211.350143	8.106	False	2010-03-05	4.0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 55 columns

train_WM, test_WM = train_test_split(walmartstoredf, test_size=0.3,random_state=42)
XTrain = train_WM.iloc[:,([3,4,5,6] + [9,10]) + list(range(11,walmartstoredf.shape[1]))]
yTrain = train_WM.Weekly_Sales
                                                    
XTest = test_WM.iloc[:,([3,4,5,6] + [9,10]) + list(range(11,walmartstoredf.shape[1]))]
yTest=test_WM.Weekly_Sales
XTrain.head()
Temperature	Fuel_Price	CPI	Unemployment	Week	IsHolidayInt	Store_2	Store_3	Store_4	Store_5	...	Store_36	Store_37	Store_38	Store_39	Store_40	Store_41	Store_42	Store_43	Store_44	Store_45
1288	49.96	2.828	126.496258	9.765	1.0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1115	65.19	3.891	225.062571	5.679	114.0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
582	65.30	2.808	211.038853	6.465	10.0	0	0	0	0	1	...	0	0	0	0	0	0	0	0	0	0
3647	56.94	3.851	135.265267	7.818	72.0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1564	86.49	3.638	225.829306	6.334	134.0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 50 columns

wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear.fit(XTrain, YTrain)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
#Performance on the test data sets
YHatTest = wmLinear.predict(XTest)
plt.plot(YTest, YHatTest,'ro')
plt.plot(YTest, YTest,'b-')
plt.show()

# calculate the accuray of the model by sum of Square and mean absolute prediction error
MAPE = np.mean(abs((YTest - YHatTest)/YTest))
MSSE = np.mean(np.square(YHatTest - YTest))

print(MAPE, MSSE)
0.08956881121002615 26699670214.63537
# Dimensionality Reduction
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
alphas = np.linspace(10, 20, 10)
testError = np.empty(10)

for i, alpha in enumerate(alphas) :
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(XTrain, YTrain)
    testError[i] = mean_squared_error(YTest, lasso.predict(XTest))
plt.plot(alphas, testError, 'r-')
plt.show()

wmLinear = linear_model.LinearRegression(normalize=True)
wmLinear
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
lasso = Lasso(alpha=17)
lasso.fit(XTrain, YTrain)
Lasso(alpha=17, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
