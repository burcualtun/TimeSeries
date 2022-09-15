
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


#####################################################################################
#############################GÖREV1####################################

#Adım1
#Iyzico_data.csv dosyasını okutunuz. transaction_date değişkeninin tipini date'e çeviriniz.
df= pd.read_csv('AllMiul/W10/HW/data.csv', parse_dates=['transaction_date'])

#Adım2 - Veri setinin başlangıc ve bitiş tarihleri nedir?
#2018-01-01
df['transaction_date'].min()
#2020-12-31
df['transaction_date'].max()

#Adım3 - Her üye işyerindeki toplam işlem sayısı kaçtır?

df.groupby(['merchant_id']).agg({"Total_Transaction": ["sum"]})

#Adım4 - Her üye işyerindeki toplam ödeme miktarı kaçtır?

df.groupby(['merchant_id']).agg({"Total_Paid": ["sum"]})

#Adım5 - Her üye işyerininin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.

import seaborn as sns
import matplotlib.pyplot as plt
df['year']=df['transaction_date'].dt.year

df.groupby(['merchant_id','year']).agg({"Total_Transaction": ["sum"]}).plot(kind='bar')

#olması gereken chart

for id in df["merchant_id"].unique():
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1, title= str(id) + '2018-2019 Transaction Count')
    df[(df.merchant_id == id) & ( df.transaction_date >= "2018-01-01") & (df.transaction_date < "2019-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.subplot(3, 1, 2, title= str(id) + '2019-2020 Transaction Count')
    df[(df.merchant_id == id) & (df.transaction_date >= "2019-01-01") & (df.transaction_date < "2020-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.show()

df.head()

################################################Görev2#######################################
#Adım1 - Feature Engineering teknikleriniuygulayanız. Yeni feature'lartüretiniz.


#Date Features

def create_date_features(df):
    df['month'] = df.transaction_date.dt.month
    df['day_of_month'] = df.transaction_date.dt.day
    df['day_of_year'] = df.transaction_date.dt.dayofyear
    df['week_of_year'] = df.transaction_date.dt.weekofyear
    df['day_of_week'] = df.transaction_date.dt.dayofweek
    df['year'] = df.transaction_date.dt.year
    df["is_wknd"] = df.transaction_date.dt.weekday // 4
    df['is_month_start'] = df.transaction_date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.transaction_date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head()

#Lag/Shifted Features

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags , var):
    for lag in lags:
        dataframe[var+'_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])[var].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728],'Total_Transaction')
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728],'Total_Paid')

# Rolling Mean Features

def roll_mean_features(dataframe, windows,var):
    for window in windows:
        dataframe[var+'_roll_mean_' + str(window)] = dataframe.groupby(["merchant_id"])[var]. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [182,365, 546,728],'Total_Transaction')
df = roll_mean_features(df, [182,365, 546,728],'Total_Paid')

# Exponentially Weighted Mean Features
def ewm_features(dataframe, alphas, lags ,var):
    for alpha in alphas:
        for lag in lags:
            dataframe[var+'_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["merchant_id"])[var].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags,'Total_Transaction')
df = ewm_features(df, alphas, lags,'Total_Paid')

# Özel günler, döviz kuru vb.

df["is_black_friday"] = 0
df.loc[df["transaction_date"].isin(["2018-11-22","2018-11-23","2019-11-29","2019-11-30"]) ,"is_black_friday"]=1

# yazın başlangıcı
df["is_summer_solstice"] = 0
df.loc[df["transaction_date"].isin(["2018-06-19","2018-06-20","2018-06-21","2018-06-22",
                                    "2019-06-19","2019-06-20","2019-06-21","2019-06-22",]) ,"is_summer_solstice"]=1

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

############################Görev 3 :  Modellemeye Hazırlık ve Modelleme################################
#Adım1 - One-hot encoding yapınız.

df = pd.get_dummies(df, columns=[ 'merchant_id', 'day_of_week', 'month','Total_Paid'])
check_df(df)


#Adım2 - Custom Cost Function'ları tanımlayınız

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

#Adım3 - Veri setinitrain vevalidation olarakayırınız.

df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)

# 2020'nin başına kadar (2019'nın sonuna kadar) train seti.
train = df.loc[(df["transaction_date"] < "2020-01-01"), :]

# 2020'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["transaction_date"] >= "2020-01-01") & (df["transaction_date"] < "2020-04-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date',  "Total_Transaction", "year","Unnamed: 0"]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

#Adım4 - LightGBM Modelini oluşturunuz ve SMAPE ile hata değerini gözlemleyiniz.


lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              #iterasyon ve optimizasyon sayısıdır.
              'num_boost_round': 15000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=50)

#Final model

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)