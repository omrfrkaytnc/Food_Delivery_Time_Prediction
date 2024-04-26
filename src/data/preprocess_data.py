from src.utils.helpers import check_df,grab_col_names,cat_summary,num_summary,outlier_thresholds,check_outlier,missing_values_table,quick_missing_imp,target_summary_with_cat
from src.utils.helpers import replace_with_thresholds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 500)
warnings.simplefilter(action='ignore', category=Warning)


df.head()
df.info()

df.replace({"NaN": np.nan}, regex=True, inplace = True)

## Veri Tiplerini Düzenleme
df['Time_Orderd']=pd.to_timedelta(df['Time_Orderd'])
df['Time_Order_picked']=pd.to_timedelta(df['Time_Order_picked'])
# Yaş değişkenini float'a çevirme
df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(float)

# Rating değişkenini  float'a çevirme
df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)

# Time değişkenini float'a çevirme
df['Time_taken(min)'] = df['Time_taken(min)'].str.extract('(\d+)').astype(float)

# multiple_deliveries değişkenini floata çevirme
df['multiple_deliveries'] = df['multiple_deliveries'].astype('float')

# Order Date değişkenini yıl,ay,gün olarak ayırma
df['Order_Date']=pd.to_datetime(df['Order_Date'])



df.head()
df.tail()
df.info()

### KEŞİFÇİ VERİ ANALİZİ ###

# veriye genel bakış
check_df(df)

# kategorik, kategorik ancak kardinal ve sayısal değişkenleri belirleme
cat_cols, cat_but_car, num_cols = grab_col_names(df)
print("Kategorik değişkenler:")
print(cat_cols)
print("\nNumerik değişkenler:")
print(num_cols)
print("\nKategorik görünümlü kardinal değişkenler:")
print(cat_but_car)


# kategorik değişken analizi
for col in cat_cols:
    cat_summary(df,col)

# numerik değişken analizi
for col in num_cols:
    num_summary(df,col,True)

# target analizi
for col in cat_cols:
    target_summary_with_cat(df,"Time_taken(min)",col)

# korelasyon analizi
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (15, 15)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

### AYKIRI DEĞER ANALİZİ ###

for col in num_cols:
    if col != "Time_taken(min)":
      print(col, check_outlier(df, col))

# 1 ile 5 arasında olmayan puanları filtreleme
drop_index = df[df["Delivery_person_Ratings"] > 5].index
df.drop(drop_index, axis=0, inplace=True)

# 18 yaş altındakilerin kaldırılması
df = df[df['Delivery_person_Age'] >= 18]


# Aykırı değerlerin baskılanması

for col in num_cols:
    if col != "Time_taken(min)":
        replace_with_thresholds(df,col)

# aykırı değer kontrolü

for col in num_cols:
    print(col, check_outlier(df, col))


### EKSİK DEĞER ANALİZİ ###

missing_values_table(df)

# eksik değerleri grafik ile gösterme
msno.bar(df)
plt.show()

# eksik değerleri doldurma
df = quick_missing_imp(df, num_method="median", cat_length=20)

