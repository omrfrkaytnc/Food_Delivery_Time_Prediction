import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    # Sadece sayısal sütunları seçme
    numeric_columns = dataframe.select_dtypes(include=np.number)
    print("##################### Quantiles #####################")
    print(numeric_columns.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



def grab_col_names(dataframe, cat_th=10, car_th=25):
    """
    Verilen veri çerçevesi için sütun isimlerini alır ve kategorik, kategorik ancak kardinal ve sayısal değişkenleri belirler.

    Parametreler:
    dataframe (pandas.DataFrame): Sütun isimlerinin alınacağı veri çerçevesi.
    cat_th (int, optional): Kategorik değişken olarak kabul edilecek eşik değer. Varsayılan değer 10.
    car_th (int, optional): Kategorik ancak kardinal değişken olarak kabul edilecek eşik değer. Varsayılan değer 20.

    Returns:
    tuple: Kategorik değişkenlerin, kategorik ancak kardinal değişkenlerin ve sayısal değişkenlerin listelerini içeren bir tuple.

    Örnek:
    cat_cols, cat_but_car, num_cols = grab_col_names(dataframe)
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols


def cat_summary(dataframe, col_name, plot=False):
    """
    Verilen kategorik bir değişkenin frekans tablosunu ve oranlarını yazdırır. Opsiyonel olarak, bu tabloyu bir grafik
    ile görselleştirebilir.

    Parametreler:
    dataframe (pandas.DataFrame): Kategorik değişkenin bulunduğu veri çerçevesi.
    col_name (str): Frekans tablosunun ve oranlarının alınacağı kategorik değişkenin adı.
    plot (bool, optional): Grafik gösterilsin mi? Varsayılan değer False.

    Returns:
    None

    Örnek:
    cat_summary(dataframe, "column_name", plot=True)
    """

    # Kategorik değişkenin frekans tablosunu ve oranlarını oluşturur
    summary = pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio (%)": 100 * dataframe[col_name].value_counts() / len(dataframe)})

    # Frekans tablosunu yazdırır
    print(summary)

    # Opsiyonel olarak, grafik gösterir
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    """
    Verilen sayısal bir değişken için temel istatistiklerin özetini yazdırır ve isteğe bağlı olarak bir histogram görseli oluşturur.

    Parametreler:
    dataframe (pandas.DataFrame): Sayısal değişkenin bulunduğu veri çerçevesi.
    numerical_col (str): Temel istatistiklerin alınacağı sayısal değişkenin adı.
    plot (bool, optional): Histogram gösterilsin mi? Varsayılan değer False.

    Returns:
    None

    Örnek:
    num_summary(dataframe, "column_name", plot=True)
    """

    # Sayısal değişken için temel istatistiklerin özetini oluşturur ve yazdırır
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    # Opsiyonel olarak, histogram görseli oluşturur ve gösterir
    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")



def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    """
    Verilen sayısal bir değişken için alt ve üst sınır eşik değerlerini hesaplar.

    Parametreler:
    dataframe (pandas.DataFrame): Sınır eşik değerlerinin hesaplanacağı veri çerçevesi.
    variable (str): Eşik değerlerinin hesaplanacağı sayısal değişkenin adı.
    low_quantile (float, optional): Alt sınır eşik değeri için kullanılacak çeyreklik. Varsayılan değer 0.05.
    up_quantile (float, optional): Üst sınır eşik değeri için kullanılacak çeyreklik. Varsayılan değer 0.95.

    Returns:
    tuple: Alt ve üst sınır eşik değerleri.

    Örnek:
    low_limit, up_limit = outlier_thresholds(dataframe, "column_name")
    """
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    Verilen sayısal bir değişken için aykırı değer kontrolü yapar.

    Parametreler:
    dataframe (pandas.DataFrame): Aykırı değer kontrolü yapılacak veri çerçevesi.
    col_name (str): Aykırı değer kontrolü yapılacak sayısal değişkenin adı.

    Returns:
    bool: Değişkenin aykırı değer içerip içermediği durumunu döndürür.

    Örnek:
    is_outlier = check_outlier(dataframe, "column_name")
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if (dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)] != pd.Timestamp(0)).any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    """
    Verilen bir DataFrame içinde belirtilen değişkenin outlier değerlerini alt ve üst sınırlarla değiştirir.

    Parametreler:
    dataframe (pandas.DataFrame): Outlier değerlerin bulunduğu DataFrame.
    variable (str): Outlier değerlerinin kontrol edileceği değişkenin adı.

    Returns:
    None
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit






def missing_values_table(dataframe, na_name=False):
    """
    Verilen veri çerçevesindeki eksik değerleri analiz eder ve eksik değer tablosunu yazdırır.

    :param dataframe: pandas.DataFrame, Eksik değerlerin analiz edileceği veri çerçevesi.
    :param na_name: bool, Opsiyonel, Eksik değer bulunan sütunların isimlerini döndürsün mü? Varsayılan değer False.

    :return: list, Eksik değer bulunan sütunların isimleri (opsiyonel, na_name=True durumunda).

    Örnek:
    missing_values_table(dataframe)
    missing_values_table(dataframe, na_name=True)
    """
    # Eksik değer bulunan sütunların listesini oluşturur
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    # Her sütundaki eksik değerlerin sayısını hesaplar ve büyükten küçüğe sıralar
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    # Her sütundaki eksik değerlerin oranını hesaplar ve büyükten küçüğe sıralar
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    # Eksik değer tablosunu oluşturur
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    # Eksik değer tablosunu yazdırır
    print(missing_df, end="\n")

    # Opsiyonel olarak, eksik değer bulunan sütunların isimlerini döndürür
    if na_name:
        return na_columns


def quick_missing_imp(data, num_method="median", cat_length=20, target="Time_taken(min)"):
    """
    Verilen veri çerçevesindeki eksik değerleri hızlı bir şekilde doldurur.

    Parametreler:
    data (pandas.DataFrame): Eksik değerlerin doldurulacağı veri çerçevesi.
    num_method (str, optional): Sayısal değişkenler için kullanılacak doldurma yöntemi. "mean" veya "median" olabilir.
                                Varsayılan değer "median".
    cat_length (int, optional): Kategorik değişken olarak kabul edilecek sınıf sayısı. Varsayılan değer 20.
    target (str, optional): Hedef değişkenin adı. Varsayılan değer "Time_taken(min)".

    Returns:
    pandas.DataFrame: Eksik değerlerin doldurulmuş hali.

    Örnek:
    df = quick_missing_imp(dataframe, num_method="median", cat_length=20)
    """
    # Eksik değerlere sahip değişkenlerin listesi oluşturulur
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]

    # Hedef değişken geçici olarak saklanır
    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # Kategorik değişkenler için eksik değerler mode ile doldurulur (eğer sınıf sayısı belirli bir eşik değerden düşük veya eşitse)
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # Sayısal değişkenler için eksik değerler num_method parametresine göre doldurulur
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    # Hedef değişken eski haline getirilir
    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data













def label_encoder(dataframe, binary_col):
    """
    Verilen kategorik bir değişkenin ikili sınıflarını 0 ve 1'e dönüştürür.

    Parametreler:
    dataframe (pandas.DataFrame): Etiket kodlama yapılacak veri çerçevesi.
    binary_col (str): İkili sınıflarına dönüştürülecek kategorik değişkenin adı.

    Returns:
    pandas.DataFrame: Etiket kodlaması yapılmış veri çerçevesi.

    Örnek:
    dataframe = label_encoder(dataframe, "binary_column")
    """


    labelencoder = LabelEncoder()

    # İkili sınıfların etiket kodlaması yapılır
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])

    return dataframe







def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Verilen kategorik değişkenleri one-hot kodlama ile dönüştürür.

    Parametreler:
    dataframe (pandas.DataFrame): One-hot kodlama yapılacak veri çerçevesi.
    categorical_cols (list): One-hot kodlama yapılacak kategorik değişkenlerin adlarını içeren liste.
    drop_first (bool, optional): İlk sütunun düşürülüp düşürülmeyeceği. Varsayılan değer False.

    Returns:
    pandas.DataFrame: One-hot kodlaması yapılmış veri çerçevesi.

    Örnek:
    dataframe = one_hot_encoder(dataframe, ["cat_column1", "cat_column2"], drop_first=True)
    """
    # Verilen kategorik değişkenleri one-hot kodlama ile dönüştürür
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

    return dataframe



def extract_and_expand_city(id):
    """
    Bu fonksiyon, bir teslimat personeli kimliğinden şehrin adını çıkarır ve uzun haliyle değiştirir.

    Parametreler:
        id: Teslimat personeli kimliği.

    Dönüş Değeri:
        Şehrin uzun hali.
    """
    res_index = id.find('RES')  # 'RES' kelimesinin indeksini bulun
    if res_index != -1:  # Eğer 'RES' kelimesi bulunduysa
        short_name = id[:res_index]  # 'RES' kelimesinin öncesini (şehir adını) alın
        city_names = {
            'JAP': 'Japura',
            'COIMB': 'Coimbatore',
            'INDO': 'Indore',
            'SUR': 'Surat',
            'CHEN': 'Chennai',
            'RANCHI': 'Ranchi',
            'MYS': 'Mysore',
            'PUNE': 'Pune',
            'HYD': 'Hyderabad',
            'MUM': 'Mumbai',
            'VAD': 'Vadodara',
            'BANG': 'Bangalore',
            'LUDH': 'Ludhiana',
            'KNP': 'Kanpur',
            'AGR': 'Agra',
            'ALH': 'Allahabad',
            'DEH': 'Dehradun',
            'KOC': 'Kochi',
            'AURG': 'Aurangabad',
            'BHP': 'Bhopal',
            'GOA': 'Goa',
            'KOL': 'Kolkata'
        }
        return city_names.get(short_name, None)  # Şehrin uzun halini döndür
    else:
        return None



def average_rating_by_weather(df):
    """
    Farklı hava koşulları altında ortalama teslimat personeli puanlarını hesaplar.

    Parametreler:
    - df (DataFrame): 'Weatherconditions' adında farklı hava koşullarını temsil eden bir sütun ve
                      'Delivery_person_Ratings' adında teslimat personeli tarafından verilen puanları
                      temsil eden bir sütun içeren Pandas DataFrame.

    Dönüşler:
    - avg_ratings_by_weather (dict): Anahtarları farklı hava koşulları olan ve her bir koşul için
                                      ortalama puanları temsil eden bir sözlük.
    """
    weather_conditions = df['Weatherconditions'].unique()
    avg_ratings_by_weather = {}
    for condition in weather_conditions:
        avg_rating = df[df['Weatherconditions'] == condition]['Delivery_person_Ratings'].mean()
        avg_ratings_by_weather[condition] = avg_rating
    return avg_ratings_by_weather



def average_rating_by_traffic(df):
    """
    Farklı yol trafik yoğunlukları altında ortalama teslimat personeli puanlarını hesaplar.

    Parametreler:
    - df (DataFrame): 'Road_traffic_density' adında farklı yol trafik yoğunluklarını temsil eden bir sütun ve
                      'Delivery_person_Ratings' adında teslimat personeli tarafından verilen puanları
                      temsil eden bir sütun içeren Pandas DataFrame.

    Dönüşler:
    - avg_ratings_by_traffic (dict): Anahtarları farklı trafik yoğunlukları olan ve her bir yoğunluk için
                                     ortalama puanları temsil eden bir sözlük.
    """
    traffic_conditions = df['Road_traffic_density'].unique()
    avg_ratings_by_traffic = {}
    for condition in traffic_conditions:
        avg_rating = df[df['Road_traffic_density'] == condition]['Delivery_person_Ratings'].mean()
        avg_ratings_by_traffic[condition] = avg_rating
    return avg_ratings_by_traffic




def average_rating_by_city(df):
    """
    Farklı şehirlerde ortalama teslimat personeli puanlarını hesaplar.

    Parametreler:
    - df (DataFrame): 'City' adında farklı şehirleri temsil eden bir sütun ve
                      'Delivery_person_Ratings' adında teslimat personeli tarafından verilen puanları
                      temsil eden bir sütun içeren Pandas DataFrame.

    Dönüşler:
    - avg_ratings_by_city (dict): Anahtarları farklı şehirler olan ve her bir şehir için
                                   ortalama puanları temsil eden bir sözlük.
    """
    cities = df['City'].unique()
    avg_ratings_by_city = {}
    for city in cities:
        avg_rating = df[df['City'] == city]['Delivery_person_Ratings'].mean()
        avg_ratings_by_city[city] = avg_rating
    return avg_ratings_by_city





def add_rating_columns(df):
    """
    DataFrame'e üç yeni sütun ekler: hava koşullarına göre ortalama puanlar, trafik yoğunluğuna göre
    ortalama puanlar ve şehre göre ortalama puanlar.

    Parametreler:
    - df (DataFrame): 'Weatherconditions' sütunuyla farklı hava koşullarını, 'Road_traffic_density'
                      sütunuyla farklı trafik yoğunluklarını ve 'City' sütunuyla farklı şehirleri içeren
                      bir Pandas DataFrame.

    Dönüşler:
    - df (DataFrame): Yeni eklenmiş sütunlarla güncellenmiş DataFrame.
    """
    # Hava koşullarına göre ortalama puanlar
    avg_ratings_weather = average_rating_by_weather(df)
    df['Avg_Rating_By_Weather'] = df['Weatherconditions'].map(avg_ratings_weather)

    # Trafik yoğunluğuna göre ortalama puanlar
    avg_ratings_traffic = average_rating_by_traffic(df)
    df['Avg_Rating_By_Traffic'] = df['Road_traffic_density'].map(avg_ratings_traffic)

    # Şehre göre ortalama puanlar
    avg_ratings_city = average_rating_by_city(df)
    df['Avg_Rating_By_City'] = df['City'].map(avg_ratings_city)

    return df




def calculate_distance(df):
    """
    Verilen bir veri çerçevesindeki restoran ve teslimat lokasyonları arasındaki mesafeyi hesaplar ve yeni bir sütun oluşturur.

    Parametreler:
    - df (DataFrame): Restoran ve teslimat lokasyonlarının enlem ve boylamını içeren bir veri çerçevesi.

    Dönüşler:
    - None: Fonksiyon, verilen DataFrame'i günceller ve 'Distance' adında yeni bir sütun ekler,
             bu sütun, her bir satır için restoran ile teslimat lokasyonu arasındaki mesafeyi içerir.
    """
    distances = []
    for index, row in df.iterrows():
        restaurant_coords = (row['Restaurant_latitude'], row['Restaurant_longitude'])
        delivery_coords = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
        distance = geodesic(restaurant_coords, delivery_coords).kilometers
        distance_rounded = round(distance, 2)
        distances.append(distance_rounded)
    df['Distance'] = distances




def calculate_preparation_time(df):
    """
    Verilen bir veri çerçevesindeki siparişlerin hazırlanma süresini hesaplar ve yeni bir sütun oluşturur.

    Parametreler:
    - df (DataFrame): 'Time_Orderd' ve 'Time_Order_picked' sütunlarını içeren bir veri çerçevesi.
                      'Time_Orderd', siparişin verildiği zamanı ve 'Time_Order_picked', siparişin hazır
                      olduğu zamanı içerir.

    Dönüşler:
    - None: Fonksiyon, verilen DataFrame'i günceller ve 'prep_time' adında yeni bir sütun ekler.
             Bu sütun, her bir satır için siparişin hazırlanma süresini dakika cinsinden içerir.
    """
    # 'Time_Orderd' ve 'Time_Order_picked' sütunlarını timedelta olarak değiştirin
    df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
    df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])

    # Teslimat hazırlanma süresini hesaplayın
    df['prep_time'] = np.where(df['Time_Order_picked'] < df['Time_Orderd'],
                                              df['Time_Order_picked'] - df['Time_Orderd'] + pd.Timedelta(days=1),
                                              df['Time_Order_picked'] - df['Time_Orderd'])

    # Teslimat hazırlanma süresini dakika cinsine dönüştürün
    df['prep_time'] = (df['prep_time'].dt.total_seconds() / 60).astype('int64')




def time_of_day(x):
    """
    Verilen bir saat değerine göre günün hangi zaman dilimine denk geldiğini belirler.

    Parametreler:
    - x (int): Saati temsil eden bir tamsayı değeri. Örneğin, 0-23 arası bir saat değeri beklenir.

    Dönüşler:
    - str: Saatin ait olduğu zaman dilimini ifade eden bir metin.
           Örneğin, 'Morning', 'Afternoon', 'Evening', 'Night' veya 'Midnight'.
    """
    if x in [4, 5, 6, 7, 8, 9, 10]:
        return "Morning"
    elif x in [11, 12, 13, 14, 15]:
        return "Afternoon"
    elif x in [16, 17, 18, 19]:
        return "Evening"
    elif x in [20, 21, 22, 23]:
        return "Night"
    else:
        return "Midnight"

