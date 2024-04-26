from src.utils.helpers import extract_and_expand_city,average_rating_by_city,average_rating_by_traffic,average_rating_by_weather,add_rating_columns,calculate_distance,calculate_preparation_time,time_of_day,check_df,grab_col_names,label_encoder,one_hot_encoder,target_summary_with_cat


# City
df.columns = ['type_of_city' if col == 'City' else col for col in df.columns]




# Delivery_Personel_ID

df['City'] = df['Delivery_person_ID'].apply(extract_and_expand_city)

# multiple_deliveries

df["multiple_deliveries"]=df["multiple_deliveries"].astype('int64')

# Delivery_person_Age

df["Delivery_person_Age"]=df["Delivery_person_Age"].astype('int64')


# Delivery_person_Rating

add_rating_columns(df)

df["Avg_Rating_By_City"].value_counts()
# latitude-longitude
cols=['Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude']
for col in cols:
    df[col]= abs(df[col])
calculate_distance(df)


# Weatherconditions
df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '')

# Order Date
df['year']= df['Order_Date'].dt.year.astype('int64')
df['month']= df['Order_Date'].dt.month.astype('int64')
df['day']= df['Order_Date'].dt.day.astype('int64')


# Time_Order_picked-Time_Orderd
df['Time_Order_picked'] = df['Time_Order_picked'].astype(str)
df['Time_Orderd'] = df['Time_Orderd'].astype(str)

df['Time_Order_picked'] = df['Time_Order_picked'].apply(lambda x: x.split(" ")[-1])
df['Time_Orderd'] = df['Time_Orderd'].apply(lambda x: x.split(" ")[-1])


df['Time_Order_picked_Hour'] = df['Time_Order_picked'].str.split(":", expand=True)[0].astype('int64')
df['Time_Order_picked_Min'] = df['Time_Order_picked'].str.split(":", expand=True)[1].astype('int64')



df['Time_Orderd_Hour']=df['Time_Orderd'].str.split(':',expand=True)[0]

df['Time_Orderd_Min']=df['Time_Orderd'].str.split(':',expand=True)[1]
df['Time_Orderd_Hour']=df['Time_Orderd_Hour'].astype('int64')
df['Time_Orderd_Min']=df['Time_Orderd_Min'].astype('int64')


calculate_preparation_time(df)

# Day

df['day_zone'] = df['Time_Order_picked_Hour'].apply(time_of_day)

df['day_name'] = df['Order_Date'].dt.day_name()





# Columns
df.columns = df.columns.str.upper()

df.rename(columns={'WEATHERCONDITIONS': 'WEATHER',
                   'ROAD_TRAFFIC_DENSITY': 'TRAFFIC_DENSITY',
                   'TYPE_OF_CITY': 'CITY_TYPE',
                   'TIME_TAKEN(MIN)': 'DELIVERY_TIME',
                   'AVG_RATING_BY_WEATHER': 'WEATHER_RATING',
                   'AVG_RATING_BY_TRAFFIC': 'TRAFFIC_RATING',
                   'AVG_RATING_BY_CITY': 'CITY_RATING',
                   'DAY_ZONE': 'DAY_TIME_ZONE',
                   'PREP_TIME': 'PREPARATION_TIME'}, inplace=True)

# Drop
df.drop(['ID', 'DELIVERY_PERSON_ID', 'ORDER_DATE', 'TIME_ORDERD', 'TIME_ORDER_PICKED'], axis=1, inplace=True)

# Veri setinin son halini kayıt etme
df.to_csv("data/processed/veri_seti_yeni.csv", index=False)



# Data Preprocessing & Feature Engineering
def delivery_data_prep(df):

    ######  Veri tiplerini düzenleme  ######

    df.replace({"NaN": np.nan}, regex=True, inplace=True)

    df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
    df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])

    # Yaş değişkenini float'a çevirme
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype(float)

    # Rating değişkenini  float'a çevirme
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)

    # Time değişkenini float'a çevirme
    df['Time_taken(min)'] = df['Time_taken(min)'].str.extract('(\d+)').astype(float)

    # multiple_deliveries değişkenini floata çevirme
    df['multiple_deliveries'] = df['multiple_deliveries'].astype(float)

    # Order Date değişkenini yıl,ay,gün olarak ayırma
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])


    #### Eksik Değer & Aykırı Değer #####

    drop_index = df[df["Delivery_person_Ratings"] > 5].index
    df.drop(drop_index, axis=0, inplace=True)

    df = df[df['Delivery_person_Age'] >= 18]

    df = quick_missing_imp(df, num_method="median", cat_length=20)

    ##### Feature Engineering ######

    # City
    df.columns = ['type_of_city' if col == 'City' else col for col in df.columns]

    # Delivery_Personel_ID
    df['City'] = df['Delivery_person_ID'].apply(extract_and_expand_city)

    # multiple_deliveries
    df["multiple_deliveries"] = df["multiple_deliveries"].astype('int64')

    # Delivery_person_Age
    df["Delivery_person_Age"] = df["Delivery_person_Age"].astype('int64')

    # Delivery_person_Rating
    add_rating_columns(df)

    # latitude-longitude
    cols = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
    for col in cols:
        df[col] = abs(df[col])

    calculate_distance(df)

    # Weatherconditions
    df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '')

    # Order Date
    df['year'] = df['Order_Date'].dt.year.astype('int64')
    df['month'] = df['Order_Date'].dt.month.astype('int64')
    df['day'] = df['Order_Date'].dt.day.astype('int64')

    # Time_Order_picked-Time_Orderd
    df['Time_Order_picked'] = df['Time_Order_picked'].astype(str)
    df['Time_Orderd'] = df['Time_Orderd'].astype(str)

    df['Time_Order_picked'] = df['Time_Order_picked'].apply(lambda x: x.split(" ")[-1])
    df['Time_Orderd'] = df['Time_Orderd'].apply(lambda x: x.split(" ")[-1])

    df['Time_Order_picked_Hour'] = df['Time_Order_picked'].str.split(":", expand=True)[0].astype('int64')
    df['Time_Order_picked_Min'] = df['Time_Order_picked'].str.split(":", expand=True)[1].astype('int64')

    df['Time_Orderd_Hour'] = df['Time_Orderd'].str.split(':', expand=True)[0].astype('int64')
    df['Time_Orderd_Min'] = df['Time_Orderd'].str.split(':', expand=True)[1].astype('int64')

    calculate_preparation_time(df)

    # Day
    df['day_zone'] = df['Time_Order_picked_Hour'].apply(time_of_day)
    df['day_name'] = df['Order_Date'].dt.day_name()

    # Columns
    df.columns = df.columns.str.upper()

    df.rename(columns={'WEATHERCONDITIONS': 'WEATHER',
                       'ROAD_TRAFFIC_DENSITY': 'TRAFFIC_DENSITY',
                       'TYPE_OF_CITY': 'CITY_TYPE',
                       'TIME_TAKEN(MIN)': 'DELIVERY_TIME',
                       'AVG_RATING_BY_WEATHER': 'WEATHER_RATING',
                       'AVG_RATING_BY_TRAFFIC': 'TRAFFIC_RATING',
                       'AVG_RATING_BY_CITY': 'CITY_RATING',
                       'DAY_ZONE': 'DAY_TIME_ZONE',
                       'PREP_TIME': 'PREPARATION_TIME'}, inplace=True)

    # Drop
    df.drop(['ID', 'DELIVERY_PERSON_ID', 'ORDER_DATE', 'TIME_ORDERD', 'TIME_ORDER_PICKED'], axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    cat_cols = [col for col in cat_cols if "DELIVERY_TIME" not in col]

    binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

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
    for col in binary_cols:
        label_encoder(df, col)

    df = one_hot_encoder(df, cat_cols, drop_first=True)

    df = df.replace({True: 1, False: 0})

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    cat_cols = [col for col in cat_cols if "DELIVERY_TIME" not in col]
    train_df = df[df['DELIVERY_TIME'].notnull()]
    test_df = df[df['DELIVERY_TIME'].isnull()]
    y = df["DELIVERY_TIME"]
    X = df.drop(["DELIVERY_TIME"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

    return X, y


