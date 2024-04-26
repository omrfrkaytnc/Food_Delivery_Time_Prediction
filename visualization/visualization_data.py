import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('data/processed/veri_seti_yeni.csv')

# Hava durumu (WEATHER) ve trafik yoğunluğu (TRAFFIC_DENSITY) dağılımları
plt.figure(figsize=(10, 6))
sns.boxplot(x='WEATHER', y='DELIVERY_TIME', data=df)
plt.title('Delivery Time by Weather')
plt.xlabel('Weather')
plt.ylabel('Delivery Time')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='TRAFFIC_DENSITY', y='DELIVERY_TIME', data=df)
plt.title('Delivery Time by Traffic Density')
plt.xlabel('Traffic Density')
plt.ylabel('Delivery Time')
plt.show()


# Teslimat süresi ile hava durumu arasındaki ilişki
plt.figure(figsize=(10, 6))
sns.boxplot(x='WEATHER', y='DELIVERY_TIME', data=df)
plt.title('Delivery Time by Weather')
plt.xlabel('Weather')
plt.ylabel('Delivery Time')
plt.show()



# Teslimat süresi ile diğer değişkenler arasındaki ilişkiyi gösteren scatter plotlar
# Mesafeyi 5 eşit parçaya bölmek
df['DISTANCE_GROUP'], distance_bins = pd.cut(df['DISTANCE'], bins=5, labels=False, retbins=True)

# Her mesafe grubunun başlangıç ve bitiş değerlerini hesapla
distance_labels = [f'{int(distance_bins[i]):}-{int(distance_bins[i+1]):}' for i in range(len(distance_bins)-1)]

# Her mesafe grubunun ortalama teslimat süresini hesapla
distance_delivery_mean = df.groupby('DISTANCE_GROUP')['DELIVERY_TIME'].mean()

# Sütun grafiği oluştur
plt.figure(figsize=(10, 6))
distance_delivery_mean.plot(kind='bar', color='skyblue')
plt.title('Average Delivery Time by Distance Group')
plt.xlabel('Distance Group')
plt.ylabel('Average Delivery Time')
plt.xticks(range(len(distance_labels)), distance_labels, rotation=45)
plt.show()






# Feature İmportance
def plot_importance(model, features, num=5, save=False):
    if hasattr(model, 'feature_importances_'):
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
        feature_imp = feature_imp.head(num)  # Sadece ilk "num" değişkeni al
        plt.figure(figsize=(15, 15))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title('Feature Importances (Top {})'.format(num))
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig('importances.png')
    else:
        print("Model doesn't have attribute 'feature_importances_'.")

def evaluate_models(X_train, X_test, y_train, y_test):
    def scores(y_test, p):
        r2 = r2_score(y_test, p)
        MAE = mean_absolute_error(y_test, p)
        MSE = mean_squared_error(y_test, p)
        rmse = np.sqrt(mean_squared_error(y_test, p))
        print(' r2_score:  {:.2f}'.format(r2))
        print(' MAE:   {:.2f}'.format(MAE))
        print(' MSE:   {:.2f}'.format(MSE))
        print(' rmse:  {:.2f}'.format(rmse))
        print("=========================")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBRegressor": XGBRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "SVM": SVR()
    }

    for model_name, model_instance in models.items():
        model_instance.fit(X_train, y_train)
        print(model_name)
        p = model_instance.predict(X_test)
        scores(y_test, p)
        if model_name == "Random Forest":  # Sadece Random Forest için önemli özellikleri çiz
            plot_importance(model_instance, features=X_train, num=5, save=True)

# Fonksiyonun çağrılması
evaluate_models(X_train, X_test, y_train, y_test)




