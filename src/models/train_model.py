from src.utils.helpers import label_encoder,one_hot_encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Feature Engineering Öncesi Model Sonuçlarını Değerlendirme

dff = df.copy()
dff.drop(["ID", "Delivery_person_ID"], axis=1, inplace=True)
dff.drop(["Order_Date","Time_Orderd","Time_Order_picked"],axis=1, inplace=True)


cat_cols, cat_but_car, num_cols = grab_col_names(dff)

cat_cols = [col for col in cat_cols if col not in ["Time_taken(min)"]]
cat_cols


# encod işlemleri

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]


for col in binary_cols:
    label_encoder(dff, col)

cat_cols, cat_but_car, num_cols = grab_col_names(dff)


dff = one_hot_encoder(dff, cat_cols, drop_first=True)
dff = dff.replace({True: 1, False: 0})
dff.head()



train_df = dff[dff['Time_taken(min)'].notnull()]
test_df = dff[dff['Time_taken(min)'].isnull()]


y = train_df["Time_taken(min)"]
X = train_df.drop(["Time_taken(min)"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
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

# Fonksiyonun çağrılması
evaluate_models(X_train, X_test, y_train, y_test)



"""
linear Regression
 r2_score:  0.62
 MAE:   4.65
 MSE:   33.83
 rmse:  5.82
=========================
Ridge
 r2_score:  0.62
 MAE:   4.65
 MSE:   33.84
 rmse:  5.82
=========================
Random Forest
 r2_score:  0.74
 MAE:   3.78
 MSE:   23.61
 rmse:  4.86
=========================
Gradient Boosting
 r2_score:  0.70
 MAE:   4.14
 MSE:   27.40
 rmse:  5.23
=========================
XGBRegressor
 r2_score:  0.77
 MAE:   3.65
 MSE:   20.93
 rmse:  4.58
=========================
Decision Tree
 r2_score:  0.52
 MAE:   4.91
 MSE:   43.08
 rmse:  6.56
=========================
SVM
 r2_score:  0.17
 MAE:   6.87
 MSE:   74.69
 rmse:  8.64
=========================




"""













