from src.utils.helpers import label_encoder,one_hot_encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,VotingRegressor
import joblib
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
    target_summary_with_cat(df, "DELIVERY_TIME", col)

cat_cols = [col for col in cat_cols if "DELIVERY_TIME" not in col]


# Label encod
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
    # LabelEncoder nesnesi oluşturulur
    labelencoder = LabelEncoder()

    # İkili sınıfların etiket kodlaması yapılır
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])

    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]
for col in binary_cols:
    label_encoder(df, col)


# one-hot encod
df = one_hot_encoder(df, cat_cols, drop_first=True)


df = df.replace({True: 1, False: 0})


# Son güncel değişken türlerini tutma
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if "DELIVERY_TIME" not in col]

for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))





df.head()


train_df = df[df['DELIVERY_TIME'].notnull()]
test_df = df[df['DELIVERY_TIME'].isnull()]


y = train_df["DELIVERY_TIME"]
X = train_df.drop(["DELIVERY_TIME"], axis=1)

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
 r2_score:  0.63
 MAE:   4.65
 MSE:   33.78
 rmse:  5.81
=========================
Ridge
 r2_score:  0.63
 MAE:   4.65
 MSE:   33.78
 rmse:  5.81
=========================
Random Forest
 r2_score:  0.83
 MAE:   3.13
 MSE:   15.23
 rmse:  3.90
=========================
Gradient Boosting
 r2_score:  0.78
 MAE:   3.60
 MSE:   20.13
 rmse:  4.49
=========================
XGBRegressor
 r2_score:  0.82
 MAE:   3.25
 MSE:   16.33
 rmse:  4.04
=========================
Decision Tree
 r2_score:  0.69
 MAE:   4.06
 MSE:   28.22
 rmse:  5.31
=========================
SVM
 r2_score:  0.28
 MAE:   6.44
 MSE:   64.88
 rmse:  8.05
=========================
"""


def voting_regressor(best_models, X, y):
    print("Voting Regressor...")

    voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                             ('GB', best_models["GB"]),
                                             ('XGB', best_models["XGB"])],
                                 ).fit(X, y)

    cv_results = cross_validate(voting_reg, X, y, cv=3, scoring=["r2", "neg_mean_squared_error", "explained_variance"])
    print(f"R2 Score: {cv_results['test_r2'].mean()}")
    print(f"Negative Mean Squared Error: {cv_results['test_neg_mean_squared_error'].mean()}")
    print(f"Explained Variance: {cv_results['test_explained_variance'].mean()}")
    return voting_reg

voting_reg = voting_regressor(best_models, X, y)


"""
R2 Score: 0.8386648728630458
Negative Mean Squared Error: -14.168833291754368
Explained Variance: 0.8387165629409014
"""


# #rastgele seçilen kullanıcı için teslimat tahmini
X.columns
random_user = X.sample(1, random_state=45)
voting_reg.predict(random_user)

#modeli kaydetme
joblib.dump(voting_reg, "voting_clf3.pkl")

new_model = joblib.load("voting_clf3.pkl")
new_model.predict(random_user)

