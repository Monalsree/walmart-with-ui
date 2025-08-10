import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import pickle
from os import path
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor

# Create required directories
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

# Load datasets
data = pd.read_csv(r'D:\walmart forecast\train.csv')
stores = pd.read_csv(r'D:\walmart forecast\stores.csv')
features = pd.read_csv(r'D:\walmart forecast\features.csv')

# Display dataset info
print(data.shape)
print(data.tail())
print(data.info())
print(stores.shape)
print(stores.tail())
print(stores.info())
print(features.shape)
print(features.tail())
print(features.info())

# Handle missing values in features
features["CPI"] = features["CPI"].fillna(features["CPI"].median())
features["Unemployment"] = features["Unemployment"].fillna(features["Unemployment"].median())
for i in range(1, 6):
    features["MarkDown"+str(i)] = features["MarkDown"+str(i)].apply(lambda x: 0 if x < 0 else x)
    features["MarkDown"+str(i)] = features["MarkDown"+str(i)].fillna(value=0)

# Merge datasets
data = pd.merge(data, stores, on='Store', how='left')
data = pd.merge(data, features, on=['Store', 'Date'], how='left')
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by=['Date'], inplace=True)
data.set_index('Date', inplace=True)  # Set Date as DatetimeIndex
data.drop(columns='IsHoliday_x', inplace=True, errors='ignore')
data.rename(columns={"IsHoliday_y": "IsHoliday"}, inplace=True)

# Extract date features
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Week'] = data.index.isocalendar().week

# Aggregate data
agg_data = data.groupby(['Store', 'Dept']).Weekly_Sales.agg(['max', 'min', 'mean', 'median', 'std']).reset_index()
print(agg_data.isnull().sum())
store_data = pd.merge(left=data.reset_index(), right=agg_data, on=['Store', 'Dept'], how='left')
store_data.dropna(inplace=True)
data = store_data.copy()
del store_data

# Ensure Date column is set as DatetimeIndex again after merging
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Create Total_MarkDown and drop individual markdowns
data['Total_MarkDown'] = data['MarkDown1'] + data['MarkDown2'] + data['MarkDown3'] + data['MarkDown4'] + data['MarkDown5']
data.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1, inplace=True)

# Remove outliers
numeric_col = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Total_MarkDown']
data_numeric = data[numeric_col].copy()
data = data[(np.abs(stats.zscore(data_numeric)) < 2.5).all(axis=1)]

# Plot negative weekly sales
y = data["Weekly_Sales"][data.Weekly_Sales < 0]
sns.displot(y, height=6, aspect=2)
plt.title("Negative Weekly Sales", fontsize=15)
plt.savefig('plots/negative_weekly_sales.png')
plt.show()

# Remove negative weekly sales
data = data[data['Weekly_Sales'] >= 0]

# Convert IsHoliday to int
data['IsHoliday'] = data['IsHoliday'].astype(int)

# Save preprocessed data
data.to_csv('./datasets/preprocessed_walmart_dataset.csv')

# Visualizations
plt.figure(figsize=(14,8))
sns.barplot(x='Month',y='Weekly_Sales',data=data)
plt.ylabel('Sales',fontsize=14)
plt.xlabel('Months',fontsize=14)
plt.title('Average Monthly Sales',fontsize=16)
plt.savefig('plots/avg_monthly_sales.png')
plt.grid()

data_monthly = pd.crosstab(data["Year"], data["Month"], values=data["Weekly_Sales"], aggfunc='sum')
fig, axes = plt.subplots(3, 4, figsize=(16, 8))
plt.suptitle('Monthly Sales for each Year', fontsize=18)
k = 1
for i in range(3):
    for j in range(4):
        if k <= 12:
            sns.lineplot(ax=axes[i, j], data=data_monthly[k])
            axes[i, j].set_ylabel(f'Month {k}', fontsize=12)
            axes[i, j].set_xlabel('Years', fontsize=12)
            k += 1
plt.subplots_adjust(wspace=0.4, hspace=0.32)
plt.savefig('plots/monthly_sales_every_year.png')
plt.show()

plt.figure(figsize=(20, 8))
sns.barplot(x='Store', y='Weekly_Sales', data=data)
plt.grid()
plt.title('Average Sales per Store', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.savefig('plots/avg_sales_store.png')
plt.show()

plt.figure(figsize=(20,8))
sns.barplot(x='Dept',y='Weekly_Sales',data=data)
plt.grid()
plt.title('Average Sales per Department', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Department', fontsize=16)
plt.savefig('plots/avg_sales_dept.png')
plt.show()

plt.figure(figsize=(10, 8))
sns.histplot(data['Temperature'], kde=True)
plt.title('Effect of Temperature', fontsize=15)
plt.xlabel('Temperature', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.savefig('plots/effect_of_temp.png')
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(data['IsHoliday'].value_counts(), labels=['No Holiday', 'Holiday'], autopct='%0.2f%%')
plt.title("Pie chart distribution", fontsize=14)
plt.legend()
plt.savefig('plots/holiday_distribution.png')
plt.show()

# Ensure DatetimeIndex for seasonal decomposition
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)
sm.tsa.seasonal_decompose(data['Weekly_Sales'].resample('MS').mean(), model='additive').plot()
plt.savefig('plots/seasonal_decompose.png')
plt.show()

# One-hot encode categorical columns
cat_col = ['Store', 'Dept', 'Type']
data_cat = data[cat_col].copy()
data_cat = pd.get_dummies(data_cat, columns=cat_col)
data = pd.concat([data, data_cat], axis=1)
data.drop(columns=cat_col, inplace=True)
data.drop(columns=['Date'], inplace=True, errors='ignore')

# Normalize numerical columns
num_col = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Total_MarkDown', 'max', 'min', 'mean', 'median', 'std']
minmax_scale = MinMaxScaler(feature_range=(0, 1))

def normalization(df, col):
    for i in col:
        arr = df[i]
        arr = np.array(arr).reshape(-1, 1)
        df[i] = minmax_scale.fit_transform(arr)
    return df

data = normalization(data.copy(), num_col)

# Correlation matrix
plt.figure(figsize=(15, 8))
corr = data[num_col].corr()
sns.heatmap(corr, vmax=1.0, annot=True)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig('plots/correlation_matrix.png')
plt.show()

# Feature selection
feature_col = data.columns.difference(['Weekly_Sales'])
radm_clf = RandomForestRegressor(oob_score=True, n_estimators=23)
radm_clf.fit(data[feature_col], data['Weekly_Sales'])

pkl_filename = "./models/feature_elim_regressor.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(radm_clf, file)
    print("Saved model to disk")
else:
    print("Model already saved")

# Feature importance
indices = np.argsort(radm_clf.feature_importances_)[::-1]
feature_rank = pd.DataFrame(columns=['rank', 'feature', 'importance'])
for f in range(data[feature_col].shape[1]):
    feature_rank.loc[f] = [f+1, data[feature_col].columns[indices[f]], radm_clf.feature_importances_[indices[f]]]
x = feature_rank.loc[0:22, 'feature'].tolist()
print(x)

# Prepare final dataset
X = data[x]
Y = data['Weekly_Sales']
data = pd.concat([X, Y], axis=1)
data.to_csv('./datasets/final_data.csv')

# Train-test split
X = data.drop(['Weekly_Sales'], axis=1)
Y = data.Weekly_Sales
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=50)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_acc = lr.score(X_test, y_test) * 100
print("Linear Regressor Accuracy - ", lr_acc)
y_pred = lr.predict(X_test)
print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))
lr_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
lr_df.to_csv('./predictions/lr_real_pred.csv')
plt.figure(figsize=(20, 8))
plt.plot(lr.predict(X_test[:200]), label="prediction", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.savefig('plots/lr_real_pred.png')
plt.show()

pkl_filename = "./models/linear_regressor.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(lr, file)
    print("Saved model to disk")
else:
    print("Model already saved")

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test) * 100
print("Random Forest Regressor Accuracy - ", rf_acc)
y_pred = rf.predict(X_test)
print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))
rf_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
rf_df.to_csv('./predictions/rf_real_pred.csv')
plt.figure(figsize=(20, 8))
plt.plot(rf.predict(X_test[:200]), label="prediction", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.savefig('plots/rf_real_pred.png')
plt.show()

pkl_filename = "./models/randomforest_regressor.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(rf, file)
    print("Saved model to disk")
else:
    print("Model already saved")

# K-Neighbors Regressor
knn = KNeighborsRegressor(n_neighbors=1, weights='uniform')
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test) * 100
print("KNeighbors Regressor Accuracy - ", knn_acc)
y_pred = knn.predict(X_test)
print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))
knn_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
knn_df.to_csv('./predictions/knn_real_pred.csv')
plt.figure(figsize=(20, 8))
plt.plot(knn.predict(X_test[:200]), label="prediction", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.savefig('plots/knn_real_pred.png')
plt.show()

pkl_filename = "./models/knn_regressor.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(knn, file)
    print("Saved model to disk")
else:
    print("Model already saved")

# XGBoost
xgbr = XGBRegressor()
xgbr.fit(X_train, y_train)
xgb_acc = xgbr.score(X_test, y_test) * 100
print("XGBoost Regressor Accuracy - ", xgb_acc)
y_pred = xgbr.predict(X_test)
print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))
xgb_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
xgb_df.to_csv('./predictions/xgb_real_pred.csv')
plt.figure(figsize=(20, 8))
plt.plot(xgbr.predict(X_test[:200]), label="prediction", linewidth=2.0, color='blue')
plt.plot(y_test[:200].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.savefig('plots/xgb_real_pred.png')
plt.show()

pkl_filename = "./models/xgboost_regressor.pkl"
if not path.isfile(pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(xgbr, file)
    print("Saved model to disk")
else:
    print("Model already saved")

# Deep Neural Network
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

estimator_model = KerasRegressor(model=create_model, epochs=100, batch_size=5000, verbose=1)
estimator_model.fit(X_train, y_train, validation_split=0.2)
plt.figure(figsize=(8, 4))
plt.plot(estimator_model.history_['loss'], label='Train Loss')
plt.plot(estimator_model.history_['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.savefig('plots/dnn_loss.png')
plt.show()

y_pred = estimator_model.predict(X_test)
dnn_acc = metrics.r2_score(y_test, y_pred) * 100
print("Deep Neural Network Accuracy - ", dnn_acc)
print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2", metrics.explained_variance_score(y_test, y_pred))
dnn_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dnn_df.to_csv('./predictions/dnn_real_pred.csv')

plt.figure(figsize=(20, 8))
plt.plot(estimator_model.predict(X_test[200:300]), label="prediction", linewidth=2.0, color='blue')
plt.plot(y_test[200:300].values, label="real_values", linewidth=2.0, color='lightcoral')
plt.legend(loc="best")
plt.savefig('plots/dnn_real_pred.png')
plt.show()

filepath = './models/dnn_regressor.json'
weightspath = './models/dnn_regressor.weights.h5'
if not path.isfile(filepath):
    model_json = estimator_model.model_.to_json()
    with open(filepath, "w") as json_file:
        json_file.write(model_json)
    estimator_model.model_.save_weights(weightspath)
    print("Saved DNN model to disk")
else:
    print("DNN model already saved")

# Compare model accuracies
acc = {'model': ['Linear Regression', 'Random Forest', 'K-Neighbors', 'XGBoost', 'Deep Neural Network'], 'accuracy': [lr_acc, rf_acc, knn_acc, xgb_acc, dnn_acc]}
acc_df = pd.DataFrame(acc)
plt.figure(figsize=(10, 8))
sns.barplot(x='model', y='accuracy', data=acc_df, palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.savefig('plots/compared_models.png')
plt.show()