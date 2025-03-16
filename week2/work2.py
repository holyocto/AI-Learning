from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor # Changed to KNeighborsRegressor
from sklearn.metrics import mean_squared_error # Changed to mean_squared_error
from sklearn.preprocessing import StandardScaler

D = datasets.load_linnerud()

X = D.data
y = D.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

scaler = StandardScaler()
X2_train = scaler.fit_transform(X_train)
X2_test = scaler.fit_transform(X_test)

# Changed to KNeighborsRegressor for regression
knn = KNeighborsRegressor(n_neighbors=5)  

knn.fit(X2_train, y_train)

y_pred = knn.predict(X2_test)

# Changed to mean_squared_error for regression
scores = mean_squared_error(y_test, y_pred)  
print(f"{scores:.6f}")