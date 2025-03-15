from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

D = datasets.load_diabetes()

X = D.data
y = D.target

X_train,X_test,y_train,y_test =train_test_split(X, y,test_size = 0.2, random_state=3)

scaler=StandardScaler()
X2_train = scaler.fit_transform(X_train)
X2_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=50)

knn.fit(X2_train, y_train)

y_pred = knn.predict(X2_test)

scores = metrics.accuracy_score(y_test, y_pred)
print(f"{scores:.6f}")