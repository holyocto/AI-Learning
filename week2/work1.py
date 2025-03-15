from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

wine = datasets.load_wine()

X = wine.data
y = wine.target

X_train,X_test,y_train,y_test =train_test_split(X, y,test_size = 0.2, random_state=3)

scaler=StandardScaler()
X2_train = scaler.fit_transform(X_train)
X2_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X2_train, y_train)

y_pred = knn.predict(X2_test)

scores = metrics.accuracy_score(y_test, y_pred)
print(f"{scores:.6f}")

x_new = [[12.5, 2.43, 2.67, 10.5, 100.0, 2.10, 3.30, 2.90, 1.9, 0.80, 1.95, 0.65, 420.0],[14.1, 1.35, 2.12, 10.8, 85.0, 2.50, 2.80, 3.10, 1.8, 0.70, 1.75, 0.58, 460.0]]
classes = ['class_0','class_1', 'class_2']

y_predict = knn.predict(x_new)
print(y_predict)

print(classes[y_predict[0]])
print(classes[y_predict[1]])