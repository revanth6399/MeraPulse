import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import pickle
dataset = pd.read_csv('heartuci.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20,random_state = 1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

forest = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=1)
forest.fit(X_train, Y_train)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred_lr = logreg.predict(X_test)

model1 = logreg
model2 = forest
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
n = len(X_test)
y = model.predict(X_test[n-1].reshape(1, -1))
print(y)

pickle.dump(model, open('heartuci.pkl','wb'))
model = pickle.load(open('heartuci.pkl','rb'))