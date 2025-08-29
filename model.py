from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
df=pd.read_csv('Iris.csv')
X=df.iloc[:,1:5]
y=df.iloc[:,5]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)
knn=KNeighborsClassifier(n_neighbors=17)#n_neighbors will be the K Value
knn.fit(X_train,y_train)
# create pickle dump
pickle.dump(knn,open("model.pkl","wb"))