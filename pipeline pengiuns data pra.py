import pandas as pd
import numpy as np
import seaborn as sns
from pandas.core.interchange.from_dataframe import categorical_column_to_series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,confusion_matrix

df=sns.load_dataset("Penguins")
X=df.drop(columns="species")
y=df["species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

numerical_col=X.select_dtypes(include=['number']).columns
categorical_col=X.select_dtypes(include=['object']).columns
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean',)),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output= False)),
    ('scaler', StandardScaler())
])

preprocessor=ColumnTransformer([
    ('numerical',numerical_transformer,numerical_col),
    ('categorical',categorical_transformer,categorical_col)
])

pipe=Pipeline([
    ('preprocessor',preprocessor),
    ('model',LogisticRegression())
])

pipe=pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)

print("accuracy_score: ",accuracy_score(y_test,y_pred))
print("precision_score: ",precision_score(y_test,y_pred,average='micro'))
print("recall_score: ",recall_score(y_test,y_pred,average='micro' ))
print("f1_score: ",f1_score(y_test,y_pred,average='micro'))
print("Cnfusion metris:\n",confusion_matrix(y_test,y_pred))

import pickle

#pickle.dump(pipe,open("model.pkl",'wb'))
