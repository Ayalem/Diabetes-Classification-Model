# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib


diabetes=pd.read_csv('diabetes_dataset.csv')

#learning about the data
print(diabetes.head())
print(diabetes.columns)
print(diabetes.info())
print(diabetes.describe())
print(diabetes['Outcome'].value_counts())
#computing sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))
#cost function with regularization :
def compute_cost(X,y,w,b,lambda_):
    m=X.shape[0]
    z=np.dot(X,w)+b
    f_wb=sigmoid(z)
    cost=np.sum(-y*np.log(f_wb)-(1-y)*np.log(1-f_wb))
    cost+=np.sum(w**2)*lambda_
    return cost/(2*m)
#gradient function vectorized:
def compute_gradient(X,y,w,b,lambda_):
     m=X.shape[0]
     z=np.dot(X,w)+b
     f_wb=sigmoid(z)
     error=f_wb-y
     dw=(1/m)*(np.dot(X.T,error)+lambda_*w)
     db=(1/m)*(np.sum(error))
     return dw,db

#gradient descent
import math
def gradient_descent(X,y,w_in,b_in,alpha,lambda_,num_iterations=1000):
    J_history=[]
    w=np.copy(w_in)
    b=b_in
    for i in range (num_iterations):
        dw,db=compute_gradient(X,y,w,b,lambda_)
        w=w-alpha*dw
        b=b-alpha*db
        if i<10000:
            if math.ceil(i%10)==0:
                J_history.append(compute_cost(X,y,w,b,lambda_))
                print(f"the cost at iteration{i} is:{round(J_history[-1],3)}")
    return w,b,J_history
#applying to data:
print(diabetes.dtypes)
features=[col for col in diabetes.columns if col != 'Outcome']
print(features)
X=diabetes[features].to_numpy() 
y=diabetes['Outcome'].to_numpy() 
print(X)
print(y)
print("the shape of X:",X.shape)
print("the shape of y:",y.shape)
#split training set:
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
print("x_train shape:",x_train.shape)
print("y_train shape:",y_train.shape)
print("x_test shape:",x_test.shape)
print("y_test shape:",x_test.shape)
#scaling the data:
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
joblib.dump(scaler,'scaler.pkl')
#fitting with manually set alpha and lambda:
m,n=x_train_scaled.shape
w=np.zeros(n)
b=0
alpha=0.01
lambda_=0.01
w_final,b_final,J_history=gradient_descent(x_train_scaled,y_train,w,b,alpha,lambda_)
print("w_final:",w_final)
print("b_final",b_final)
joblib.dump(w,'weights.pkl')
joblib.dump(b,'bias.pkl')
#ploting learning curve:
plt.plot(J_history ,color='seagreen')
plt.title("cost vs iteration")
plt.xlabel('iterations')
plt.ylabel('cost')
#predicting on test set:
def predict(X,w,b):
    z=np.dot(X,w)+b
    f_wb=sigmoid(z)
    prediction=(f_wb>=0.5).astype(int)#convert it to 0/1
    return prediction
#cheking error:
y_pred = predict(x_test, w_final, b_final)
print(y_pred)
missclassified=0
num_pred=len(y_pred)
for i in range(num_pred):
    if y_pred[i]!=y_test[i]:
     missclassified+=1
error=missclassified/num_pred
print(error)
