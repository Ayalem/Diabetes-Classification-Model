from flask import Flask,request,render_template
import numpy as np
import joblib
w=joblib.load('weights.pkl')
b=joblib.load('bias.pkl')
scaler=joblib.load('scaler.pkl')
app=Flask(__name__)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def manual_prediction(X,w,b,scaler):
     X_scaled=scaler.transform([X])
     z=np.dot(X_scaled,w)+b
     f_wb=sigmoid(z)
     prediction=(f_wb>=0.5).astype(int)
     if prediction==1:
         return "you are diabetic"
     else:
      return "you are not diabetic"

@app.route('/')
def home():
    return "hello world"

@app.route('/predict',methods=['GET','POST'])
def predict():
    error=None
    if request.method=='POST':
        try:
            field_names = [
                 'Age', 'Pregnancies', 'BMI', 'Glucose', 'BloodPressure', 'HbA1c', 
                 'LDL', 'HDL', 'Triglycerides', 'WaistCircumference', 'HipCircumference', 
                  'WHR', 'FamilyHistory', 'DietType', 'Hypertension', 'MedicationUse'
            ]

            features=[]
            for field in field_names:
                if field in request.form:
                    if field in [ 'BMI', 'Glucose', 'BloodPressure',  'HbA1c', 'LDL', 'HDL', 'Triglycerides', 'WaistCircumference', 
                                 'HipCircumference', 'WHR']:
                                
                     features.append(float(request.form[field]))
                    else:
                      features.append(int(request.form[field]))
            prediction=manual_prediction(features,w,b,scaler)
        except Exception as e:
           error=str(e)
           prediction=None
        return render_template('predict.html',prediction=prediction,error=error)
   
    return render_template('predict.html', prediction=None, error=None)

if __name__=='__main__':
    app.run(debug=True)

