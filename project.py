import pickle



import numpy as np
import pandas as pd
from flask_cors import  cross_origin
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

def prediction(x):
    with open(r"C:\Users\User\OneDrive\Desktop\PGA 23\Project diabetsmodel.pickle","rb") as s:
        model=pickle.load(s)
    return model.predict(x)

@cross_origin()
@app.route('/')
def home():
    print("Home loaded")
    return render_template('index.html')
@cross_origin()
@app.route('/predict',methods=['POST'])
def predict():
    print("predict function loaded")
    HighBP = float(request.form['HighBP'])
    HighChol = float(request.form['HighChol'])
    CholCheck = float(request.form['CholCheck in 5 Year'])
    BMI = float(request.form['BMI'])
    Smoker = float(request.form['Smoked at least 100 cigarettes in your entire life?'])
    Stroke = float(request.form['Had a stroke'])
    HeartDiseaseorAttack = float(request.form['Had coronary heart disease (CHD) or myocardial infarction (MI)'])
    PhysActivity = float(request.form['physical activity in past 30 days'])
    Fruits = float(request.form['Consume Fruit 1 or more times per day'])
    Veggies = float(request.form['Consume Vegetables 1 or more times per day'])
    HvyAlcoholConsump = float(request.form['HvyAlcoholConsump(adult men >=14 drinks per week and adult women>=7 drinks per week)'])
    AnyHealthcare = float(request.form['AnyHealthcareHave any kind of health care coverage, including health insurance, prepaid plans'])
    NoDocbcCost = float(request.form['NoDocbcCost Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?'])
    GenHlth = float(request.form['Would you say that in general your health is'])
    MentHlth = float(request.form['MentHlthdays of poor mental health scale 1-30 days'])
    PhysHlth = float(request.form['physical illness or injury days in past 30 days'])
    DiffWalk = float(request.form['Do you have serious difficulty walking or climbing stairs?'])
    Sex = float(request.form['Sex'])
    Age = float(request.form['Age category 1=18-24 2=25-35 3=36-46 4=47-57 5=58-68 6=69-79 7=80 or above'])
    Education = float(request.form['Education level scale 1-6 1=Never attended school or only kindergarten 2=elementary 3=up to 10th 4=up to 12th 5=Any UG 6=PG or above'])
    Income = float(request.form['Income(per anum) scale 0-8 0=>No Income 1=>10,000 2=>20,000 3=>30,000 4=>40,000 5=>50,000 6=>60,000 7=>70,000 8=above 70,000'])

    
    # predictions using the loaded model file
    preds =prediction(
        [[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity,Fruits,Veggies,HvyAlcoholConsump,AnyHealthcare,NoDocbcCost,GenHlth,MentHlth,PhysHlth,DiffWalk,Sex,Age,Education,Income]])
    print("Prediction:",prediction)
    if preds == 1:
            preds = "diabetes"

    else:
            preds = "Normal"

    # showing the prediction results in a UI
    if  preds =="diabetes":

        return render_template('diabetes.html', prediction=preds)
    else:
        return render_template('Normal.html',prediction=preds)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
	#app.run(debug=True)
