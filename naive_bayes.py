#-------------------------------------------------------------------------
# AUTHOR: John Li
# FILENAME: naive_bayes.py
# SPECIFICATION: Using Naive Bayes to predict whether to play tennis or not based on weather conditions
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook_map = {'Sunny':1, 'Overcast':2, 'Rain':3}
temp_map    = {'Hot':1, 'Mild':2, 'Cool':3}
hum_map     = {'High':1, 'Normal':2}
wind_map    = {'Weak':1, 'Strong':2}
label_map   = {'Yes':1, 'No':2}

X = []
Y = []
# dbTraining rows look like: [Day, Outlook, Temperature, Humidity, Wind, PlayTennis]
for r in dbTraining:
    X.append([
        outlook_map[r[1]],
        temp_map[r[2]],
        hum_map[r[3]],
        wind_map[r[4]],
    ])
    Y.append(label_map[r[5]])


#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print("Day\tOutlook\tTemperature\tHumidity\tWind\tPlayTennis\tConfidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
THRESH = 0.75
for r in dbTest:
    xt = [[
        outlook_map[r[1]],
        temp_map[r[2]],
        hum_map[r[3]],
        wind_map[r[4]],
    ]]
    probs = clf.predict_proba(xt)[0]        # [P(class=1='Yes'), P(class=2='No')]
    if probs[0] >= probs[1]:
        pred_label = 'Yes'
        conf = probs[0]
    else:
        pred_label = 'No'
        conf = probs[1]

    if conf >= THRESH:
        print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{pred_label}\t{conf:.2f}")

