import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from preprocessing import preprocess
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.metrics import accuracy_score
import pickle
from model import train_model
from preprocessing import preprocess
import sqlite3
from users import check_login



def login():
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    if check_login(username, password):
        print("Login successful!")
        with open('model-0.1.0.pkl', "rb") as f:
            rf_Classifier = pickle.load(f)

            testdata = pd.read_csv('testdata.csv')

            # preprocess the data and split it in x and y features
            preprocess(testdata)
            X = testdata.drop("TARGET",axis=1)
            y = testdata.TARGET  

            # test the model and get the accuracy   
            predicted = rf_Classifier.predict(X)
            accuracy=accuracy_score(predicted, y)

            # retrain the model if there is model drift
            if accuracy< 0.9:
                
                
                train_model(X,y)
                print("Model retrained successfully")

            else:
                print("the accuracy of the model is:", accuracy)
    else:
        print("Login failed. Incorrect username or password.")

# Login

if __name__=='__main__':
    login()


