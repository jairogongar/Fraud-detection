import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from preprocessing import preprocess
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('./application_data.csv')
def train_model(data):
    
    #Preprocess the data
    preprocess(data)
    # Target feature selection
    X = data.drop("TARGET",axis=1)
    y = data.TARGET

    # split dataset
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

    # train the model
   

    with mlflow.start_run():
        rf_Classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
        rf_Classifier.fit(X_train, y_train)
        
        mlflow.log_param("n_estimators", 10)
        mlflow.log_metric("test_accuracy", rf_Classifier.score(X_test, y_test))
        mlflow.sklearn.log_model(rf_Classifier, "random-forest-model")
        
        mlflow.register_model("runs:/{}/model".format('fraud_model'), "production")
        
        with open('model-0.1.0.pkl', 'wb') as f: 
            pickle.dump(rf_Classifier, f)
            
if __name__ == "__main__":
    train_model(data)
