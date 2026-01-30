
import numpy as np
import pandas as pd
from config import ENGINE
import logging
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,accuracy_score,roc_auc_score,RocCurveDisplay,make_scorer,precision_score,recall_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
class supervised_model_train:
    
    def __init__(self):
        
        self.query="""
        select * from analysis_table
        """
        
        
    def FE(self,df):
        
        df=pd.read_sql(self.query,ENGINE)
        
        if df is None:
            
            logging.error("Something Went Wrong....")
        
        try:
        
            cat_col=["person_home_ownership","loan_intent","loan_grade","clusters"]
            
            encoded=OneHotEncoder(sparse_output=False,drop="first")
            encoded_array=encoded.fit_transform(df[cat_col])
            encoded_colum=encoded.get_feature_names_out(cat_col)
            
            final_encode=pd.DataFrame(encoded_array,columns=encoded_colum)
            
            self.df_final=pd.concat([df.drop(["person_home_ownership","loan_grade","loan_intent"],axis=1),final_encode],axis=1)
            print(self.df_final.head())
            
            return self.df_final
        
        except Exception as e:
            
            logging.error(f"Something Went Wrong {e}")
            raise ValueError
        
    def predict(self):
        
        if self.df_final is None:
            
            logging.error("Dataframe is empty......")
            raise ValueError
        
        try:
            
            X=self.df_final.drop(["cluster_name","person_default"],axis=1)
            Y=self.df_final["person_default"]
            
            X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.7,random_state=101)
            
            scaler=StandardScaler()
            scaled_xtrain=scaler.fit_transform(X_train)
            scaled_xtest=scaler.transform(X_test)
            
            model=[LogisticRegression(solver="saga",class_weight="balanced"),RandomForestClassifier(class_weight="balanced"),
                   DecisionTreeClassifier(class_weight="balanced"),KNeighborsClassifier(),SVC(class_weight="balanced"),AdaBoostClassifier()]
            
            # making pos_label readable by giving label Y/N instead it read 1/0 which are not available in label column
            recall = make_scorer(recall_score, pos_label='Y', zero_division=0)
            precision = make_scorer(precision_score, pos_label='Y', zero_division=0)
            
            # cross validation on all models
            # for i in model:
                
            #     logging.info(f"Evaluation of {i}")
            #     score=cross_validate(i,scaled_xtrain,y_train,scoring={"precision":precision,"recall":recall,"accuracy":"accuracy"},cv=5)
            
            #     score_df=pd.DataFrame(score)
            #     print(score_df)
            #     logging.info(score_df.mean())
            
            
            # choosinh logreg for final implementation and deploying it using pipeline
            
            operation=Pipeline([("scaler",StandardScaler()),
            ("model",LogisticRegression(solver="saga",class_weight="balanced"))])
        
            para={
                "model__penalty":["l1","l2"],
                "model__C":np.logspace(-4,4,15)                
            }
            
            grid_model=GridSearchCV(estimator=operation,param_grid=para,cv=5,scoring=precision,n_jobs=-1,verbose=2)
            grid_model.fit(X_train,y_train)
            
            logging.info(grid_model.best_estimator_)
            logging.info(grid_model.best_params_)
            logging.info(grid_model.best_score_)
            
            pre=grid_model.predict(X_test)
            prob_pre=grid_model.predict_proba(X_test)[: ,1]
            
            cr=classification_report(y_test,pre)
            print(cr)
            
            ConfusionMatrixDisplay.from_predictions(y_test,pre)
            
            ytest_label=pd.get_dummies(y_test,drop_first=True)
            
            RocCurveDisplay.from_predictions(ytest_label,prob_pre)
            plt.show()
            
            
        except Exception as e:
            
            logging.error(e)
        
        
