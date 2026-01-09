import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,DBSCAN
from sklearn.pipeline import Pipeline
import logging


class Model_train:
    
    def model(self,df):
        
        df=df.drop("person_default",axis=1)
        
        scaler=StandardScaler()
        scaler.fit_transform(df)
        
        for i in range(2,30):
            
            model=KMeans(n_clusters=i)
            model.fit()