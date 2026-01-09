import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,DBSCAN
from sklearn.pipeline import Pipeline
import logging


class Model_train:
    
    def model(self,df):
        
        df=df.drop("person_default",axis=1)
        
        scaler=StandardScaler()
        scaled_df=scaler.fit_transform(df)
        
        ssd=[]
        silhoute_score=[]
        
        for i in range(2,30):
            
            model=KMeans(n_clusters=i)
            cluster_labels=model.fit(scaled_df)
            ssd.append(model.inertia_)
            score=silhouette_score(scaled_df,cluster_labels)
            silhoute_score.append(score)
        
        plt.figure(figsize=(10,6))
        plt.plot(range(2,30),ssd,"--o")
        plt.show()
            