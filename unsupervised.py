import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA
import logging

class unsupervised_Model_train:
    
    def pca_model(self,scaled_df):
        
        # max_variance=[]
        
        # for i in range(1,22):
        #     model=PCA(n_components=i)
        #     model.fit(scaled_df)
            
        #     max_variance.append(np.sum(model.explained_variance_ratio_))
        
        # plt.figure(figsize=(8, 5))
        # plt.plot(range(1, 22), max_variance, marker='o')
        # plt.xlabel("Number of Components")
        # plt.ylabel("Cumulative Explained Variance")
        # plt.title("PCA Elbow Method")
        # plt.grid(True)
        # plt.show()
        
        
        logging.info("Final implementation of PCA")
        
        components=16
        model_final=PCA(n_components=components)
        
        final_pca=model_final.fit_transform(scaled_df)
        print("\nSelected Components:", model_final.n_components_)
        
        pca_df=pd.DataFrame(final_pca,
        columns=[f"PC{i+1}" for i in range(components)])
        
        print(pca_df.head())
        print(pca_df.shape)
        
        loadings_df = pd.DataFrame(
        model_final.components_,
        index=[f"PC{i+1}" for i in range(components)],
        columns=scaled_df.columns
    )

        print("\nPCA Loadings:")
        print(loadings_df)
        
        print(model_final.explained_variance_ratio_)
        print(np.sum(model_final.explained_variance_ratio_))
        
        # plt.figure(figsize=(15, 6), dpi=100)
        # sns.heatmap(loadings_df, cmap="coolwarm",annot=True)
        # plt.title("PCA Feature Contributions")
        # plt.tight_layout()
        # plt.show()
        
        return pca_df
    
    def cluster_model(self,pca_df,df):
            
        # ssd=[]
        # silhoute_score=[]
        
        # for i in range(2,30):
            
        #     model=KMeans(n_clusters=i,random_state=101)
        #     cluster_labels=model.fit_predict(pca_df)
        #     ssd.append(model.inertia_)
            
        #     score=silhouette_score(pca_df,cluster_labels)
        #     silhoute_score.append(score)
            
        #     print(f"Clusters: {i}, Silhouette Score: {score:.4f}")
        
        # plt.figure(figsize=(10,6))
        # plt.plot(range(2,30),ssd,"--o")
        # plt.show()
        
        logging.info("Final implementation of KMEANS")
        final_model=KMeans(n_clusters=2,random_state=101,init="k-means++")
        label_clusters=final_model.fit_predict(pca_df)
        
        logging.info(label_clusters)
        
        df["clusters"]=label_clusters+1
        return df.groupby("clusters")[[""]]
        
