import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging
from sqlalchemy import inspect
from config import ENGINE

class unsupervised_Model_train:
    
    def pca_model(self,scaled_df):
        
        try:
        
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
        except Exception as e:
            logging.error(e)
        
    def cluster_model(self,pca_df,df):
        
        try:
                
            # ssd=[]
            # silhoute_score=[]
            
            # for i in range(2,30):
                
            #     model=KMeans(n_clusters=i,random_state=101)
            #     cluster_labels=model.fit_predict(pca_df)
            #     ssd.append(model.inertia_)
                
                # score=silhouette_score(pca_df,cluster_labels)
                # silhoute_score.append(score)
                
                # print(f"Clusters: {i}, Silhouette Score: {score:.4f}")
            
            # plt.figure(figsize=(10,6))
            # plt.plot(range(2,30),ssd,"--o")
            # plt.show()
            
            n_clusters=10
            logging.info("Final implementation of KMEANS")
            final_model=KMeans(n_clusters=n_clusters,random_state=101,init="k-means++")
            label_clusters=final_model.fit_predict(pca_df)
            df["clusters"]=label_clusters+1
            logging.info(label_clusters)

                    
            # ----------------------------
    # 1. NUMERIC CLUSTER SUMMARY
    # ----------------------------
            numeric_cols = [
                "person_age",
                "cb_person_cred_hist_length",
                "person_income",
                "loan_amnt"
            ]

            cluster_summary = (
                df.groupby("clusters")[numeric_cols]
                .mean()
                .reset_index()  # converts cluster index back to column
                .sort_values(by=["person_income", "loan_amnt"], ascending=[False, False])
            )

            print("\nNUMERIC CLUSTER SUMMARY")
            print(cluster_summary)

            # ----------------------------
            # 2. CATEGORICAL SUMMARY (MODE)
            # ----------------------------
            categorical_cols = ["loan_intent", "person_home_ownership"]

            cat_summary = (
                df.groupby("clusters")[categorical_cols]
                .agg(lambda x: x.mode()[0])
                .reset_index()
            )

            print("\nCATEGORICAL CLUSTER SUMMARY")
            print(cat_summary)

            # ----------------------------
            # 3. MAP CLUSTER NAMES
            # ----------------------------
            cluster_names = {
                1: "High-Earning Aggressive Borrowers",
                2: "Over-Leveraged High-Risk Borrowers",
                3: "Established Low-Risk Professionals",
                4: "Balanced Borrowers",
                5: "Growth-Stage Borrowers",
                6: "Mainstream Young Borrowers",
                7: "Stable Early-Career Borrowers",
                8: "Entry-Level Borrowers",
                9: "Conservative Starters",
                10: "Ambitious Credit Seekers"
            }

            df["cluster_name"] = df["clusters"].map(cluster_names)

            print("\nSAMPLE DATA WITH CLUSTER NAMES")
            print(df.head())
            
            inspector=inspect(ENGINE)
            
            if not inspector.has_table("table name"):
                
                df.to_sql(
                    name="table name",
                    con=ENGINE,
                    if_exists="fail",
                    index=False
                    
                )
                logging.info("Analysis Data stored in MYSQL.....")
            
            elif inspector.has_table("table name"):
                logging.info("Skipping Load Table already Exists.....")
                
            else:
                raise RuntimeError("Failed to store data in MySQL")
            
            
                
                
    # Add  two PCA components to the dataframe
            df["PC1"] = pca_df["PC1"]
            df["PC16"] = pca_df["PC16"]

            plt.figure(figsize=(10,6))
            sns.scatterplot(
                data=df,
                x="PC1",
                y="PC16",
                hue="cluster_name",      # colored by cluster names
                palette="tab10",
                s=60,                    # size of points
                alpha=0.4                # slightly transparent for overlapping points
            )
            plt.title("2D Visualization of 10 Clusters (PCA Projection)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend(bbox_to_anchor=(1.05,8), loc="upper left")  # legend outside plot
            plt.show()

            # ----------------------------
            # 4. VISUALIZATION (numeric features only)
            # ----------------------------
            melt_df_1 = cluster_summary.melt(
                id_vars="clusters",
                value_vars=["person_age","cb_person_cred_hist_length"],
                var_name="metric",
                value_name="mean_value"
            )
            melt_df_2 = cluster_summary.melt(
                id_vars="clusters",
                value_vars=["person_income","loan_amnt"],
                var_name="metric",
                value_name="mean_value"
            )

            fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,6))
            sns.barplot(data=melt_df_1, x="clusters", y="mean_value", hue="metric",ax=ax[0],palette="viridis")
            ax[0].set_title("Age & Credit History by Cluster")
            ax[0].set_xlabel("Cluster")
            ax[0].set_ylabel("Mean Value")
            sns.barplot(data=melt_df_2, x="clusters", y="mean_value", hue="metric",ax=ax[1],palette="rocket")
            ax[1].set_title("Income & Loan Amount by Cluster")
            ax[1].set_xlabel("Cluster")
            ax[1].set_ylabel("Mean Value")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(e)
        
