from data_load import Load_File
from data_clean import Cleaner
from config import CSV_PATH
from FE import Feature_eng
from unsupervised import unsupervised_Model_train

def main():
    
    
    obj1 = Load_File()
    df_original = obj1.load(CSV_PATH)

    obj2 = Cleaner()
    df_clean = obj2.clean(df_original)

    obj3 = Feature_eng()
    scaled_df = obj3.FE(df_clean)

    obj4 = unsupervised_Model_train()
    pca_df = obj4.pca_model(scaled_df)

    df_final = obj4.cluster_model(pca_df, df_clean)

    
if __name__=="__main__":
    
    main()
