from data_load import Load_File
from data_clean import Cleaner
from config import CSV_PATH
from unsupervised import Model_train

def main():
    
    
    def load():
        
        file_path=CSV_PATH
        obj1=Load_File()
        df=obj1.load(file_path)
        
        return df
    
    df=load()
    
    def clean():
        
        obj2=Cleaner()
        df_clean=obj2.clean(df)
        return df_clean
        
    df=clean()
    
    def model_train():
        
        obj3=Model_train()
        obj3.model(df)
    
    model_train()    
    
    
if __name__=="__main__":
    
    main()