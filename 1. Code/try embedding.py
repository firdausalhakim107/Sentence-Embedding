import pandas as pd # dataframe manipulation
import numpy as np # linear algebra
from sentence_transformers import SentenceTransformer



df = pd.read_csv("/Users/apple/Desktop/Thesis/Customer-Churn/1. Datasets/3. IBM telco churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def compile_text(x):


    text =  f"""gender: {x['gender']},  
                SeniorCitizen: {x['seniorcitizen']}, 
                Partner: {x['partner']}, 
                Dependents: {x['dependents']}, 
                tenure: {x['tenure']}, 
                PhoneService: {x['phoneservice']}
                MultipleLines: {x['multiplelines']}
                InternetService: {x['internetservice']}
                OnlineSecurity: {x['onlinesecurity']}
                DeviceProtection: {x['devicerotection']}
                TechSupport: {x['techsupport']}
                StreamingTV: {x['streamingtv']}
                StreamingMovies: {x['streamingmovies']}
                Contract: {x['contract']}
                PaperlessBilling: {x['paperlessbilling']}
                PaymentMethod: {x['paymentmethod']}
                MonthlyCharges: {x['monthlycharges']}
                TotalCharges: {x['totalcharges']}
                Churn: {x['churn']}
            """

    return text

sentences = df.apply(lambda x: compile_text(x), axis=1).tolist()



model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")

output = model.encode(sentences=sentences, show_progress_bar= True, normalize_embeddings  = True)

df_embedding = pd.DataFrame(output)
df_embedding


df_embedding.to_csv("/Users/apple/Desktop/Thesis/Customer-Churn/1. Datasets/embedding_train.csv",index = False)

