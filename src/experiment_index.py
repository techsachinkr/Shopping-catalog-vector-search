import os
import pandas as pd
import faiss
from embed_dataset import GenerateEmbeddings


base_dir_path=os.getcwd()

sampled_datavals_product=pd.read_parquet(os.path.join(base_dir_path,"dataset/sampled_dataset/sampled_datavals_product_data.parquet"))
sampled_datavals_product_imbalanced=pd.read_parquet(os.path.join(base_dir_path,"dataset/sampled_dataset/sampled_datavals_product_data_imbalanced.parquet"))

sampled_queries = pd.read_parquet(os.path.join(base_dir_path,"dataset/sampled_dataset/sampled_query_examples.parquet"))


dataset_examples=pd.read_parquet(os.path.join(base_dir_path,"dataset/source_files/shopping_queries_dataset_examples.parquet"))
filt_examples=dataset_examples[(dataset_examples["product_locale"]=="us") & (dataset_examples["esci_label"]=="E")]

class ExperimentIndex:
        def __init__(self, index_path, embeddings_obj) -> None:
            self.index_path = index_path
            self.index = self.load_index(self.index_path)
            self.embeddings_obj = embeddings_obj

        def load_index(self,index_path):
            return faiss.read_index(index_path)
        
        def experiment_with_index(self,query_val, top_k):
            query_embedding = self.embeddings_obj.encode_query(query_val)
            relevant_examples_df=filt_examples[filt_examples["query"]==query_val]
            relevant_ids=relevant_examples_df["product_id"].tolist()

            scores, indices = self.index.search(query_embedding, top_k)
            retrieved_examples=sampled_datavals_product[sampled_datavals_product['ID'].isin(indices[0])]
            
            return scores,retrieved_examples
        

if __name__ == "__main__":
    generate_embeddings_base_obj=GenerateEmbeddings(model_name="intfloat/multilingual-e5-small")

    experiment_index_obj=ExperimentIndex(os.path.join(base_dir_path,"vector_indexes/balanced_dataset/384dim/product_fields_exclude_desc_384dim_hnsw.index"), generate_embeddings_base_obj)
    scores,retrieved_examples=experiment_index_obj.experiment_with_index("black dress",10)
    print(scores)
    print(retrieved_examples)

   
     