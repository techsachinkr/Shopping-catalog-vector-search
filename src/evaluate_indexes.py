from typing import List
import faiss
import numpy as np
import pandas as pd
import os
from embed_dataset import GenerateEmbeddings
import logging
import time


base_dir_path=os.getcwd()

sampled_datavals_product=pd.read_parquet(os.path.join(base_dir_path,"dataset/sampled_dataset/sampled_datavals_product_data.parquet"))
sampled_datavals_product['ID'] = np.arange(0, len(sampled_datavals_product))

sampled_queries = pd.read_parquet(os.path.join(base_dir_path,"dataset/sampled_dataset/sampled_query_examples.parquet"))


dataset_examples=pd.read_parquet(os.path.join(base_dir_path,"dataset/source_files/shopping_queries_dataset_examples.parquet"))
filt_examples=dataset_examples[(dataset_examples["product_locale"]=="us") & (dataset_examples["esci_label"]=="E")]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluateIndex:

  def __init__(self, index_path, embeddings_obj, eval_queries) -> None:
      self.index_path = index_path
      self.index = self.load_index(self.index_path)
      self.embeddings_obj = embeddings_obj
      self.eval_queries = eval_queries
      self.top_k_values = [1, 5, 10]

  def load_index(self,index_path):
      return faiss.read_index(index_path)


  def calculate_top_k_metrics(self):
      metrics={
          "HITS@k": {1:[],5:[],10:[]},
          "MRR": {1:[],5:[],10:[]}
      }
      times=[]
      for queryval in self.eval_queries:
          query_embedding = self.embeddings_obj.encode_query(queryval)
          relevant_examples_df=filt_examples[filt_examples["query"]==queryval]
          relevant_ids=relevant_examples_df["product_id"].tolist()
          curr_times=[]
          for top_k in self.top_k_values:
              start_time = time.time()
              scores, indices = self.index.search(query_embedding, top_k)
              retrieved_examples=sampled_datavals_product[sampled_datavals_product['ID'].isin(indices[0])]
              retrieved_examples_ids=retrieved_examples["product_id"].tolist()

              hitvals=self._calculate_hits_at_k(relevant_ids,retrieved_examples_ids)
              metrics["HITS@k"][top_k].append(hitvals)

              mrrvals=self._calculate_mrr(relevant_ids,retrieved_examples_ids)
              metrics["MRR"][top_k].append(mrrvals)
              curr_times.append(time.time() - start_time)
          times.append(float(np.mean(curr_times)))
      avg_metrics = {
          "HITS@k": {1: round(float(np.mean(metrics["HITS@k"][1])),2), 5: round(float(np.mean(metrics["HITS@k"][5])),2), 10: round(float(np.mean(metrics["HITS@k"][10])),2)},
          "MRR": {1: round(float(np.mean(metrics["MRR"][1])),2), 5: round(float(np.mean(metrics["MRR"][5])),2), 10:round(float(np.mean(metrics["MRR"][10])),2)},
          'avg_search_time_seconds': float(np.mean(times)),
          'std_search_time_seconds': float(np.std(times))
      }
      
      return avg_metrics
  

  def calculate_top_k_metrics_debug(self):
        all_metrics=[]
        
        times=[]
        for queryval in self.eval_queries:
            metrics={
            "HITS@k": {1:[],5:[],10:[]},
            "MRR": {1:[],5:[],10:[]}
            }
            query_embedding = self.embeddings_obj.encode_query(queryval)
            relevant_examples_df=filt_examples[filt_examples["query"]==queryval]
            relevant_ids=relevant_examples_df["product_id"].tolist()
            curr_times=[]
            metrics["queryval"]=queryval
            for top_k in self.top_k_values:
                start_time = time.time()
                scores, indices = self.index.search(query_embedding, top_k)
                retrieved_examples=sampled_datavals_product[sampled_datavals_product['ID'].isin(indices[0])]
                retrieved_examples_ids=retrieved_examples["product_id"].tolist()

                hitvals=self._calculate_hits_at_k(relevant_ids,retrieved_examples_ids)
                metrics["HITS@k"][top_k].append(hitvals)

                mrrvals=self._calculate_mrr(relevant_ids,retrieved_examples_ids)
                metrics["MRR"][top_k].append(mrrvals)
                curr_times.append(time.time() - start_time)
            times.append(float(np.mean(curr_times)))
            all_metrics.append(metrics)
        return all_metrics
    

  def _calculate_hits_at_k(self, relevant_ids: List[str], retrieved_ids: List[str]) -> float:
      """Calculate HITS@K metric"""
      for retrieved_id in retrieved_ids:
          if retrieved_id in relevant_ids:
            return float(1)
      return float(0)

  
  def _calculate_mrr(self, relevant_ids: List[str], retrieved_ids: List[str]) -> float:
      """Calculate Mean Reciprocal Rank for a single query"""
      for rank, retrieved_id in enumerate(retrieved_ids, 1):
          if retrieved_id in relevant_ids:
              return 1.0 / rank
      return 0.0
      
if __name__ == "__main__":

    eval_queries=sampled_queries["query"].tolist()

    index_suffix=["flat","ivf","hnsw","ivf_pq"]

    metrics_vals=[]

    generate_embeddings_base_obj=GenerateEmbeddings(model_name="intfloat/multilingual-e5-base")
    generate_embeddings_small_obj=GenerateEmbeddings(model_name="intfloat/multilingual-e5-small")

    # For balanced dataset

    # For all_product_fields_768dim_embeddings.npy

    for suffix in index_suffix:
        dirpath=os.path.join(base_dir_path,"vector_indexes/balanced_dataset/768dim/all_product_fields_768dim_"+suffix+".index")
        
        eval_index_obj= EvaluateIndex(dirpath, generate_embeddings_base_obj, eval_queries)
        metrics = eval_index_obj.calculate_top_k_metrics()
        logger.info("Metrics for balanced dataset, all_product_fields_768dim_"+suffix+".index")
        logger.info(metrics)
        metrics_vals.append({"dataset":"balanced","index_type":suffix, "embeddings_type":"all_product_fields_768dim",
                             "HITS@k_1":metrics["HITS@k"][1],
                             "HITS@k_5":metrics["HITS@k"][5],
                             "HITS@k_10":metrics["HITS@k"][10],
                             "MRR_1":metrics["MRR"][1],
                             "MRR_5":metrics["MRR"][5],
                             "MRR_10":metrics["MRR"][10],
                             "avg_search_time_seconds":metrics["avg_search_time_seconds"],
                             "std_search_time_seconds":metrics["std_search_time_seconds"]
                             })


    # For product_fields_exclude_desc_384dim_embeddings.npy


    for suffix in index_suffix:
        dirpath=os.path.join(base_dir_path,"vector_indexes/balanced_dataset/384dim/product_fields_exclude_desc_384dim_"+suffix+".index")
        eval_index_obj= EvaluateIndex(dirpath, generate_embeddings_small_obj, eval_queries)
        metrics = eval_index_obj.calculate_top_k_metrics()
        logger.info("Metrics for balanced dataset, product_fields_exclude_desc_384dim_"+suffix+".index")
        logger.info(metrics)
        metrics_vals.append({"dataset":"balanced","index_type":suffix, "embeddings_type":"product_fields_exclude_desc_384dim",
                             "HITS@k_1":metrics["HITS@k"][1],
                             "HITS@k_5":metrics["HITS@k"][5],
                             "HITS@k_10":metrics["HITS@k"][10],
                             "MRR_1":metrics["MRR"][1],
                             "MRR_5":metrics["MRR"][5],
                             "MRR_10":metrics["MRR"][10],
                             "avg_search_time_seconds":metrics["avg_search_time_seconds"],
                             "std_search_time_seconds":metrics["std_search_time_seconds"]
                             })

    
    # For imbalanced dataset

    # For all_product_fields_768dim_embeddings.npy

    for suffix in index_suffix:
        dirpath=os.path.join(base_dir_path,"vector_indexes/imbalanced_dataset/768dim/all_product_fields_768dim_"+suffix+".index")
        eval_index_obj= EvaluateIndex(dirpath, generate_embeddings_base_obj, eval_queries)
        metrics = eval_index_obj.calculate_top_k_metrics()
        logger.info("Metrics for imbalanced dataset, all_product_fields_768dim_"+suffix+".index")
        logger.info(metrics)
        metrics_vals.append({"dataset":"imbalanced","index_type":suffix, "embeddings_type":"all_product_fields_768dim",
                             "HITS@k_1":metrics["HITS@k"][1],
                             "HITS@k_5":metrics["HITS@k"][5],
                             "HITS@k_10":metrics["HITS@k"][10],
                             "MRR_1":metrics["MRR"][1],
                             "MRR_5":metrics["MRR"][5],
                             "MRR_10":metrics["MRR"][10],
                             "avg_search_time_seconds":metrics["avg_search_time_seconds"],
                             "std_search_time_seconds":metrics["std_search_time_seconds"]
                             })


    # For product_fields_exclude_desc_384dim_embeddings.npy

    for suffix in index_suffix:
        dirpath=os.path.join(base_dir_path,"vector_indexes/imbalanced_dataset/384dim/product_fields_exclude_desc_384dim_"+suffix+".index")
        eval_index_obj= EvaluateIndex(dirpath, generate_embeddings_small_obj, eval_queries)
        metrics = eval_index_obj.calculate_top_k_metrics()
        logger.info("Metrics for imbalanced dataset, product_fields_exclude_desc_384dim_"+suffix+".index")
        logger.info(metrics)
        metrics_vals.append({"dataset":"imbalanced","index_type":suffix, "embeddings_type":"product_fields_exclude_desc_384dim",
                             "HITS@k_1":metrics["HITS@k"][1],
                             "HITS@k_5":metrics["HITS@k"][5],
                             "HITS@k_10":metrics["HITS@k"][10],
                             "MRR_1":metrics["MRR"][1],
                             "MRR_5":metrics["MRR"][5],
                             "MRR_10":metrics["MRR"][10],
                             "avg_search_time_seconds":metrics["avg_search_time_seconds"],
                             "std_search_time_seconds":metrics["std_search_time_seconds"]
                             })

    metrics_df=pd.DataFrame(metrics_vals)
    metrics_df.to_excel(os.path.join(base_dir_path,"metrics/metrics.xlsx"),index=False)


    debug_metrics_vals=[]
    
    dirpath=os.path.join(base_dir_path,"vector_indexes/imbalanced_dataset/768dim/all_product_fields_768dim_hnsw.index")
    eval_index_obj= EvaluateIndex(dirpath, generate_embeddings_base_obj, eval_queries)
    metrics = eval_index_obj.calculate_top_k_metrics_debug()
    debug_metrics_df=pd.DataFrame(metrics)
    debug_metrics_df.to_excel(os.path.join(base_dir_path,"metrics/debug_metrics.xlsx"),index=False)


