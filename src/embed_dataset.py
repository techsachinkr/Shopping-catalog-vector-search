import logging
import re
import numpy as np
import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

base_dir_path=os.getcwd()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerateEmbeddings:

    def __init__(self,model_name="intfloat/multilingual-e5-base", embed_dtype="float32"):
      self.model=SentenceTransformer(model_name)
      self.embedding_dim = self.model.get_sentence_embedding_dimension()
      logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
      self.embeddings = None
      self.embed_dtype = embed_dtype


    def preprocess_text(self,text):
      """Clean and preprocess text.
      
        Args:
          text: Input text
          
      Returns:
          Cleaned and processed text
      
      """
      text = str(text)
      # Remove HTML tags
      text = re.sub(r'<[^>]+>', ' ', text)
      # Remove special characters but keep basic punctuation
      text = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', text)
      # Remove extra whitespace
      text = ' '.join(text.split())
      return text.lower()

    def create_combined_text(self,product_df, product_cols):
      """Create combined text representations for each product.
      
        Args:
          products_df: DataFrame containing product information
          product_cols: List of column names to consider for text representation
          
      Returns:
          List of combined text representations
      
      """
      logger.info("\nCreating combined text representations...")
      combined_texts = []
      for _, row in product_df.iterrows():
          text_parts = []
          for col in product_cols:
            curr_val=row[col]
            if curr_val:
              processed_text = self.preprocess_text(curr_val)
              text_parts.append(processed_text)
          combined_text = " ".join(text_parts)
          combined_texts.append(combined_text)
      return combined_texts


    def generate_embeddings(self, product_df, product_cols, batch_size=32):
        """
            Generate embeddings for all products using 'passage:' prefix
            
            Args:
                batch_size: Batch size for embedding generation
                
            Returns:
                Numpy array of embeddings
        """
        combined_text = self.create_combined_text(product_df, product_cols)
        embeddings = []
        for i in range(0, len(combined_text), batch_size):
              batch_texts = combined_text[i:i + batch_size]
              batch_embeddings = self.model.encode(
                  batch_texts,
                  convert_to_numpy=True,  # FAISS expects numpy arrays not pytorch tensors(default)
                  normalize_embeddings=True,  # Important for cosine similarity
                  show_progress_bar=True
              )
              embeddings.append(batch_embeddings)
              
              if (i // batch_size + 1) % 10 == 0:
                  logger.info(f"Processed {i + len(batch_texts)} / {len(combined_text)} texts")
          
        self.embeddings = np.vstack(embeddings).astype(self.embed_dtype)
        logger.info(f"Generated embeddings shape: {self.embeddings.shape}")
        return self.embeddings


    def encode_query(self, query: str) -> np.ndarray:
      """
          Encode a single query with 'query:' prefix

          Args:
              query: Search query text
              
          Returns:
              Query embedding as numpy array
      """
      
      query_embedding = self.model.encode([f"query: {query}"], normalize_embeddings=True)
      return query_embedding.astype(self.embed_dtype)


    def save_embeddings(self, embeddings_vals, save_dir: str, experiment_prefix: str):
        """
        Save embeddings
        
        Args:
            save_dir: Directory to save data
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Saving embeddings   

        if embeddings_vals is not None:
            np.save(os.path.join(save_dir, experiment_prefix + "_embeddings.npy"), embeddings_vals)
            logger.info("Saved embeddings")
        
  
        logger.info(f"Embeddings and data saved to {save_dir}")

if __name__ == "__main__":

    # For balanced dataset

    sampled_datavals_product=pd.read_parquet(os.path.join(base_dir_path,"dataset/sampled_dataset/sampled_datavals_product_data.parquet"))

    # Generating embeddings for the dataset using intfloat/multilingual-e5-base model for all product columns

    generate_embeddings_obj = GenerateEmbeddings(model_name="intfloat/multilingual-e5-base")
    product_cols=["product_title","product_description","product_bullet_point","product_brand","product_color"]
    embeddings=generate_embeddings_obj.generate_embeddings(sampled_datavals_product,product_cols)
    generate_embeddings_obj.save_embeddings(embeddings,os.path.join(base_dir_path,"embeddings_generated","balanced_dataset"),"all_product_fields_768dim")

    # Generating embeddings for the dataset using intfloat/multilingual-e5-small model for all product columns except product_description

    generate_embeddings_obj = GenerateEmbeddings(model_name="intfloat/multilingual-e5-small")
    product_cols=["product_title","product_bullet_point","product_brand","product_color"]
    embeddings=generate_embeddings_obj.generate_embeddings(sampled_datavals_product,product_cols)
    generate_embeddings_obj.save_embeddings(embeddings,os.path.join(base_dir_path,"embeddings_generated","balanced_dataset"),"product_fields_exclude_desc_384dim")

    # For imbalanced dataset

    sampled_datavals_product_imbalanced=pd.read_parquet(os.path.join(base_dir_path,"dataset/sampled_dataset/sampled_datavals_product_data_imbalanced.parquet"))

    # Generating embeddings for the dataset using intfloat/multilingual-e5-base model for all product columns

    generate_embeddings_obj = GenerateEmbeddings(model_name="intfloat/multilingual-e5-base")
    product_cols=["product_title","product_description","product_bullet_point","product_brand","product_color"]
    embeddings=generate_embeddings_obj.generate_embeddings(sampled_datavals_product_imbalanced,product_cols)
    generate_embeddings_obj.save_embeddings(embeddings,os.path.join(base_dir_path,"embeddings_generated","imbalanced_dataset"),"all_product_fields_768dim")

    # Generating embeddings for the dataset using intfloat/multilingual-e5-small model for all product columns except product_description

    generate_embeddings_obj = GenerateEmbeddings(model_name="intfloat/multilingual-e5-small")
    product_cols=["product_title","product_bullet_point","product_brand","product_color"]
    embeddings=generate_embeddings_obj.generate_embeddings(sampled_datavals_product_imbalanced,product_cols)
    generate_embeddings_obj.save_embeddings(embeddings,os.path.join(base_dir_path,"embeddings_generated","imbalanced_dataset"),"product_fields_exclude_desc_384dim")




