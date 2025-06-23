import os
import numpy as np
import faiss
import logging

base_dir_path=os.getcwd()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorIndexGenerator:
    """
    Class for generating vector indices.
    """
    def __init__(self,embeddings_file_name,embeddings_path, vector_indexes_path):
        self.embeddings_dir = os.path.join(base_dir_path,embeddings_path)
        self.index_save_dir = os.path.join(base_dir_path,vector_indexes_path)
        self.loaded_embeddings = self.load_embeddings(embeddings_file_name)
        self.embeddings = self.loaded_embeddings
        self.embedding_dim = self.embeddings.shape[1]
        self.indices = {}


    def load_embeddings(self, embeddings_file_name):
        """
        Load embeddings from a file.
        """
        embeddings_path = os.path.join(self.embeddings_dir, embeddings_file_name)
        logger.info(f"Loading embeddings from {embeddings_path}")
        if os.path.exists(embeddings_path):
            loaded_embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings with shape {loaded_embeddings.shape}")
            logger.info(f"Loaded embeddings with shape {loaded_embeddings.shape}")
            return loaded_embeddings
        else:
            logger.error(f"Embeddings file not found at {embeddings_path}")
            return None
    
    def create_flat_index(self) -> faiss.IndexFlatIP:
        """
        Create FAISS Flat index (exact search with inner product/cosine similarity)
        
        Returns:
            FAISS Flat index
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not available.")
        
        logger.info("Creating Flat index...")
        logger.info(self.embedding_dim)
        index_flat = faiss.IndexFlatIP(self.embedding_dim)
        index_flat.add(self.embeddings)
        
        self.indices['flat'] = index_flat
        logger.info(f"Flat index created with {index_flat.ntotal} vectors")
        return index_flat

    def create_ivf_index(self, n_centroids:int=12) -> faiss.IndexIVFFlat:
        """
        Create FAISS IVF (Inverted File) index for faster approximate search
        
        Args:
            nlist: Number of clusters/centroids
            
        Returns:
            FAISS IVF index
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not available.")
        
        logger.info(f"Creating IVF index with {n_centroids} clusters...")
        
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        # rule of thumb is that you need at least 39 * n_centroids training points for effective clusteringn
        # n_centroids = min(500 // 39, your_desired_centroids) = 12
        index_ivf = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_centroids)
        
        # Train the index
        logger.info("Training IVF index...")
        index_ivf.train(self.embeddings)
        index_ivf.add(self.embeddings)
        
        self.indices['ivf'] = index_ivf
        logger.info(f"IVF index created with {index_ivf.ntotal} vectors")
        return index_ivf

    
    def create_hnsw_index(self, M: int = 16, ef_construction: int = 128, ef_search: int = 32) -> faiss.IndexHNSWFlat:
        """
        Create FAISS HNSW (Hierarchical Navigable Small World) index

        Args:
            M: Number of bi-directional links for new elements
            ef_construction: Size of dynamic candidate list
            
        Returns:
            FAISS HNSW index
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not available. Set embeddings generator first.")

        logger.info(f"Creating HNSW index with M={M}, ef_construction={ef_construction}...")

        index_hnsw = faiss.IndexHNSWFlat(self.embedding_dim, M)
        index_hnsw.hnsw.ef_construction = ef_construction
        index_hnsw.hnsw.ef_search = ef_search
        index_hnsw.add(self.embeddings)

        self.indices['hnsw'] = index_hnsw
        logger.info(f"HNSW index created with {index_hnsw.ntotal} vectors")
        return index_hnsw


    
    def create_ivf_pq_index(self, nlist: int = 4, m: int = 4, nbits: int = 3) -> faiss.IndexIVFPQ:
        """
        Create FAISS IVF+PQ index combining clustering and compression
        
        Args:
            nlist: Number of clusters
            m: Number of sub-quantizers
            nbits: Number of bits per sub-quantizer
            
        Returns:
            FAISS IVF+PQ index
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not available.")
        
        logger.info(f"Creating IVF+PQ index with nlist={nlist}, m={m}, nbits={nbits}...")
        
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index_ivf_pq = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, nbits)
        
        # Train the index
        logger.info("Training IVF+PQ index...")
        index_ivf_pq.train(self.embeddings)
        index_ivf_pq.add(self.embeddings)
        
        self.indices['ivf_pq'] = index_ivf_pq
        logger.info(f"IVF+PQ index created with {index_ivf_pq.ntotal} vectors")
        return index_ivf_pq

    def write_index(self, index_obj, index_folder, index_name: str) -> None:
        """
        Write the index to a file.
        
        Args:
            index_name: Name of the index file
        """
        faiss.write_index(index_obj, os.path.join(self.index_save_dir, index_folder, index_name))
        logger.info(f"Index written to {index_name}")

if __name__ == "__main__":

    # For balanced dataset

    # Creating indexes for all_product_fields_768dim_embeddings.npy

    experiment_prefix = "all_product_fields_768dim_"

    embeddings_file_name=experiment_prefix + "embeddings.npy"
    embeddings_path="embeddings_generated/balanced_dataset"
    vector_indexes_path="vector_indexes/balanced_dataset"

    vector_index_obj=VectorIndexGenerator(embeddings_file_name,embeddings_path,vector_indexes_path)

    index_folder="768dim"

    # creating Flat Index
    flat_index = vector_index_obj.create_flat_index()
    vector_index_obj.write_index(flat_index,index_folder,experiment_prefix+"flat.index")

    # creating IVF Index
    ivf_index = vector_index_obj.create_ivf_index()
    vector_index_obj.write_index(ivf_index,index_folder,experiment_prefix+"ivf.index")

    # creating HNSW Index
    hnsw_index = vector_index_obj.create_hnsw_index()
    vector_index_obj.write_index(hnsw_index,index_folder,experiment_prefix+"hnsw.index")

    # creating IVF+PQ Index
    ivf_pq_index = vector_index_obj.create_ivf_pq_index()
    vector_index_obj.write_index(ivf_pq_index,index_folder,experiment_prefix+"ivf_pq.index")


    # Creating indexes for product_fields_exclude_desc_384dim_embeddings.npy

    experiment_prefix = "product_fields_exclude_desc_384dim_"

    embeddings_file_name=experiment_prefix + "embeddings.npy"
    vector_index_obj=VectorIndexGenerator(embeddings_file_name,embeddings_path,vector_indexes_path)
    index_folder="384dim"

    # creating Flat Index
    flat_index = vector_index_obj.create_flat_index()
    vector_index_obj.write_index(flat_index,index_folder,experiment_prefix+"flat.index")

    # creating IVF Index
    ivf_index = vector_index_obj.create_ivf_index()
    vector_index_obj.write_index(ivf_index,index_folder,experiment_prefix+"ivf.index")

    # creating HNSW Index
    hnsw_index = vector_index_obj.create_hnsw_index()
    vector_index_obj.write_index(hnsw_index,index_folder,experiment_prefix+"hnsw.index")

    # creating IVF+PQ Index
    ivf_pq_index = vector_index_obj.create_ivf_pq_index()
    vector_index_obj.write_index(ivf_pq_index,index_folder,experiment_prefix+"ivf_pq.index")




    # For imbalanced dataset

    # Creating indexes for all_product_fields_768dim_embeddings.npy

    experiment_prefix = "all_product_fields_768dim_"

    embeddings_file_name=experiment_prefix + "embeddings.npy"
    embeddings_path="embeddings_generated/imbalanced_dataset"
    vector_indexes_path="vector_indexes/imbalanced_dataset"

    vector_index_obj=VectorIndexGenerator(embeddings_file_name,embeddings_path,vector_indexes_path)

    index_folder="768dim"

    # creating Flat Index
    flat_index = vector_index_obj.create_flat_index()
    vector_index_obj.write_index(flat_index,index_folder,experiment_prefix+"flat.index")

    # creating IVF Index
    ivf_index = vector_index_obj.create_ivf_index()
    vector_index_obj.write_index(ivf_index,index_folder,experiment_prefix+"ivf.index")

    # creating HNSW Index
    hnsw_index = vector_index_obj.create_hnsw_index()
    vector_index_obj.write_index(hnsw_index,index_folder,experiment_prefix+"hnsw.index")

    # creating IVF+PQ Index
    ivf_pq_index = vector_index_obj.create_ivf_pq_index()
    vector_index_obj.write_index(ivf_pq_index,index_folder,experiment_prefix+"ivf_pq.index")


    # Creating indexes for product_fields_exclude_desc_384dim_embeddings.npy

    experiment_prefix = "product_fields_exclude_desc_384dim_"

    embeddings_file_name=experiment_prefix + "embeddings.npy"
    vector_index_obj=VectorIndexGenerator(embeddings_file_name,embeddings_path,vector_indexes_path)
    index_folder="384dim"

    # creating Flat Index
    flat_index = vector_index_obj.create_flat_index()
    vector_index_obj.write_index(flat_index,index_folder,experiment_prefix+"flat.index")

    # creating IVF Index
    ivf_index = vector_index_obj.create_ivf_index()
    vector_index_obj.write_index(ivf_index,index_folder,experiment_prefix+"ivf.index")

    # creating HNSW Index
    hnsw_index = vector_index_obj.create_hnsw_index()
    vector_index_obj.write_index(hnsw_index,index_folder,experiment_prefix+"hnsw.index")

    # creating IVF+PQ Index
    ivf_pq_index = vector_index_obj.create_ivf_pq_index()
    vector_index_obj.write_index(ivf_pq_index,index_folder,experiment_prefix+"ivf_pq.index")