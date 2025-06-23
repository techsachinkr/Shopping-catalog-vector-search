Download Source dataset

Run following commands if you navigated to this directory in command line

cd ../../../
git clone https://github.com/amazon-science/esci-data.git
cd esci-data
git lfs pull
cd  Shopping-catalog-vector-search
cp  ../esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet  ./dataset/source_files/shopping_queries_dataset_examples.parquet
cp  ../esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet  ./dataset/source_files//shopping_queries_dataset_products.parquet
