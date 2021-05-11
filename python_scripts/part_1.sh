hdfs dfs -du -h ml_models
hdfs dfs -ls ml_models
hdfs dfs -rm -r input_csv_for_recommend_system/data.parquet

hdfs dfs -put for_recomend_system/data.parquet input_csv_for_recommend_system


hdfs dfs -rm -r ml_models/CV.model
hdfs dfs -rm -r ml_models/IDF.model
hdfs dfs -rm -r ml_models/myCollaborativeFilter
hdfs dfs -rm -r ml_models/my_LR_model_8
hdfs dfs -rm -r ml_models/my_LR_model_enrollees
hdfs dfs -rm -r ml_models/word2vec_2021_05_05_by_prod_id
hdfs dfs -rm -r ml_models/word2vec_2021_05_04
hdfs dfs -rm -r ml_models/word2vec-model_2021_05_04
hdfs dfs -rm -r ml_models/word2vec-model_2021_05_05_by_prod_id

