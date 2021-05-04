hdfs dfs -du -h ml_models

hdfs dfs -rm -r input_csv_for_recommend_system/data.parquet

hdfs dfs -put for_recomend_system/data.parquet input_csv_for_recommend_system

hdfs dfs -ls ml_models