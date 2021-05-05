hdfs dfs -du -h ml_models

hdfs dfs -rm -r input_csv_for_recommend_system/data.parquet

hdfs dfs -put for_recomend_system/data.parquet input_csv_for_recommend_system

hdfs dfs -rm -r ml_models/my_als_2021-04-28
hdfs dfs -rm -r ml_models/my_als_2021-05-05_samlpe_20_percents
hdfs dfs -rm -r ml_models/my_als_model_2021-05-05_samlpe_20_percents
