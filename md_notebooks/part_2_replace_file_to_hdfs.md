# Перенос исходного файла на сервер

### Скопируем подготовленный файл на удаленный сервер

NB! Эта команда запускается на локальном компьютере, а не на удаленном сервере.

```bash
cd ./GB_course_Final_project/

scp -i id_rsa_305_koryagin.txt -r ./data/data.paruet 305_koryagin@37.139.32.56:~/for_recomend_system
scp -i id_rsa_305_koryagin.txt -r ./data/Product_dict.parquet 305_koryagin@37.139.32.56:~/for_recomend_system
```

#### Проверяем записанные данные

```bash
ls -l for_recomend_system/
```

```shell
-rwx------ 1 305_koryagin 305_koryagin 2331217025 Apr 23 17:33 data.csv
-rw-r--r-- 1 305_koryagin 305_koryagin  262165701 May  4 17:13 data.parquet
-rw-r--r-- 1 305_koryagin 305_koryagin    1797698 Apr 25 07:59 Product_dict.parquet
```

#### Создадим папку input_csv_for_stream на HDFS, из которой будем читать файлы и скопируем файлы на HDFS из папки на локальном сервере

```bash
hdfs dfs -mkdir input_csv_for_recommend_system

hdfs dfs -put for_recomend_system/data.parquet input_csv_for_recommend_system
hdfs dfs -put for_recomend_system/Product_dict.parquet input_csv_for_recommend_system
hdfs dfs -ls input_csv_for_recommend_system
```

```shell
Found 2 items
-rw-r--r--   2 305_koryagin 305_koryagin    1797698 2021-04-25 08:01 input_csv_for_recommend_system/Product_dict.parquet
-rw-r--r--   2 305_koryagin 305_koryagin  262165701 2021-05-04 17:33 input_csv_for_recommend_system/data.parquet
```

### Запись файла `Product_dict.parquet` в Cassanara

#### Создаем таблицу `products`
```bash
/cassandra/bin/cqlsh
```

```sql
CREATE KEYSPACE IF NOT EXISTS final_project 
   WITH REPLICATION = {
      'class' : 'SimpleStrategy', 'replication_factor' : 1 };

USE final_project;

CREATE TABLE IF NOT EXISTS products (
	product_id int,
	name text,
    primary key (product_id));
```

#### Записываем данные из файла в таблицу базы
```bash
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
```

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
products_df = spark.read \
	.parquet(user_path + "input_csv_for_recommend_system/Product_dict.parquet") \
    .withColumnRenamed(existing='__index_level_0__', new='product_id')

# Пишем в cassandra
products_df.write \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="products", keyspace="final_project") \
    .mode("append") \
    .save()

# Проверяем записанное
cassandra_products_df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="products", keyspace="final_project") \
    .load()

cassandra_products_df.show(n=3, truncate=False)
```
```shell
+----------+-------------------------------------------------------------------+
|product_id|name                                                               |
+----------+-------------------------------------------------------------------+
|104124    |(54701) Раствор Ликосол-2000 для конт.линз фл 240мл 817            |
|92248     |(97549) Риностоп спрей наз. 0,05% фл. 15мл 701                     |
|350363    |(270994) Кларитросин табл.п.п.о. 500мг №10 403                     |
+----------+-------------------------------------------------------------------+
```
