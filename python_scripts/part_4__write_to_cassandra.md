### Cassandra через консоль
#### Запускаем cassandra
```bash
/cassandra/bin/cqlsh
```

#### Таблица products
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

### Cassandra через spark
#### Запускаем spark-приложение
```bash
/spark2.4/bin/pyspark --packages com.datastax.spark:spark-cassandra-connector_2.11:2.4.2
```

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
products_df = spark \
    .read.parquet(user_path + "input_csv_for_recommend_system/Product_dict.parquet") \
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

cassandra_products_df.show(n=5, truncate=False)
```
```bash
+----------+-------------------------------------------------------------------+
|product_id|name                                                               |
+----------+-------------------------------------------------------------------+
|104124    |(54701) Раствор Ликосол-2000 для конт.линз фл 240мл 817            |
|92248     |(97549) Риностоп спрей наз. 0,05% фл. 15мл 701                     |
|350363    |(270994) Кларитросин табл.п.п.о. 500мг №10 403                     |
|129004    |(73584) Флуконазол-Тева капс. 150мг №1 622                         |
|125915    |(111778) Валсартан-Гидрохлоротиазид таб. п.п.о. 80мг+12,5мг №28 738|
+----------+-------------------------------------------------------------------+
```