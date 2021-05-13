# Исследовательский анализ на spark

#### Запускаем spark

```bash
/spark2.4/bin/pyspark
```

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, StructField, IntegerType, DoubleType

user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
```

## Создание DataFrame

```python
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")
```

## Обзор

```python
data.printSchema()
```

```shell
root
 |-- sale_date_date: timestamp (nullable = true)
 |-- contact_id: integer (nullable = true)
 |-- shop_id: integer (nullable = true)
 |-- product_id: integer (nullable = true)
 |-- product_sub_category_id: integer (nullable = true)
 |-- product_category_id: integer (nullable = true)
 |-- brand_id: integer (nullable = true)
 |-- quantity: double (nullable = true)
 |-- __index_level_0__: long (nullable = true)
```

```python
data.show(n=5, truncate=False)
print('Count rows of dataset: {}'.format(data.count()))
```

```shell
+-------------------+----------+-------+----------+-----------------------+-------------------+--------+--------+-----------------+
|sale_date_date     |contact_id|shop_id|product_id|product_sub_category_id|product_category_id|brand_id|quantity|__index_level_0__|
+-------------------+----------+-------+----------+-----------------------+-------------------+--------+--------+-----------------+
|2018-12-07 00:00:00|1260627   |1455   |168308    |906                    |205                |-1      |1.0     |0                |
|2018-12-07 00:00:00|198287    |279    |134832    |404                    |93                 |-1      |1.0     |1                |
|2018-12-07 00:00:00|2418385   |848    |101384    |404                    |93                 |-1      |1.0     |2                |
|2018-12-07 00:00:00|1285774   |1511   |168570    |906                    |205                |-1      |1.0     |3                |
|2018-12-07 00:00:00|1810323   |1501   |168319    |906                    |205                |-1      |1.0     |4                |
+-------------------+----------+-------+----------+-----------------------+-------------------+--------+--------+-----------------+

Count rows of dataset: 19346684
```

#### Посмотрим число пропусков в каждом столбце.

```python
for col in data.columns:
    print('null values in {} = {}'.format(col, data.filter(data[col].isNull()).count()))
```

```shell

null values in sale_date_date = 0
null values in contact_id = 0
null values in shop_id = 0
null values in product_id = 0
null values in product_sub_category_id = 0
null values in product_category_id = 0
null values in brand_id = 0
null values in quantity = 0
null values in __index_level_0__ = 0
```

#### Посмотрим число -1 в каждом столбце.

```python
for col in data.columns:
    if data.filter(data[col] == '-1').count() > 0:
        print('count of -1 values in {} = {}'.format(col, data.filter(data[col] == '-1').count()))
```

```shell
count of -1 values in brand_id = 15605526
```

#### Описание колонки `quantity`

```python
data.select('quantity').describe().show()
```

```shell
+-------+------------------+
|summary|          quantity|
+-------+------------------+
|  count|          19346684|
|   mean|1.3729506576930124|
| stddev|2.4194553333532656|
|    min|             0.001|
|    max|            2454.0|
+-------+------------------+
```

#### Число уникальных значений в каждой колонке

```python
data.select([F.countDistinct(col).alias(col) for col in data.columns]).show()
```

```shell
+--------------+----------+-------+----------+-----+-----------------------+-------------------+--------+--------+
|sale_date_date|contact_id|shop_id|product_id| name|product_sub_category_id|product_category_id|brand_id|quantity|
+--------------+----------+-------+----------+-----+-----------------------+-------------------+--------+--------+
|           214|   1642379|    851|     36549|36113|                    440|                145|    1617|    1296|
+--------------+----------+-------+----------+-----+-----------------------+-------------------+--------+--------+
```