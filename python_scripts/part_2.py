"""
/spark2.4/bin/pyspark \
    --driver-memory 512m \
    --driver-cores 1
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType, FloatType, DateType

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# для начала готовим DataFrame
data = spark.read \
    .options(delimiter=',', inferschema=True, header=True) \
    .csv(path="input_csv_for_recommend_system/data.csv")


data.count()  # 20'000'003
data.show(n=3)
data.printSchema()

"""
>>> data.printSchema()
root
 |-- sale_date_date: string (nullable = true)
 |-- contact_id: string (nullable = true)
 |-- shop_id: integer (nullable = true)
 |-- product_id: integer (nullable = true)
 |-- name: string (nullable = true)
 |-- product_sub_category_id: string (nullable = true)
 |-- product_category_id: string (nullable = true)
 |-- brand_id: string (nullable = true)
 |-- quantity: string (nullable = true)
"""

# Посмотрим число пропусков в каждом столбце.
for col in data.columns:
    print(col, "\t", "with null values: ", data.filter(data[col].isNull()).count())
"""
('sale_date_date', 'count of null values = ', 2)
('contact_id', 'count of null values = ', 2)
('shop_id', 'count of null values = ', 3)
('product_id', 'count of null values = ', 3)
('name', 'count of null values = ', 3)
('product_sub_category_id', 'count of null values = ', 3)
('product_category_id', 'count of null values = ', 3)
('brand_id', 'count of null values = ', 3)
('quantity', 'count of null values = ', 3)
"""

# Посмотрим число -1 в каждом столбце.
for col in data.columns:
    print(col, "count of -1 values = ", data.filter(data[col] == '-1').count())
"""
('sale_date_date', 'count of -1 values = ', 0)
('contact_id', 'count of -1 values = ', 0)
('shop_id', 'count of -1 values = ', 0)
('product_id', 'count of -1 values = ', 193)
('name', 'count of -1 values = ', 0)
('product_sub_category_id', 'count of -1 values = ', 647720)
('product_category_id', 'count of -1 values = ', 649395)
('brand_id', 'count of -1 values = ', 16169893)
('quantity', 'count of -1 values = ', 13874)
"""

# Посмотрим конец таблицы по 2 столбцам
data \
    .sort(F.col("sale_date_date").asc()) \
    .select('sale_date_date', 'contact_id') \
    .show(n=5, truncate=False)
"""
+--------------+-----------+
|sale_date_date|contact_id |
+--------------+-----------+
|null          |null       |
|null          |null       |
|(затронуто стр|к: 20000000|
|2018-01-01    |1970794    |
|2018-01-01    |850958     |
+--------------+-----------+
only showing top 5 rows
"""

# Удалить последние 3 строки в DF
data = data.where(F.col('sale_date_date') != '(затронуто стр')
data.count()  # 20'000'000

# TODO: Исправить типы данных на числовые (даты) где необходимо

# Переведем sale_date_date в формат DateType
data = data.withColumn(colName="sale_date_date", col=data["sale_date_date"].cast(DateType()))

# Переведем contact_id, shop_id, product_id, product_sub_category_id, product_category_id, brand_id в формат IntegerType
data = data \
    .withColumn(colName="contact_id", col=data["contact_id"].cast(IntegerType())) \
    .withColumn(colName="shop_id", col=data['shop_id'].cast(IntegerType())) \
    .withColumn(colName='product_id', col=data['product_id'].cast(IntegerType())) \
    .withColumn(colName='product_sub_category_id', col=data['product_sub_category_id'].cast(IntegerType())) \
    .withColumn(colName='product_category_id', col=data['product_category_id'].cast(IntegerType())) \
    .withColumn(colName='brand_id', col=data['brand_id'].cast(IntegerType()))

# Переведем quantity в формат FloatType
data = data.withColumn(colName='quantity', col=F.regexp_replace(str='quantity', pattern=',', replacement='.'))
data = data.withColumn(colName='quantity', col=data['quantity'].cast(FloatType()))

# TODO: Решить, что делать со значениями '-1'


# TODO: Пересохранение файла в формат .parquet
# data.write.parquet(path="input_csv_for_recommend_system/data.parquet")
data.write.json(path="input_csv_for_recommend_system/data.json")
