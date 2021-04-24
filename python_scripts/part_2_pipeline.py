"""
/spark2.4/bin/pyspark \
    --driver-memory 512m \
    --driver-cores 1
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType, DateType

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# для начала готовим DataFrame
data = spark.read \
    .options(delimiter=',', inferschema=True, header=True) \
    .csv(path="input_csv_for_recommend_system/data.csv")

# Удалить последние 3 строки в DF
data = data.where(F.col('sale_date_date') != '(затронуто стр')

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

# TODO: Пересохранение файла в формат .parquet
# data.write.parquet(path="input_csv_for_recommend_system/data.parquet")
# data.write.json(path="input_csv_for_recommend_system/data.json")

# Выдает файл с размером = 0
