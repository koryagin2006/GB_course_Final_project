"""
/spark2.4/bin/pyspark
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

data = spark.read.parquet("input_csv_for_recommend_system/data.parquet")


def clean_minus_1(df):
    return df \
        .where(F.col('product_id') != '-1') \
        .where(F.col('product_sub_category_id') != '-1') \
        .where(F.col('product_category_id') != '-1')


data = clean_minus_1(df=data)
data.count()  # 19 303 330

# Пересохранение файла в формат .parquet - выдает ошибку, не сохраняет
# data.write.parquet(path="input_csv_for_recommend_system/data.parquet", mode='overwrite')
