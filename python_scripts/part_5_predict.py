"""
/spark2.4/bin/pyspark
"""

import numpy as np
from pyspark.shell import sc
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from pyspark.sql import functions as F
# from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, IntegerType

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
# Load model
model = MatrixFactorizationModel.load(sc, "ml_models/myCollaborativeFilter")


def train_test_split_by_week(df, week_col_name, test_size_weeks):
    """ Разделение на train и test по неделям """
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - test_size_weeks
    df_train = df.filter(F.col(week_col_name) < threshold_week)
    df_test = df.filter(F.col(week_col_name) >= threshold_week)
    return df_train, df_test


def custom_predictions(user_id, num_preds=5):
    """
    Предсказание
    :param user_id: id покупателя
    :param num_preds: количество рекомендуемых товаров
    :return: отсортированный список из рекомендуемых товаров
    """
    return [i.product for i in model.recommendProducts(user=user_id, num=num_preds)]


def transform_for_als(df, user_col_name, item_col_name, rating_col_name):
    """
    Преобразование для ALS
    :param df: исходный датафрейм
    :param user_col_name: имя колонки пользователей/покупателей
    :param item_col_name: имя колонки с товарами
    :param rating_col_name: имя колонки с рейтингом (сумма, количество продаж)
    :return: преобразованный датафрейм
    """
    return df \
        .select(user_col_name, item_col_name, rating_col_name) \
        .groupBy(user_col_name, item_col_name).sum(rating_col_name) \
        .withColumnRenamed(existing='sum(quantity)', new='quantity')


def bought_products(df, user_id):
    list_1 = df \
        .filter(condition=F.col('contact_id') == user_id) \
        .filter(condition=F.col('quantity') >= 1) \
        .orderBy('quantity', ascending=False) \
        .select('product_id') \
        .collect()
    return [i.product_id for i in list_1]


data = spark.read.parquet("input_csv_for_recommend_system/data.parquet") \
    .withColumn(colName='week_of_year', col=F.weekofyear(F.col('sale_date_date')))

data_train, data_test = train_test_split_by_week(df=data, week_col_name='week_of_year', test_size_weeks=3)

train = transform_for_als(data_train, 'contact_id', 'product_id', 'quantity')
test = transform_for_als(data_test, 'contact_id', 'product_id', 'quantity')

udf_bought_products = F.udf(f=(lambda z: bought_products(df=test, user_id=z)), returnType=ArrayType(IntegerType()))
# results =
test \
    .select('contact_id', udf_bought_products('contact_id')) \
    .show(n=10)

# --------------------------
user = 1006083
custom_predictions(user_id=user, num_preds=5),
bought_products(df=test, user_id=user)
