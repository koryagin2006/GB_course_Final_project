"""
/spark2.4/bin/pyspark
"""
from pyspark.shell import sc
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

# from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

data = spark.read.parquet("input_csv_for_recommend_system/data.parquet")  # для начала готовим DataFrame

data = data \
    .withColumn(colName='week_of_year', col=F.weekofyear(F.col('sale_date_date')))  # Введем колонку с номером недели


def train_test_split_by_week(df, week_col_name, test_size_weeks):
    """ Разделение на train и test по неделям """
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - test_size_weeks
    df_train = df.filter(F.col(week_col_name) < threshold_week)
    df_test = df.filter(F.col(week_col_name) >= threshold_week)
    return df_train, df_test


data_train, data_test = train_test_split_by_week(df=data, week_col_name='week_of_year', test_size_weeks=3)


def transform_for_als(df, user_col_name, item_col_name, rating_col_name):
    return df \
        .select(user_col_name, item_col_name, rating_col_name) \
        .groupBy(user_col_name, item_col_name).sum(rating_col_name) \
        .withColumnRenamed(existing='sum(quantity)', new='quantity')


train = transform_for_als(data_train, 'contact_id', 'product_id', 'quantity')
test = transform_for_als(data_test, 'contact_id', 'product_id', 'quantity')

# Обучение
model = ALS.trainImplicit(ratings=train, rank=10, alpha=0.01, nonnegative=True, seed=42)

# Save model
model.save(sc, "ml_models/myCollaborativeFilter")
# sameModel = MatrixFactorizationModel.load(sc, "ml_models/myCollaborativeFilter")


