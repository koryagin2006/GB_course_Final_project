"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()


def clean_minus_1(df):
    return df \
        .where(F.col('product_id') != '-1') \
        .where(F.col('product_sub_category_id') != '-1') \
        .where(F.col('product_category_id') != '-1')


def train_test_split_by_week(df, week_col_name, test_size_weeks):
    """ Разделение на train и test по неделям """
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - test_size_weeks
    df_train = df.filter(F.col(week_col_name) < threshold_week)
    df_test = df.filter(F.col(week_col_name) >= threshold_week)
    return df_train, df_test


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
        .withColumn(colName='sum(quantity)', col=F.when(condition=F.col('sum(quantity)') > 0, value=1)) \
        .withColumnRenamed(existing='sum(quantity)', new='rating')


# для начала готовим DataFrame
data = clean_minus_1(df=spark.read.parquet("input_csv_for_recommend_system/data.parquet"))
# Введем колонку с номером недели
data = data.withColumn(colName='week_of_year', col=F.weekofyear(F.col('sale_date_date')))

data_train, data_test = train_test_split_by_week(df=data, week_col_name='week_of_year', test_size_weeks=3)
train = transform_for_als(data_train, 'contact_id', 'product_id', 'quantity')
test = transform_for_als(data_test, 'contact_id', 'product_id', 'quantity')

# Обучение
als = ALS(maxIter=5, regParam=0.1, userCol="contact_id", itemCol="product_id",
          ratingCol="rating", implicitPrefs=False)
model = als.fit(dataset=train)

# Save model
model.save(path="ml_models/my_als_2021-04-28")
model.save(path="ml_models/my_als_2021-05-03")

# Load model
model = ALSModel.load('ml_models/my_als_2021-04-28')

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(dataset=test)
predictions.orderBy(['contact_id', 'prediction'], ascending=[True, False]).show(n=20, truncate=False)
predictions.orderBy('prediction', ascending=False).show(n=20, truncate=False)

# check the root mean squared error
evaluator = RegressionEvaluator(metricName='rmse', predictionCol='prediction', labelCol='quantity')
rmse = evaluator.evaluate(predictions)
print('Root mean squared error of the test_data: %.4f' % rmse)

# Generate top 5 products recommendations for each contact
userRecs = model.recommendForAllUsers(numItems=5)

userRecs.printSchema()

userRecs \
    .select('contact_id', F.col("recommendations").cast("String")) \
    .printSchema()

userRecs.show(n=10, truncate=False)
