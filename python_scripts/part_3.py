"""
/spark2.4/bin/pyspark
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# для начала готовим DataFrame
data = spark.read.parquet("input_csv_for_recommend_system/data.parquet")
data.show(n=5, truncate=True)

# Посмотрим диапазон дат
data.select(
    F.min('sale_date_date').alias('first_date'),
    F.max('sale_date_date').alias('last_date'),
    F.datediff(F.max('sale_date_date'), F.min('sale_date_date')).alias('datediff')
).show(truncate=False)

# Разряженность матрицы = 0.0333%
n_users = data.select(F.countDistinct(col='contact_id')).collect()[0][0]
n_items = data.select(F.countDistinct(col='product_id')).collect()[0][0]
n_interactions = data.count()
print(n_users, n_items, n_interactions)

# популярность - группируем товары по сумме продаж
popularity_by_items = data \
    .groupBy("product_id").sum("quantity") \
    .withColumnRenamed(existing='sum(quantity)', new='total_quantity')

popularity_by_items.select('total_quantity').describe().show(truncate=False)
popularity_by_items \
    .orderBy('total_quantity', ascending=False) \
    .withColumn(colName="total_quantity", col=popularity_by_items["total_quantity"].cast(IntegerType(), )) \
    .show(n=5)

# популярность - группируем пользователей по сумме продаж
popularity_by_users = data \
    .groupBy("contact_id").sum("quantity") \
    .withColumnRenamed(existing='sum(quantity)', new='total_quantity')

popularity_by_users.select('total_quantity').describe().show(truncate=False)
popularity_by_users.orderBy('total_quantity', ascending=False) \
    .withColumn(colName="total_quantity", col=popularity_by_users["total_quantity"].cast(IntegerType(), )) \
    .show(n=5)

# Введем колонку с номером недели
data = data.withColumn('week_of_year', F.weekofyear(F.col('sale_date_date')))

# Посмотрим диапазон между первой и последней неделей
data.select(F.min('week_of_year'), F.max('week_of_year'),
            (F.max('week_of_year') - F.min('week_of_year')).alias('week_diff')).show(truncate=False)


def train_test_split_by_week(df, week_col_name, test_size_weeks):
    """
    Разделение на train и test по неделям
    :param df: исходный датафрейм
    :param week_col_name: название колонки с номерами недели в году
    :param test_size_weeks: число недель для теста
    :return: 2 датасета
    """
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - test_size_weeks
    train = df.filter(F.col(week_col_name) < threshold_week)
    test = df.filter(F.col(week_col_name) >= threshold_week)
    return train, test


data_train, data_test = train_test_split_by_week(df=data, week_col_name='week_of_year', test_size_weeks=3)
data_test.show()



