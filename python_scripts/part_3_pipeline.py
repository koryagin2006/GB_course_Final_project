"""
/spark2.4/bin/pyspark
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType, DateType

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# для начала готовим DataFrame
data = spark.read.parquet("input_csv_for_recommend_system/data.parquet")
data.show(n=5, truncate=True)

# Введем колонку с номером недели
data = data.withColumn('week_of_year', F.weekofyear(F.col('sale_date_date')))


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


# Разделим набор данных на тренировочную и тестовую выборки
data_train, data_test = train_test_split_by_week(df=data,
                                                 week_col_name='week_of_year',
                                                 test_size_weeks=3)
