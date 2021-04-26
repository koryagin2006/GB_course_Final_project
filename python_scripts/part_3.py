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
data = data.withColumn('week_of_year', F.weekofyear(F.col('sale_date_date'))).select('sale_date_date', 'week_of_year')

# Посмотрим диапазон неделей
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



def popularity_recommendation(data, k=5):
    """Топ-k популярных товаров"""
    popularity_by_items = data \
        .groupBy("product_id").sum("quantity") \
        .withColumnRenamed(existing='sum(quantity)', new='sum_quantity') \
        .orderBy('sum_quantity', ascending=False) \
        .select('product_id') \
        .head(n=k)
    return [int(row.product_id) for row in popularity_by_items]

popular_recs = popularity_recommendation(data=data_train, k=5)
popular_recs



data_train \
    .groupBy("contact_id").pivot("product_id").sum("quantity") \
    .show(n=5)
# pyspark.sql.utils.AnalysisException: u'The pivot column product_id has more than 10000 distinct values, this could indicate an error. If this was intended, set spark.sql.pivotMaxValues to at least the number of distinct values of the pivot column.;'

data.select(F.countDistinct('product_id')).show()
+--------------------------+
|count(DISTINCT product_id)|
+--------------------------+
|                     36549|
+--------------------------+

train = data_train \
    .withColumnRenamed('contact_id', 'user') \
    .withColumnRenamed('product_id', 'item') \
    .select('user', 'item', 'quantity') \
    .groupBy('user', 'item').sum('quantity')

test = data_test \
    .withColumnRenamed('contact_id', 'user') \
    .withColumnRenamed('product_id', 'item') \
    .select('user', 'item', 'quantity') \
    .groupBy('user', 'item').sum('quantity')



'---------------------------------------------------------------------------'

from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

df = spark.createDataFrame(
	[(0, 0, 4.0), (0, 1, 2.0), (1, 1, 3.0), (1, 2, 4.0), (2, 1, 1.0), (2, 2, 5.0)], 
	["user", "item", "rating"])
df.show()

"""
+----+----+------+
|user|item|rating|
+----+----+------+
|   0|   0|   4.0|
|   0|   1|   2.0|
|   1|   1|   3.0|
|   1|   2|   4.0|
|   2|   1|   1.0|
|   2|   2|   5.0|
+----+----+------+
"""

als = ALS(rank=10, maxIter=5, seed=0)
model = als.fit(dataset=df)
model.rank  # 10

model \
	.userFactors \
	.orderBy("id") \
	.collect()  # [Row(id=0, features=[...]), Row(id=1, ...), Row(id=2, ...)]

test = spark.createDataFrame([(0, 2), (1, 0), (2, 0)], ["user", "item"])
test.show()
"""
+----+----+
|user|item|
+----+----+
|   0|   2|
|   1|   0|
|   2|   0|
+----+----+
"""
predictions = sorted(model.transform(test).collect(), key=lambda r: r[0])

predictions[0]  # Row(user=0, item=2, prediction=-0.13807615637779236)
predictions[1]  # Row(user=1, item=0, prediction=2.6258413791656494)
predictions[2]  # Row(user=2, item=0, prediction=-1.5018409490585327)

user_recs = model.recommendForAllUsers(3)

user_recs \
    .where(user_recs.user == 0) \
    .select("recommendations.item", "recommendations.rating") \
    .collect()  # [Row(item=[0, 1, 2], rating=[3.910..., 1.992..., -0.138...])]

item_recs = model.recommendForAllItems(3)
item_recs \
    .where(item_recs.item == 2) \
    .select("recommendations.user", "recommendations.rating") \
    .collect()  # [Row(user=[2, 1, 0], rating=[4.901..., 3.981..., -0.138...])]

user_subset = df.where(df.user == 2)
user_subset_recs = model.recommendForUserSubset(user_subset, 3)
user_subset_recs \
    .select("recommendations.item", "recommendations.rating") \
    .first()  # Row(item=[2, 1, 0], rating=[4.901..., 1.056..., -1.501...])
    
item_subset = df.where(df.item == 0)
item_subset_recs = model.recommendForItemSubset(item_subset, 3)
item_subset_recs	\
	.select("recommendations.user", "recommendations.rating") \
	.first()  # Row(user=[0, 1, 2], rating=[3.910..., 2.625..., -1.501...])

als_path = temp_path + "/als"
als.save(als_path)
als2 = ALS.load(als_path)
als.getMaxIter()  # 5

model_path = temp_path + "/als_model"
model.save(model_path)
model2 = ALSModel.load(model_path)
model.rank == model2.rank  # True

sorted(model.userFactors.collect()) == sorted(model2.userFactors.collect())  # True
sorted(model.itemFactors.collect()) == sorted(model2.itemFactors.collect())  # True
