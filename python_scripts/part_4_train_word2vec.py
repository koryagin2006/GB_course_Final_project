"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.types import DateType, StringType

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"


def train_test_split_by_week(df, week_col_name, test_size_weeks):
    """ Разделение на train и test по неделям """
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - test_size_weeks
    df_train = df.filter(F.col(week_col_name) < threshold_week)
    df_test = df.filter(F.col(week_col_name) >= threshold_week)
    return df_train, df_test


# Create DataFrames
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")
products = spark.read.parquet(user_path + "input_csv_for_recommend_system/Product_dict.parquet")

# Prepare DFs (rename, drop columns etc.)
products = products.withColumnRenamed(existing='__index_level_0__', new='product_id')
data = data \
    .select('sale_date_date', 'contact_id', 'shop_id', 'product_id', 'quantity') \
    .withColumn(colName="sale_date_date", col=data["sale_date_date"].cast(DateType()))

# data = data.withColumn(colName='week_of_year', col=F.weekofyear(F.col('sale_date_date')))

# Join `data` and `products`
data = data.join(other=products, on='product_id', how='inner')

# Введем колонку, определяющую номер чека и уберем лишние колонки
orders = data \
    .select(F.concat_ws('_', data.sale_date_date, data.shop_id, data.contact_id).alias('order_id'),
            'product_id', 'quantity')

# Отберем чеки с КПЧ меньше 10
number_of_positions_in_a_check = 10
orders_less = orders \
    .groupBy('order_id').count() \
    .where(condition=F.col('count') < number_of_positions_in_a_check) \
    .select('order_id')

orders_filtered = orders \
    .join(other=orders_less, on='order_id', how='inner') \
    .withColumn(colName="product_id", col=orders["product_id"].cast(StringType()))  # Word2Vec принимает только str

orders_for_w2v = orders_filtered \
    .groupBy('order_id') \
    .agg(F.collect_list(col='product_id')) \
    .withColumnRenamed(existing='collect_list(product_id)', new='actual_products')

(train, test) = orders_for_w2v.randomSplit(weights=[0.9, 0.1], seed=3)

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="actual_products", outputCol="result")
model = word2Vec.fit(dataset=train)

result = model.transform(dataset=test)

# Сохраняем модель
word2Vec.save(path='ml_models/word2vec_2021_05_05_by_prod_id')
model.save(path='ml_models/word2vec-model_2021_05_05_by_prod_id')

# Загрузка модели
loadedWord2Vec = Word2Vec.load(path='ml_models/word2vec_2021_05_05_by_prod_id')
loadedModel = Word2VecModel.load(path='ml_models/word2vec-model_2021_05_05_by_prod_id')

#  Проверка, что все хорошо сохранилось
print(loadedWord2Vec.getVectorSize() == word2Vec.getVectorSize())
print(loadedModel.getVectors().first().word == model.getVectors().first().word)

# model.getVectors().show(n=30, truncate=False)

# TODO: Настроить метрику
# TODO: Настроить перевзвешивание tf или брать sum(quantity) / max(sum(quantity) over users)