"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.types import DateType, StringType

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

# Load Data
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")
products = spark.read.parquet(user_path + "input_csv_for_recommend_system/Product_dict.parquet")

# Data Preparation
# convert the contact_id to StringType, conert the sale_date_date to DateType
data = data \
    .select('sale_date_date', 'contact_id', 'shop_id', 'product_id', 'quantity') \
    .withColumn(colName="sale_date_date", col=data["sale_date_date"].cast(DateType())) \
    .withColumn(colName="product_id", col=data["product_id"].cast(StringType()))
products = products.withColumnRenamed(existing='__index_level_0__', new='product_id')

# Number of unique users in our dataset
users = data.select('contact_id').distinct()
print('Number of unique users = ' + str('{0:,}'.format(users.count()).replace(',', '\'')))  # 1'636'831

# extract 90% of customer ID's
(users_train, users_valid) = users.randomSplit(weights=[0.9, 0.1], seed=5)
print(users_train.count(), users_valid.count())  # (1473'217, 163'614)

# split data into train and validation set
train_df = data.join(other=users_train, on='contact_id', how='inner')
validation_df = data.join(other=users_valid, on='contact_id', how='inner')
print(train_df.count(), validation_df.count())  # 


# Введем колонку, определяющую номер чека и уберем лишние колонки
def create_col_orders(df):
    return df \
        .select(F.concat_ws('_', data.sale_date_date, data.shop_id, data.contact_id).alias('order_id'),
                'product_id', 'quantity') \
        .groupBy('order_id') \
        .agg(F.collect_list(col='product_id')) \
        .withColumnRenamed(existing='collect_list(product_id)', new='actual_products')


train_orders = create_col_orders(df=train_df)
validation_orders = create_col_orders(df=validation_df)

# TODO: Проработать необходимость. Отберем чеки с КПЧ меньше 10
# number_of_positions_in_a_check = 10
# orders_less = orders.groupBy('order_id').count() \
#     .where(condition=F.col('count') < number_of_positions_in_a_check) \
#     .select('order_id')
# orders_filtered = orders.join(other=orders_less, on='order_id', how='inner') \
#     .withColumn(colName="product_id", col=orders["product_id"].cast(StringType()))  # Word2Vec принимает только str


# Learn a mapping from words to Vectors.    
word2Vec = Word2Vec(
    vectorSize=100, minCount=5, numPartitions=1, seed=33, windowSize=3,
    inputCol='actual_products', outputCol='result')
model = word2Vec.fit(dataset=train_orders)

# Сохраняем модель
model.save(path='ml_models/word2vec_model_2021_05_11')

# Загрузка модели
loadedModel = Word2VecModel.load(path='ml_models/word2vec_model_2021_05_11')
print('Good saving? -> ' + str(loadedModel.getVectors().first().word == model.getVectors().first().word))

result = model.transform(dataset=validation_orders)

# model.getVectors().show(n=30, truncate=True)

# TODO: Настроить метрику
