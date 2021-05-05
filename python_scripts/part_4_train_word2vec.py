"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.types import DateType

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()


def train_test_split_by_week(df, week_col_name, test_size_weeks):
    """ Разделение на train и test по неделям """
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - test_size_weeks
    df_train = df.filter(F.col(week_col_name) < threshold_week)
    df_test = df.filter(F.col(week_col_name) >= threshold_week)
    return df_train, df_test


# для начала готовим DataFrames
data = spark.read.parquet("hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/data/input_csv_for_recommend_system/data.parquet")
data = data.withColumn(colName="sale_date_date", col=data["sale_date_date"].cast(DateType()))

products = spark.read \
    .parquet("input_csv_for_recommend_system/Product_dict.parquet") \
    .withColumnRenamed(existing='__index_level_0__', new='product_id')

# Введем колонку с номером недели
data = data.withColumn(colName='week_of_year', col=F.weekofyear(F.col('sale_date_date')))

# Соединим к data name из products
data = data.join(other=products, on='product_id', how='inner')

# Введем колонку, определяющую номер чека и уберем лишние колонки
orders = data.select(F.concat_ws('__', data.sale_date_date, data.shop_id, data.contact_id).alias('order_id'),
                     'name', 'quantity', 'week_of_year')

# Отберем чеки с КПЧ меньше 10
max_num_items = 10
orders_less = orders \
    .groupBy('order_id').count() \
    .where(condition=F.col('count') < max_num_items) \
    .select('order_id')

orders_filtered = orders.join(other=orders_less, on='order_id', how='inner')
# orders_filtered.count()  # 16 683 162

orders_for_w2v = orders_filtered \
    .groupBy('order_id', 'week_of_year') \
    .agg(F.collect_list(col='name')) \
    .withColumnRenamed(existing='collect_list(name)', new='actual_products')

orders_train, orders_test = train_test_split_by_week(df=orders_for_w2v, week_col_name='week_of_year', test_size_weeks=3)
orders_train = orders_train.select('order_id', 'actual_products')
orders_test = orders_test.select('order_id', 'actual_products')

orders_train.show(n=5, truncate=True)
# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="actual_products", outputCol="result")
model = word2Vec.fit(dataset=orders_train)

result = model.transform(dataset=orders_test)
result.show(n=5, truncate=True)

# Сохраняем модель
word2Vec.save(path='ml_models/word2vec_2021_05_04')
model.save(path='ml_models/word2vec-model_2021_05_04')

# Загрузка модели
loadedWord2Vec = Word2Vec.load(path='ml_models/word2vec_2021_05_04')
loadedModel = Word2VecModel.load(path='ml_models/word2vec-model_2021_05_04')

#  Проверка, что все хорошо сохранилось
print(loadedWord2Vec.getVectorSize() == word2Vec.getVectorSize())
print(loadedModel.getVectors().first().word == model.getVectors().first().word)

model.getVectors().show(n=30, truncate=False)

required_product = '(60668) Редуксин капс. 15мг №60 686'
print(required_product)


def my_synonyms(word, num_synonyms=5):
    return model \
        .findSynonyms(word=word, num=num_synonyms) \
        .select('word').withColumnRenamed(existing='word', new='similar for {}'.format(required_product))


my_synonyms(word=required_product, num_synonyms=3).show(truncate=False)