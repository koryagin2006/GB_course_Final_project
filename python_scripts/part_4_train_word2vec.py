"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec

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


# для начала готовим DataFrames
data = clean_minus_1(df=spark.read.parquet("input_csv_for_recommend_system/data.parquet"))
products = spark.read \
    .parquet("input_csv_for_recommend_system/Product_dict.parquet") \
    .withColumnRenamed(existing='__index_level_0__', new='product_id')

# Введем колонку с номером недели
data = data.withColumn(colName='week_of_year', col=F.weekofyear(F.col('sale_date_date')))

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
    .agg(F.collect_list(col='name'))

# orders_for_w2v.count()  # 5916047

orders_train, orders_test = train_test_split_by_week(df=orders_for_w2v, week_col_name='week_of_year', test_size_weeks=3)
orders_train.show(n=5, truncate=False)
# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="actual_products", outputCol="result")
model = word2Vec.fit(dataset=orders_train)

result = model.transform(dataset=orders_test)
result.show(n=5, truncate=False)
data.select('name').distinct().show(n=50, truncate=False)

synonyms = model.findSynonyms(word='(20694) Зубная паста Рокс Кофе и Табак 74г', num=5)
