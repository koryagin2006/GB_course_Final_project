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
model = Word2VecModel.load(path='ml_models/word2vec_model_2021_05_11')


# Data Preparation
# convert the contact_id to StringType, conert the sale_date_date to DateType
data = data \
    .select('sale_date_date', 'contact_id', 'shop_id', 'product_id', 'quantity') \
    .withColumn(colName="sale_date_date", col=data["sale_date_date"].cast(DateType())) \
    .withColumn(colName="product_id", col=data["product_id"].cast(StringType()))
products = products.withColumnRenamed(existing='__index_level_0__', new='product_id')

# extract 90% of customer ID's
users = data.select('contact_id').distinct()
(users_train, users_valid) = users.randomSplit(weights=[0.9, 0.1], seed=5)

# split data into train and validation set
train_df = data.join(other=users_train, on='contact_id', how='inner')
validation_df = data.join(other=users_valid, on='contact_id', how='inner')


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


# result = model.transform(dataset=validation_orders)

model.getVectors().show(n=5, truncate=True)

train_orders.show(n=10, truncate=True)
"""
+--------------------+--------------------+
|            order_id|     actual_products|
+--------------------+--------------------+
|2018-01-01_137_10...|[102814, 141059, ...|
|2018-01-01_1453_1...|            [139810]|
|2018-01-01_1455_1...|            [116399]|
|2018-01-01_1460_1...|[59701, 168308, 1...|
|2018-01-01_1468_1...|             [74945]|
|2018-01-01_1473_1...|[43401, 59746, 75...|
|2018-01-01_1478_1...|             [82717]|
|2018-01-01_1480_1...|            [146946]|
|2018-01-01_1502_1...|             [26975]|
|2018-01-01_1512_1...|             [88610]|
+--------------------+--------------------+
"""


# tdidf
# Word frequency, ie tf
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel

 
 # vocabSize is the size of the total vocabulary, minDF is the minimum number of occurrences in the text
cv = CountVectorizer(inputCol="actual_products", outputCol="countFeatures", vocabSize=200 * 10000, minDF=1.0)
 # Training word frequency statistical model
cv_model = cv.fit(train_orders)
cv_model.write().overwrite().save(user_path + "ml_models/CV.model")
cv_model = CountVectorizerModel.load(user_path + "ml_models/CV.model")

# Get the word frequency vector result
cv_result = cv_model.transform(train_orders)
cv_result.show(n=10)
"""
+--------------------+--------------------+--------------------+
|            order_id|     actual_products|       countFeatures|
+--------------------+--------------------+--------------------+
|2018-01-01_137_10...|[102814, 141059, ...|(28639,[430,706,1...|
|2018-01-01_1453_1...|            [139810]|(28639,[1062],[1.0])|
|2018-01-01_1455_1...|            [116399]| (28639,[414],[1.0])|
|2018-01-01_1460_1...|[59701, 168308, 1...|(28639,[0,16,99,1...|
|2018-01-01_1468_1...|             [74945]|(28639,[2071],[1.0])|
|2018-01-01_1473_1...|[43401, 59746, 75...|(28639,[307,7209,...|
|2018-01-01_1478_1...|             [82717]| (28639,[487],[1.0])|
|2018-01-01_1480_1...|            [146946]|(28639,[2075],[1.0])|
|2018-01-01_1502_1...|             [26975]| (28639,[903],[1.0])|
|2018-01-01_1512_1...|             [88610]|(28639,[1941],[1.0])|
+--------------------+--------------------+--------------------+
"""

# idf
from pyspark.ml.feature import IDF, IDFModel
 
idf = IDF(inputCol="countFeatures", outputCol="idfFeatures")
idf_model = idf.fit(cv_result)
idf_model.write().overwrite().save(user_path + "ml_models/IDF.model")
idf_model = IDFModel.load(user_path + "ml_models/IDF.model")

# tf-idf 

tfidf_result = idf_model.transform(cv_result)
tfidf_result.show(n=10, truncate=True)
"""
+--------------------+--------------------+--------------------+--------------------+
|            order_id|     actual_products|       countFeatures|         idfFeatures|
+--------------------+--------------------+--------------------+--------------------+
|2018-01-01_137_10...|[102814, 141059, ...|(28639,[430,706,1...|(28639,[430,706,1...|
|2018-01-01_1453_1...|            [139810]|(28639,[1062],[1.0])|(28639,[1062],[7....|
|2018-01-01_1455_1...|            [116399]| (28639,[414],[1.0])|(28639,[414],[6.5...|
|2018-01-01_1460_1...|[59701, 168308, 1...|(28639,[0,16,99,1...|(28639,[0,16,99,1...|
|2018-01-01_1468_1...|             [74945]|(28639,[2071],[1.0])|(28639,[2071],[7....|
|2018-01-01_1473_1...|[43401, 59746, 75...|(28639,[307,7209,...|(28639,[307,7209,...|
|2018-01-01_1478_1...|             [82717]| (28639,[487],[1.0])|(28639,[487],[6.6...|
|2018-01-01_1480_1...|            [146946]|(28639,[2075],[1.0])|(28639,[2075],[7....|
|2018-01-01_1502_1...|             [26975]| (28639,[903],[1.0])|(28639,[903],[7.2...|
|2018-01-01_1512_1...|             [88610]|(28639,[1941],[1.0])|(28639,[1941],[7....|
+--------------------+--------------------+--------------------+--------------------+
"""


 # Select the first 20 as keywords, here is only a word index
def sort_by_tfidf(partition):
    TOPK = 20
    for row in partition:
                 # Find index and IDF value and sort
        _dict = list(zip(row.idfFeatures.indices, row.idfFeatures.values))
        _dict = sorted(_dict, key=lambda x: x[1], reverse=True)
        result = _dict[:TOPK]
        for word_index, tfidf in result:
            yield row.article_id, row.channel_id, int(word_index), round(float(tfidf), 4)


keywords_by_tfidf = tfidf_result.rdd.mapPartitions(sort_by_tfidf) \
    .toDF(["article_id", "channel_id", "index", "weights"])
 
 # Build keywords and index
keywords_list_with_idf = list(zip(cv_model.vocabulary, idf_model.idf.toArray()))

 
def append_index(data):
    for index in range(len(data)):
                 data[index] = list(data[index]) # Convert tuples to list
                 data[index].append(index) # Add index
        data[index][1] = float(data[index][1])

 
append_index(keywords_list_with_idf)
sc = spark.sparkContext
 rdd = sc.parallelize(keywords_list_with_idf) # Create rdd
idf_keywords = rdd.toDF(["keywords", "idf", "index"])
 
 # Find article keywords and weights tfidf
keywords_result = keywords_by_tfidf.join(idf_keywords, idf_keywords.index == keywords_by_tfidf.index).select(
    ["article_id", "channel_id", "keywords", "weights"])
 print("Keyword Weight", keywords_result.take(10))
 
 # Article keywords and word vectors join
keywords_vector = keywords_result.join(vectors, vectors.word == keywords_result.keywords, 'inner')