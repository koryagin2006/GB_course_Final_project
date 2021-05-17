# Выдача рекомендаций

```bash
/spark2.4/bin/pyspark
```

## ALS - predict

#### Подготовка объектов

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
import time

user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"
model_als_path = "ml_models/my_als_model_2021-05-11_last_15_weeks.model_als"


def load_model(user_path, model_path):
    return ALSModel.load(path=user_path + model_path)


# Load model_als and data
model_als = load_model(user_path=user_path, model_path=model_als_path)
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")


def predict_als(user_id, n_recs=3, model=model_als):
    start = time.time()
    preds_dict = {}
    recs_list = model
    .recommendForAllUsers(numItems=n_recs)
    .where(condition=F.col('user_id') == user_id)
    .withColumn(colName="rec_exp", col=F.explode("recommendations"))
    .select(F.col("rec_exp.item_id"))


#
preds_dict['user_id'] = user_id
preds_dict['recommendations'] = [int(row.item_id) for row in recs_list.collect()]
preds_dict['prediction time'] = round(number=time.time() - start, ndigits=3)
return preds_dict
```

#### Сделаем предсказание для `contact_id` = 471

```python
print(predict_als(user_id=471, n_recs=3))
```

```python
{
    'user_id': 471,
    'prediction time': 153.663,
    'recommendations': [162780, 135427, 46797]
}
```