# Подготовка набора данных на локальной машине

#### Импорт библиотек и данных (сначала маленький датасет)

```python
import pandas as pd

data_small = pd.read_csv(filepath_or_buffer='../data/data.csv', nrows=5)
```

### Посмотрим, какие есть колонки

```python
data_small.columns.tolist()
```

```text
['sale_date_date',
 'contact_id',
 'shop_id',
 'product_id',
 'name',
 'product_sub_category_id',
 'product_category_id',
 'brand_id',
 'quantity']
```

### Загружаем весь датасет, но только выбранные колонки

```python
% % time
cols = ['sale_date_date', 'contact_id', 'shop_id', 'product_id', 'product_sub_category_id', 'product_category_id',
        'brand_id', 'quantity']
data = pd.read_csv(filepath_or_buffer='../data/data.csv', usecols=cols, nrows=19999997)
```

```text
Wall time: 1min 5s
```

#### Общая информация о `data`

```python
data.info(verbose=False)
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19999997 entries, 0 to 19999996
Columns: 8 entries, sale_date_date to quantity
dtypes: int64(6), object(2)
memory usage: 1.2+ GB
```

### Предварительная очистка

##### Изменяем типы

```python
data['quantity'] = data['quantity'].str.replace(pat=',', repl='.', regex=False).astype('float')
data['sale_date_date'] = data['sale_date_date'].astype('datetime64')

data[['contact_id', 'shop_id', 'product_id', 'product_sub_category_id', 'product_category_id', 'brand_id']] = \
    data[['contact_id', 'shop_id', 'product_id', 'product_sub_category_id', 'product_category_id', 'brand_id']].astype(int)
```

```python
quantity_neg_1 = data['quantity'] != -1
product_neg_1 = data['product_id'] != -1
product_sub_category_neg_1 = data['product_sub_category_id'] != -1
product_category_neg_1 = data['product_category_id'] != -1

data = data[quantity_neg_1 & product_neg_1 & product_sub_category_neg_1 & product_category_neg_1]
```

### Сохранение в `.parquet`

```python
data.to_parquet(path='../data/data.parquet')
```
