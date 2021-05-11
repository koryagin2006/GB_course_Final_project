# Перенос исходного файла на сервер

### Скопируем подготовленный файл на удаленный сервер с помощью команды `- scp`.

NB! Эта команда запускается на локальном компьютере, а не на удалённом сервере.

```shell
cd ./GB_course_Final_project/

scp -i id_rsa_305_koryagin.txt -r ./data/data.paruet 305_koryagin@37.139.32.56:~/for_recomend_system
scp -i id_rsa_305_koryagin.txt -r ./data/Product_dict.parquet 305_koryagin@37.139.32.56:~/for_recomend_system
```

### Подключаемся к серверy

```shell
ssh 305_koryagin@37.139.32.56 -i ./id_rsa_305_koryagin.txt
```

#### Проверяем записанные данные

```shell
ls -l for_recomend_system/
```

<details>
    <summary> → вывод консоли TERMINAL</summary>

```shell
-rwx------ 1 305_koryagin 305_koryagin 2331217025 Apr 23 17:33 data.csv
-rw-r--r-- 1 305_koryagin 305_koryagin  262165701 May  4 17:13 data.parquet
-rw-r--r-- 1 305_koryagin 305_koryagin    1797698 Apr 25 07:59 Product_dict.parquet
```

</details>

#### Создадим папку input_csv_for_stream на HDFS, из которой стрим будет читать файлы и скопируем файл на HDFS. Cкопируем файл на HDFS из папки на локальном сервере

```shell
hdfs dfs -mkdir input_csv_for_recommend_system

hdfs dfs -put for_recomend_system/data.parquet input_csv_for_recommend_system
hdfs dfs -put for_recomend_system/Product_dict.parquet input_csv_for_recommend_system
hdfs dfs -ls input_csv_for_recommend_system
```

<details>
    <summary> → вывод консоли SSH</summary>

```shell
Found 2 items
-rw-r--r--   2 305_koryagin 305_koryagin    1797698 2021-04-25 08:01 input_csv_for_recommend_system/Product_dict.parquet
-rw-r--r--   2 305_koryagin 305_koryagin  262165701 2021-05-04 17:33 input_csv_for_recommend_system/data.parquet
```

</details>
