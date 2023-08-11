import pandas as pd
import numpy as np
from typing import Type

# Разделение датасета
from sklearn.model_selection import train_test_split
if __name__ == '__main__':

    ABALONE_DATASET_PATH = 'https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets/abalone.csv'
    def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.
        """
        start_mem = df.memory_usage().sum() / 1024 ** 2
        # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

        # end_mem = df.memory_usage().sum() / 1024 ** 2
        # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df


    def import_data(file) -> pd.DataFrame:
        """create a dataframe and optimize its memory usage"""
        df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
        df = reduce_mem_usage(df)
        return df


    df = import_data(ABALONE_DATASET_PATH)

    df.columns = df.columns.str.replace(' ', '_')

    df['Sex'].astype('category')


    class DataPreprocessing():
        """Подготовка исходных данных"""


        def __init__(self) -> None:
            """Параметры класса"""
            self.medians = None
            self.modes = None

        def fit(self, data: pd.DataFrame) -> None:
            """Сохранение статистик"""
            # Расчет статистик
            self.medians = data.median(numeric_only=True)

        def transform(self, data: pd.DataFrame, y: pd.Series) -> tuple([pd.DataFrame, pd.Series]):

            """Трансформация данных"""

            data = pd.concat([data, y], axis=1)

            # удаление строк с нулевым значением высоты
            data = data[data['Height'] != 0]

            # удаление строк, где вес частей меньше общего веса
            data = data[~((data['Whole_weight'] <= data['Shucked_weight']) |
                          (data['Whole_weight'] <= data['Viscera_weight']) |
                          (data['Whole_weight'] <= data['Shell_weight']))]

            # обработка пропущенных значений
            data.fillna(self.medians, inplace=True)

            # замена значений признака пола с f на F
            data['Sex'] = data['Sex'].replace('f', 'F')

            y = data['Rings']
            X = data.drop('Rings', axis=1)

            return X, y


    y = df['Rings']
    X = df.drop(['Rings'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False, random_state=7575)

    # объявляем объект препроцессор и получаем статистические характеристики по обучающей выборке
    preprocessor = DataPreprocessing()
    preprocessor.fit(X_train)

    # применим преоборазования для наших данных
    X_train, y_train = preprocessor.transform(X_train, y_train)
    X_test, y_test = preprocessor.transform(X_test, y_test)

    train_prep = pd.concat([X_train, y_train], axis=1)
    test_prep = pd.concat([X_test, y_test], axis=1)

    train_prep.to_csv('train_prep.csv', index=False)
    test_prep.to_csv('test_prep.csv', index=False)



