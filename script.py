import pickle
import pandas as pd
import logging


# Настройка базовых конфигураций логов
def configure_logging(level):
    logging.basicConfig(
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s.%(msecs)03d] %(module)10s:%(lineno)-3d %(levelname)-7s - %(message)s",
    )


logger = logging.getLogger(__name__)
configure_logging(level=logging.INFO)

# Импорт данных
data = pd.read_parquet('test.parquet')


# Первый этап обработки данных
def step_1(df):
    # Распарсим наши данные так, чтобы даты стали новыми фичами
    def row_to_dict(row):
        row_dict = {'id': row.iloc[0]}
        for date, value in zip(row.iloc[1], row.iloc[2]):
            row_dict[date] = value
        return row_dict

    new_data = pd.DataFrame(df.apply(row_to_dict, axis=1).tolist())
    new_data.columns = new_data.columns.astype(str)
    logger.info("Первый этап обработки данных завершен")
    return new_data


def step_2(df):
    # Создаем новые фичи
    mean = df.iloc[:, 1:].apply(lambda row: row.dropna().mean(), axis=1).rename('mean')  # среднее значение
    median = df.iloc[:, 1:].apply(lambda row: row.dropna().median(), axis=1).rename('median')  # медиана
    variance = df.iloc[:, 1:].apply(lambda row: row.dropna().var(), axis=1).rename('variance')  # дисперсия
    sign_sum = df.iloc[:, 1:].apply(lambda row: 1 if row.dropna().sum() >= 0 else 0, axis=1).rename(
        'sign_sum')  # знак суммы значений ряда
    sign_last_value = df.iloc[:, 1:].apply(
        lambda row: 1 if not row.dropna().empty and row.dropna().iloc[-1] >= 0 else 0, axis=1).rename(
        'sign_last_value')  # знак последнего значения
    count_non_nan = df.iloc[:, 1:].apply(lambda row: row.dropna().count(), axis=1).rename(
        'count_non_nan')  # количество наблюдений в ряду

    concat_df = pd.concat([df, mean, median, variance, sign_sum, sign_last_value, count_non_nan], axis=1)
    concat_df.columns = concat_df.columns.astype(str)
    logger.info("Второй этап обработки данных завершен")
    return concat_df


def step_3(df):
    # Загрузка обученной модели из pickle файла
    with open('model.pkl', 'rb') as file:
        trained_model = pickle.load(file)

    # Сохраняем id
    ids = df['id']

    # Делаем прогноз
    pred = trained_model.predict_proba(df)[:, 1]
    new_submission = pd.DataFrame({'id': ids, 'score': pred})
    new_submission.to_csv('new_submission.csv', index=False)
    # new_submission
    logger.info("submission файл создан")


def main():
    data_1 = step_1(data)
    data_2 = step_2(data_1)
    data_3 = step_3(data_2)


if __name__ == "__main__":
    main()