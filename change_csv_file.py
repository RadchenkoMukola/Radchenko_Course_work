import csv
import pandas as pd

from langdetect import detect, LangDetectException

from joblib import Parallel, delayed

from googletrans import Translator

def detect_language_parallel(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'Unknown'

def translate_comments(comment):
    translator = Translator()
    translation = translator.translate(comment, src='auto', dest='en')
    return translation.text

def process_comments(csv_file):
    # Чтение CSV файла
    df = pd.read_csv(csv_file)

    df = df.drop_duplicates(subset=['body'])

    # Создание новых колонок
    df['translation'] = ''
    df['original_locale'] = ''

    # Итерация по всем комментариям
    for index, row in df.iterrows():
        body = row['body']
        # Поиск индекса "(Original)"
        original_index = body.find('(Original)')
        # Если найдено, то обработка перевода
        if original_index != -1:
            # Получение перевода и оригинала
            translation = body[:original_index].strip()
            translation = translation.replace('(Translated by Google)', '').strip()
            original = body[original_index + len('(Original)'):].strip()
            # Обновление колонок
            df.at[index, 'translation'] = translation
            df.at[index, 'body'] = original

    # Второй цикл для обработки комментариев без (Original)
    for index, row in df.iterrows():
        body = row['body']
        if '(Translated by Google)' in body:
            # Получение перевода
            translation = body.split('(Translated by Google)')[1].strip()
            # Обновление колонок
            df.at[index, 'translation'] = translation
            df.at[index, 'body'] = body.split('(Translated by Google)')[0].strip()

    # Ускорение определения языка с использованием многопоточности
    num_cores = 6  # Укажите желаемое количество ядер вашего процессора
    languages = Parallel(n_jobs=num_cores)(
        delayed(detect_language_parallel)(row['body']) for index, row in df.iterrows())

    # Обновление DataFrame с результатами определения языка
    df['original_locale'] = languages

    # Перевод комментариев, которые были оставлены на английском языке или не были переведены
    for index, row in df.iterrows():
        body = row['body']
        translation = row['translation']
        if row['original_locale'] == 'en' and row['translation'] == '':
            translation = body
        if row['translation'] == '' and row['original_locale'] != 'Unknown':
          translation = translate_comments(body)
          print(translation)


        df.at[index, 'translation'] = translation

    # Сохранение обновленного DataFrame в CSV файл
    df.to_csv('processed_comments.csv', index=False)

# Пример вызова функции
process_comments('ReviewsRVZ.csv')

