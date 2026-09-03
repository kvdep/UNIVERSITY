import psycopg2

'''
1. Создание таблиц базы данных по варианту (см. далее список заданий). Вариант выбирается в соответствии с номером по журналу
по формуле:
№варианта = (Номер по журналу % 20) + 1, где % - операция взятия остатка от деления.
Вначале создайте саму базу данных на сервере PostgreSQL (с помощью pgAdmin или другого клиента). В своей программе
реализуйте функцию, которая создает структуру базы данных – создает все указанные таблицы (названия таблиц и столбцов
задайте латинскими буквами (таблиц - во множественном числе), без пр
'''

# Функция для создания структуры базы данных
def create_database_structure():
    try:
        # Подключение к PostgreSQL
        connection = psycopg2.connect(
            dbname="shops",
            user="postgres",
            password="123",
            host="localhost",  # или IP-адрес сервера
            port="5432"       # порт PostgreSQL
        )
        connection.autocommit = True
        cursor = connection.cursor()

        # Создание таблицы Магазины (Магазины)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Магазины (
                Номер SERIAL PRIMARY KEY,
                Название VARCHAR(255) NOT NULL,
                Адрес VARCHAR(255) NOT NULL,
                Телефон VARCHAR(20)
            );
        """)

        print("Таблица Магазины успешно создана.")

        # Создание таблицы Комплектующие (Комплектующие)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Комплектующие (
                Модель SERIAL PRIMARY KEY,
                Название VARCHAR(255) NOT NULL,
                Фирма_производитель VARCHAR(255),
                Цена NUMERIC(15, 2) NOT NULL
            );
        """)

        print("Таблица Комплектующие успешно создана.")

        # Создание таблицы Наличие (Наличие)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Наличие (
                Магазин INTEGER REFERENCES Магазины(Номер),
                Модель INTEGER REFERENCES Комплектующие(Модель),
                Год INTEGER CHECK (Год >= 2018 AND Год <= 2020),
                Количество INTEGER NOT NULL,
                PRIMARY KEY (Магазин, Модель, Год)
            );
        """)

        print("Таблица Наличие успешно создана.")

    except Exception as e:
        print(f"Ошибка при создании структуры базы данных: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()
            print("Соединение с PostgreSQL закрыто.")

# Вызов функции
if __name__ == "__main__":
    create_database_structure()