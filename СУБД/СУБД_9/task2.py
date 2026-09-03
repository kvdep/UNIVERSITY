'''
2. Напишите функции, которые добавляют данные в таблицы. Для каждой таблицы напишите два варианта функций: первый вариант
функции добавляет одну строку данных, второй вариант – добавляет все указанные в задании строки. В каждую из функций данные
(т.е. одна строка или набор строк) передаются в качестве параметров. Т.е. и в запрос, который выполняется с помощью execute
(executemany), эти данные тоже приходят в качестве параметров (нужно использовать параметризованные запросы INSERT).
'''
import psycopg2


def add_magazin_one(connection, name, address, phone=None):
    """Добавляет одну запись в таблицу Магазины"""
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO Магазины (Название, Адрес, Телефон)
                VALUES (%s, %s, %s);
            """, (name, address, phone))
        connection.commit()
        print(f"Магазин '{name}' добавлен.")
    except Exception as e:
        connection.rollback()
        print(f"Ошибка при добавлении магазина: {e}")


def add_magaziny_many(connection, magaziny_list):
    """Добавляет несколько записей в таблицу Магазины"""
    try:
        with connection.cursor() as cursor:
            cursor.executemany("""
                INSERT INTO Магазины (Название, Адрес, Телефон)
                VALUES (%s, %s, %s);
            """, magaziny_list)
        connection.commit()
        print(f"Магазины добавлены.")
    except Exception as e:
        connection.rollback()
        print(f"Ошибка при добавлении магазинов: {e}")

connection = psycopg2.connect(
    dbname="shops",
    user="postgres",
    password="123",
    host="localhost",  # или IP-адрес сервера
    port="5432"       # порт PostgreSQL
)
add_magazin_one(connection, 'Книжный мир', 'ул. Пушкина, 10', '8-800-555-35-35')
add_magaziny_many(connection, [
    ('Электроника', 'пр. Ленина, 25', '8-900-123-45-67'),
    ('Продукты', 'ул. Гагарина, 5', None),
    ('Одежда', 'пл. Свободы, 1', '8-911-987-65-43')
])
connection.close()
