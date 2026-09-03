import psycopg2

# Функция для создания структуры базы данных
def create_database_structure():
    try:
        # Подключение к PostgreSQL
        connection = psycopg2.connect(
            dbname="demo",
            user="postgres",
            password="123",
            host="localhost",  # или IP-адрес сервера
            port="5432"       # порт PostgreSQL
        )
        connection.autocommit = True
        cursor = connection.cursor()

       
        print("успешно подключены к PostgreSQL.")

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