import psycopg2
conn = psycopg2.connect(
    dbname="demo",
    user="postgres",
    password="123",
    host="localhost",  # или IP-адрес сервера
    port="5432"       # порт PostgreSQL
)
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'Наличие'")
print("Столбцы таблицы Наличие:", cur.fetchall())
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'Магазины'")
print("Столбцы таблицы Магазины:", cur.fetchall())

cur.execute("SELECT * FROM \"Наличие\" WHERE \"Количество\" > 100")
print("Записи с количеством > 100:", cur.fetchall())