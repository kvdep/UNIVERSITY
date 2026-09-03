'''
5. Напишите функции для каждой таблицы, которые выполняют 
а) удаление данных по некоторому условию – значения для условий передавайте в качестве параметров в функцию; 
б) удаление всех данных. Предусмотрите выполнение условий сохранения целостности данных
'''

import psycopg2
conn = psycopg2.connect(
    dbname="demo",
    user="postgres",
    password="123",
    host="localhost",  # или IP-адрес сервера
    port="5432"       # порт PostgreSQL
)
#cursor = conn.cursor()

def delete_on_condition(table_name, condition, value):
    cur = conn.cursor()
    try:
        cur.execute(f'DELETE FROM {table_name} WHERE {condition} = %s', (value,))
        conn.commit()
        print(f'Успешно, удалено по {condition} по величине {value}')
    except:
        conn.rollback()
        print('Коллапс(')
    
    cur.close()

def delete_all(table_name):
    cur = conn.cursor()
    try:
        cur.execute(f'DELETE FROM {table_name}')
        conn.commit()
    except:
        conn.rollback()
        print('Коллапс(')
    cur.close()

delete_on_condition('Наличие', 'Количество', 100)
#delete_all('Наличие')
conn.close()
