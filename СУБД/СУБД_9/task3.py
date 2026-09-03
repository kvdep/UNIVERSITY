'''
3. Напишите функции, которые выполняют выборки из таблиц. Для каждой таблицы варианты функций должны быть такими:
выбрать все данные из таблицы; 
выбрать данные, в которых значение(-я) поля(-ей) равно(-ы) заданному(-ым) – параметризованные
запросы, заданное(-ые) значение(-я) передае(ю)тся в функцию в качестве параметра(-ов); 
выбрать данные, полученные после соединения таблиц, в запросе должен быть хотя бы один параметр. 
Все параметры, которые присутствуют в запросах, должны передаваться в функцию при ее вызове.
'''
import psycopg2
def select_all_from_table(conn, table_name):
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table_name};")
        rows = cur.fetchall()
        return rows
    except psycopg2.Error as e:
        print(f"Error selecting all from {table_name}: {e}")
        return None

def select_from_table_where(conn, table_name, conditions):
    try:
        cur = conn.cursor()
        where_clauses = [f"{col} = %s" for col in conditions.keys()]
        query = f"SELECT * FROM {table_name} WHERE {' AND '.join(where_clauses)};"
        cur.execute(query, list(conditions.values()))
        rows = cur.fetchall()
        return rows
    except psycopg2.Error as e:
        print(f"Error selecting from {table_name} with conditions {conditions}: {e}")
        return None

def select_with_join(conn, table1, table2, join_on_1, join_on_2, condition, cond_val):
    try:
        cur = conn.cursor()
        query = f"""
            SELECT t1.*, t2.*
            FROM {table1} t1
            JOIN {table2} t2 ON t1.{join_on_1} = t2.{join_on_2}
            WHERE {condition} > %s;
        """
        cur.execute(query, (cond_val,))
        rows = cur.fetchall()
        return rows
    except psycopg2.Error as e:
        print(f"Error selecting with join: {e}")
        return None

if __name__ == '__main__':
    conn = psycopg2.connect(
    dbname="demo",
    user="postgres",
    password="123",
    host="localhost",  
    port="5432" 
)
    try:
        print("Выбор всех строчек таблицы Магазины':")
        Магазины = select_all_from_table(conn, "Магазины")
        if Магазины:
            for row in Магазины:
                print(row)

        print("\nВывод всех строчек, где телефон магазина равен '8-800-556-17-89:")
        courses_with_3_credits = select_from_table_where(conn, "Магазины", {"Телефон": "5561789"})
        if courses_with_3_credits:
            for row in courses_with_3_credits:
                print(row)

        print("\nСоединение таблиц по номеру магазина и вывод строчек, где количество товара > 100")
        student_enrollments = select_with_join(conn, "Наличие", "Магазины", "Магазин", "Номер", 'Количество', 100)
        if student_enrollments:
            for row in student_enrollments:
                print(row)

    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
    finally:
        if conn:
            conn.close()
            