'''
4. Напишите функции для каждой таблицы, которые выполняют обновление ее полей. Предусмотрите выполнение условий
сохранения целостности данных. Значения, которыми нужно обновить данные в таблице, должны быть переданы в
соответствующую функцию в качестве параметра.
'''

import psycopg2
conn = psycopg2.connect(
    dbname="demo",
    user="postgres",
    password="123",
    host="localhost",  
    port="5432"      
)


def update_stock(model,shop_num,quantity):
    cur = conn.cursor()
    try: 
        cur.execute('UPDATE Наличие SET Количество = %s WHERE Модель = %s AND Магазин = %s', (quantity, model, shop_num))
        conn.commit()
        print(f'Successfully updated stock!')
    except Exception as e:        
        conn.rollback();print("Error updating stock!", e)
        
    cur.close()


def update_parts_price(model_num, price):
    cur = conn.cursor()
    try: 
        cur.execute('UPDATE Комплектующие SET Цена = %s WHERE Модель = %s', (price, model_num))
        conn.commit()
        print(f'Successfully updated price!')
    except Exception as e:
        conn.rollback();print(f'Error updating price! {e}')

    cur.close()

def update_shop_phone_number(shop_num, phone_number):
    cur = conn.cursor()
    try: 
        cur.execute('UPDATE Магазины SET Телефон = %s WHERE Номер = %s', (phone_number, shop_num))
        conn.commit()
        print(f'Successfully updated phone number!')
    except Exception as e:
        conn.rollback();print(f'Error updating phone number! {e}')

    cur.close()

update_stock(1,336,100); update_parts_price(900, 251); update_shop_phone_number(9, '5553535');
conn.close()
