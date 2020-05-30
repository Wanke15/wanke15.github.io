### 高并发场景下的数据一致性

高并发场景下由于线程每一步完成的顺序和效率不一样，就可能导致数据一致性的问题。
对于传统的数据库而言，一般会采用悲观锁和乐观锁的方法保证数据的一致性，但再短时间内执行大量的SQL，对于服务器的压力可想而知，需要优化数据库的表设计、索引、SQL语句等。
也有通过Redis和Lua语言提供的原子性来取代现有的数据库技术，提高数据的存储响应，以应对高并发场景。
在这里通过一个抢红包的场景来模拟乐观锁。悲观锁和Redis相关的尝试后续再补充
``` python
import random
import threading
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor

import MySQLdb


class MySQL:
    """
    Postgres client.
    """

    def __init__(self):
        self.database = "recommender"
        self.user = "root"
        self.password = "123456"
        self.host = "127.0.0.1"
        self.port = "3306"

        self.db = MySQLdb.connect("localhost", self.user, self.password, self.database, charset='utf8')

        self.cursor = self.db.cursor()


class RedPacket:
    def __init__(self, db_instance):
        self.db_instance = db_instance

    def create_packet(self, amount, pid=None, version=0):
        try:
            if pid:
                real_sql = "insert into red_pocket values('{}', {}, {})".format(pid, amount, version)
            else:
                pid = uuid.uuid4().__str__()
                real_sql = "insert into red_pocket values('{}', {}, {})".format(pid, amount, version)
            self.db_instance.cursor.execute(real_sql)
            self.db_instance.db.commit()
        except:
            try:
                real_sql = "update red_pocket set amount = {}, version = {} where id='{}'".format(amount, version, pid)
                self.db_instance.cursor.execute(real_sql)
                self.db_instance.db.commit()
            except Exception as e:
                # Rollback in case there is any error
                self.db_instance.db.rollback()
        finally:
            return pid

    def get_money(self, person, pid):
        # 模拟不同的请求发起时间
        time.sleep(random.random() * 1)
        try:
            self.db_instance.cursor.execute("select amount, version from red_pocket where id='{}'".format(pid))
            cur_amt = self.db_instance.cursor.fetchone()
            m = min(round(random.random(), 5), cur_amt[0])
            upd_amt = round(cur_amt[0] - m, 5)
        except Exception as e:
            print('\nSelect ERROR: {}'.format(e))
            raise e
        if cur_amt[0] == 0:
            print('\nPerson: {} 红包已抢光!'.format(person))
            return -1
        else:
            print('\nPerson: {}, current: {}'.format(person, cur_amt[0]))

        # 模拟不同的业务逻辑处理时长
        time.sleep(random.random() * 1)

        try:
            real_sql = "update red_pocket set amount = {}, version = {} where id='{}' and version={}".format(upd_amt, cur_amt[
                1] + 1, pid, cur_amt[1])
            # self.db_instance.cursor.execute(real_sql)
            # self.db_instance.db.commit()
            result = self.db_instance.cursor.execute(real_sql)
            self.db_instance.db.commit()
            if result:
                print('\nPerson: {} version: {}, 抢到: {}, 当前余额: {}'.format(person, cur_amt[1], m, upd_amt))
            else:
                print('\nPerson: {} 抢红包失败'.format(person))
                for _ in range(2):
                    self.get_money(person, pid)
        except Exception as e:
            print('\nPerson: {} 抢红包失败: {}'.format(person, e))
            self.db_instance.db.rollback()

        return m


if __name__ == '__main__':
    mysql = MySQL()

    packet_instance = RedPacket(mysql)
    pid = packet_instance.create_packet(10, 'fe253e80-9455-4926-802a-6a032c4f6252', 0)

    # a = 0
    # while True:
    #     a += 1
    #     print("-"*50+'\n')
    #     money = packet_instance.get_money(a, pid)
    #     if money <= 0:
    #         break

    thread_pool = ThreadPoolExecutor(20)
    futs = [thread_pool.submit(packet_instance.get_money, _+1, pid) for _ in range(100)]

```

<img height=436 width=272 src=./assets/optimistic_lock_1.png>
