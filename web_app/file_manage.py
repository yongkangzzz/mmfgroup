import os
from flask import request
import sqlite3
import random
from datetime import datetime
def generate_random_str(randomlength=16):
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return random_str


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + "Directory created successfully")
        return True
    else:
        print(path + "Directory has already existed!")
        return False

def clean(path):
    clear_dir(path)
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        os.rmdir(path_file)

    print("clean finished")

def clear_dir(path):
    isExists = os.path.exists(path)
    if not isExists:
        print("no such dir")
        return
    else:
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)

def check_database(object):
    db = 'user_info.sqlite3'
    #db = 'sqlite:///user_info.sqlite3'
    con = sqlite3.connect(db)
    cur = con.cursor()
    select_sql = "select file_name,expired_time from user_info"
    cur.execute(select_sql)
    date_set = cur.fetchall()
    for dir in os.listdir("./static/user"):
        flag = 0
        for row in date_set:
            print(row)
            if dir == row[0] and datetime.now() < datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S.%f"):
                flag = 1
        if flag == 0:
            file_path = './static/user/' + dir
            clear_dir(file_path)
            os.rmdir(file_path)

    cur.close()
    con.close()