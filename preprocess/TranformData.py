import re
import pandas as pd
df = pd.read_csv('/Users/nattakul/Downloads/Data_2567.csv')


def transform_condition(condition):

    co_subjects = re.findall(r"(?:CON|CO)\d{6}-\d{3,4}", condition)
    #print(co_subjects)
    # แทนที่ CO ด้วยข้อความ 'ต้องเรียนไปพร้อมกัน'
    for sub in co_subjects:
        if "CON" in sub:
            condition = condition.replace(sub, f"ต้องเรียนวิชารหัส {sub[3:]} ไปพร้อมกัน")
            #print(f'CON - {condition}')
        elif "CO" in sub:
            condition = condition.replace(sub, f"ต้องเรียนวิชารหัส {sub[2:]} ไปพร้อมกัน")
            print(f'CO - {condition}')
        else:
            condition = condition.replace(sub, f"รหัสวิชา {sub}")
            #print(f'NoOOOOO - {condition}')

    condition = condition.replace("และ", " และรหัสวิชา ")

    return f"นักศึกษาต้องสอบผ่านวิชาที่มี {condition}"


df["เงื่อนไขรายวิชา"] = df["เงื่อนไขรายวิชา"].apply(transform_condition)

df.to_csv("/Users/nattakul/Downloads/Data_2567_Conditions.csv", index=False, encoding="utf-8")

