import re
import pandas as pd
import ast
file_path = 'Data_2567.csv'

data = pd.read_csv(file_path)

data.head()

def thai(text):
    thai_text = re.sub(r'[^\u0E00-\u0E7F\s]', '', text)
    return thai_text.strip()

def typess(details):
    details = re.sub(r'ประเภทการสอน : C', 'ประเภทการสอน: Lecture(ทฤษฏี)', details)
    details = re.sub(r'ประเภทการสอน : T', 'ประเภทการสอน: Tue(ติวแนวข้อสอบ)', details)
    details = re.sub(r'ประเภทการสอน : L', 'ประเภทการสอน: Lab(ปฏิบัติ)', details)
    return details

def typess2(details):
    details = re.sub(r'\bบ\.', 'วิชาบังคับของสาขา', str(details))
    details = re.sub(r'\bล\.','วิชาเลือกของสาขา',str(details))
    details = re.sub(r'\bบล\.','วิชาบังคับเลือกของสาขา',str(details))
    return details 

def typess3(details):
    details = re.sub(r' ,',' , ', str(details))
    return details

def clean_conditions(condition_list):
    if isinstance(condition_list, str):
        try:
            condition_list = ast.literal_eval(condition_list)
        except (SyntaxError, ValueError):
            return condition_list
    
    seen = set()
    cleaned_conditions = []

    for item in condition_list:
        code = item.replace("หรือ", "").replace("และ", "")
        if code not in seen:
            seen.add(code)
            if ("หรือ" in item or "และ" in item) or (code not in cleaned_conditions):
                cleaned_conditions.append(item)

    return cleaned_conditions

data["เงื่อนไขรายวิชา"] = data["เงื่อนไขรายวิชา"].apply(clean_conditions)
data['หมายเหตุ'] = data['หมายเหตุ'].apply(typess2)
data['รายละเอียดวิชา'] = data['รายละเอียดวิชา'].apply(thai)
data['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'] = data['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'].apply(typess)
cleaned_file_path = 'Data_2567_update.csv'
data.to_csv(cleaned_file_path, index=False,encoding="utf-8-sig")


