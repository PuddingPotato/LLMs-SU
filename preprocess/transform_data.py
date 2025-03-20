import pandas as pd
import ast
import re

data = pd.read_csv(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\scicourses_2567.csv')

def thai_char(text):
    thai_text = re.sub(r'[^\u0E00-\u0E7F\s]', '', text)
    return thai_text.strip()

def clean_teaching_type(details):
    details = re.sub(r'ประเภทการสอน : C', 'ประเภทการสอน: Lecture(ทฤษฏี)', details)
    details = re.sub(r'ประเภทการสอน : T', 'ประเภทการสอน: Tue(ติวแนวข้อสอบ)', details)
    details = re.sub(r'ประเภทการสอน : L', 'ประเภทการสอน: Lab(ปฏิบัติ)', details)
    return details

def clean_subject_type(details):
    details = re.sub(r'\bบ\.', 'วิชาบังคับของสาขา', str(details))
    details = re.sub(r'\bล\.','วิชาเลือกของสาขา',str(details))
    details = re.sub(r'\bบล\.','วิชาบังคับเลือกของสาขา',str(details))
    return details 

def comma_format(details):
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

def dataframe_to_text(dataframe):

    transformed_text = []
    metadata = []
    course_sections = dataframe['รหัสวิชา'].value_counts().to_dict()
    for i in course_sections.keys(): # Course ID
        course_details = ""

        filtered_df = dataframe[dataframe['รหัสวิชา'] == i].copy()
        filtered_df.reset_index(drop = True, inplace = True)

        metadata.append(filtered_df['อ้างอิง'][0])

        course_info_map = f"""ปีการศึกษา: {filtered_df['ปี'][0]} ภาคเรียนที่ {filtered_df['เทอม'][0]}
ภาควิชา: {filtered_df['ภาควิชา'][0]}
ชื่อวิชา: {filtered_df['ชื่อไทย'][0]} หรือ {filtered_df['ชื่อภาษาอังกฤษ'][0]}
รหัสวิชา: {filtered_df['รหัสวิชา'][0]}
จำนวนหน่วยกิต: {filtered_df['หน่วยกิต'][0]}\n
เงื่อนไขที่ต้องผ่านก่อนขอสมัครเรียน: {f"นักศึกษาจำเป็นต้องสอบผ่านวิชาที่มีรหัสวิชา {filtered_df['เงื่อนไขรายวิชา'][0]} มาก่อน" if filtered_df['เงื่อนไขรายวิชา'][0] != "ไม่มีเงื่อนไขการลงทะเบียนเรียน" else "ไม่มีเงื่อนไขการลงทะเบียนเรียน"}
รายละเอียดการเรียนการสอน: {filtered_df['รายละเอียดวิชา'][0]}
จำนวนกลุ่ม: {course_sections[i]} กลุ่มดังนี้"""
        # print(course_info_map)

        course_details += course_info_map
        
        for j in range(course_sections[i]): # Group
            group_info = f"""กลุ่ม {j + 1}:
กลุ่มนักศึกษาที่เปิดรับ: {filtered_df['หมายเหตุ'][j]}
ชื่ออาจารย์ผู้สอน: {filtered_df['อาจารย์'][j]}
วันที่นัดสอบกลางภาค: {filtered_df['สอบกลางภาค'][j]}
วันที่นัดสอบกปลายภาค: {filtered_df['สอบปลายภาค'][j]}
มีตารางเรียน/คาบเรียนดังนี้:"""
            # print(group_info)

            course_details += group_info

            for sch in filtered_df['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'][j]: # Schedule
                schedule_items = sch.split(' _ ')

                schedule_map = f"""
-{schedule_items[0].strip()},
 {schedule_items[1].strip()}น.,
 {schedule_items[2].strip()},
 {schedule_items[3].strip()},
 {schedule_items[4].strip()}\n"""
                # print(schedule_map)

                course_details += schedule_map
                

        transformed_text.append(course_details)
    
    return transformed_text, metadata

day_order = {
    "อาทิตย์": 0,
    "จันทร์": 1,
    "อังคาร": 2,
    "พุธ": 3,
    "พฤหัสบดี": 4,
    "ศุกร์": 5,
    "เสาร์": 6
}

data['หมายเหตุ'] = data['หมายเหตุ'].apply(clean_subject_type)

data['รายละเอียดวิชา'] = data['รายละเอียดวิชา'].apply(thai_char)

data['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'] = data['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'].apply(clean_teaching_type)
data['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'] = data['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'].apply(lambda val: val.split('คาบดังนี้')[1])
data['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'] = data['รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน'].apply(
    lambda i: i[1:]
    ).apply(lambda u: u.replace('ประเภทการสอน:', 'ประเภทการสอน :')).apply(
            lambda x: x.split(', ')
            ).apply(lambda y: sorted(y, key = lambda z: day_order[z.split(' _ ')[0].split(' : ')[1].strip()]))

data["เงื่อนไขรายวิชา"] = data["เงื่อนไขรายวิชา"].apply(clean_conditions)


data['หน่วยกิต'] = data['หน่วยกิต'].apply(
    lambda x: x.split('(')[0]
)

data['สอบกลางภาค'] = data['สอบกลางภาค'].fillna("ไม่ได้ระบุวันสอบกลางภาค")
data['สอบปลายภาค'] = data['สอบปลายภาค'].fillna("ไม่ได้ระบุวันสอบปลายภาค")
data['เงื่อนไขรายวิชา'] = data['เงื่อนไขรายวิชา'].apply(
    lambda x: "ไม่มีเงื่อนไขการลงทะเบียนเรียน" if len(x) == 0 else x)
data['อาจารย์'] = data['อาจารย์'].fillna("ไม่ระบุชื่อผู้สอน")
data['หมายเหตุ'] = data['หมายเหตุ'].fillna("ไม่ระบุหมายเหตุของวิชา")

transformed_data = pd.DataFrame({'content': dataframe_to_text(data)[0],
                                 'metadata': dataframe_to_text(data)[1]})

transformed_data.to_csv(r'C:\Users\User\Desktop\Project LLMs\RAG-LangChain\data\transformed_data_2567.csv', index = False)