import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import ast
import time
from datetime import datetime
import re
import logging
import urllib.parse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize browser
firefox_options = Options()
driver = webdriver.Firefox(options=firefox_options)
wait = WebDriverWait(driver, 10)

all_data = pd.DataFrame(columns=[
                "ปี",
                "เทอม",
                "รหัสวิชา",
                "ชื่อไทย",
                "รหัสกลุ่ม",
                "ชื่อภาษาอังกฤษ",
                "ภาควิชา",
                "หน่วยกิต",
                "รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน",
                "รายละเอียดวิชา",
                "เงื่อนไขรายวิชา",
                "อาจารย์",
                "สอบกลางภาค",
                "สอบปลายภาค",
                "หมายเหตุ",
                'อ้างอิง'
                ]
)

def select_filters(year, semester):

    logger.info('Proceed with scraping!')
    logger.info(f"Filtering for year {year} semester {semester}")
    
    try:
        # Level 1: ID Status
        lv1_id_status = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(2) > td.normaldetail > select'))
        )
        lv1_id_status.click()
        lv1_available_id = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(2) > td.normaldetail > select > option:nth-child(1)'))
        )
        lv1_available_id.click()
        logger.info(f"Selected {lv1_available_id.text}")

        # Level 2: Faculty
        lv2_faculty = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(3) > td.normaldetail > select'))
        )
        lv2_faculty.click()
        lv2_science_option = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'option[value="007วิทยาศาสตร์"]'))
        )
        lv2_science_option.click()
        logger.info(f"Selected {lv2_science_option.text}")

        # Level 3: Max item
        lv3 = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(4) > td.normaldetail > select'))
        )
        lv3.click()
        lv3_maxitem = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'option[value="250"]'))
        )
        lv3_maxitem.click()
        logger.info(f"Selected {lv3_maxitem.text}")

        # Level 4.1: Year
        lv4_1 = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(5) > td:nth-child(2) > table > tbody > tr:nth-child(1) > td:nth-child(3) > select:nth-child(1)'))
        )
        lv4_1.click()
        lv4_1_year = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f'option[value="{year}"]'))
        )
        lv4_1_year.click()
        logger.info(f"Selected {lv4_1_year.text}")

        # Level 4.2: Semester
        lv4_2 = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(5) > td:nth-child(2) > table > tbody > tr:nth-child(1) > td:nth-child(3) > select:nth-child(2)'))
        )
        lv4_2.click()
        lv4_2_semester = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f'option[value="{semester}"]'))
        )
        lv4_2_semester.click()
        logger.info(f"Selected {lv4_2_semester.text}")

        # Level 4.3: Campus
        lv4_3 = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(5) > td:nth-child(2) > table > tbody > tr:nth-child(2) > td:nth-child(2) > select'))
        )
        lv4_3.click()
        lv4_3_campus = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(5) > td:nth-child(2) > table > tbody > tr:nth-child(2) > td:nth-child(2) > select > option:nth-child(3)'))
        )
        lv4_3_campus.click()
        logger.info(f"Selected {lv4_3_campus.text}")

        # Level 4.4: Level of Education
        lv4_4_lvl_of_ed = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(5) > td:nth-child(2) > table > tbody > tr:nth-child(3) > td:nth-child(2) > select'))
        )
        lv4_4_lvl_of_ed.click()
        lv4_4_bachelor = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#page > table:nth-child(4) > tbody > tr > td:nth-child(3) > table:nth-child(2) > tbody > tr:nth-child(5) > td:nth-child(2) > table > tbody > tr:nth-child(3) > td:nth-child(2) > select > option:nth-child(3)'))
        )
        lv4_4_bachelor.click()
        logger.info(f"Selected {lv4_4_bachelor.text}")

        search_button = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="submit"][value="ค้นหา"]'))
        )
        search_button.click()
        logger.info("Searching")

        return True

    except Exception as e:
        logger.error(f"Error during filter selection: {e}")
        return False

def split_groups_with_embedded_label(data):
    groups = []
    current_group = []
    counter = 1 
    for item in data:
        if 'อาจารย์:' in item[0] and current_group: 

            for sub_item in current_group:
                sub_item.insert(0, f'{counter:02}')
            groups.append(current_group) 
            current_group = [] 
            counter += 1 
        
        current_group.append(item)
    
    if current_group:
        for sub_item in current_group:
            sub_item.insert(0, f'{counter:02}')
        groups.append(current_group)
    
    return groups

def transform_group_data(data):
    grouped_data = {}
    for group_splitted in data:
        group_id = group_splitted[0][0] 
        details = {item[1]: item[2] for item in group_splitted}
        grouped_data[group_id] = details
    return grouped_data

def remove_eng(text):
    cleaned = re.sub(r'(หมายเหตุ.*|หมวด.*|CON:.*|CO:.*)', '', text)
    cleaned = re.sub(r'[A-Za-z0-9]+', '', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

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

def details(course_link, year, semester):
    global all_data
    try:
        link_web = f'https://reg.su.ac.th/registrar/{course_link}'
        driver.get(link_web)
        time.sleep(3)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        raw_data = []
        tr = soup.find_all('font',class_='normaldetail')
        for u in tr:
            text = u.get_text()
            raw_data.append(text)

        specify_data = [0, 1, 2, 4, 5, 6, 8]
        data_specified = [raw_data[i] for i in specify_data if i < len(raw_data)]

        schedule_data = []
        prof_name = []
        rows = soup.find_all('tr', class_='normalDetail')
        for row in rows:
            cells = row.find_all('td')
            data = [cell.get_text(strip=True) for cell in cells]
            if len(data) == 14:
                schedule_detail = [data[1],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13]]
                schedule_data.append(schedule_detail)
                cleaned_data = []
                current_group = None
                for row in schedule_data:
                    if row[0]:
                        current_group = row[0]
                    if current_group:
                        cleaned_data.append([current_group, row[1], row[2],row[3],row[4],row[5]])
            elif len(data) == 5:
                course_informations = [data[3],data[4]]
                prof_name.append(course_informations)

        descriptions = soup.find_all('td',class_='normalDetail')
        text_descriptions = [i.get_text() for i in descriptions]
        thai_text = [remove_eng(text) for text in text_descriptions]

        group_splitted = split_groups_with_embedded_label(prof_name)
        grouped_data = transform_group_data(group_splitted)

        grouped_schedule = {}
        for entry in cleaned_data:
            group = entry[0]
            detail = f"วัน : {entry[1]} _ เวลา : {entry[2]} _ ห้องเรียน : {entry[3]} _ อาคาร : {entry[4]} _ ประเภทการสอน : {entry[5]}"
            if group not in grouped_schedule:
                grouped_schedule[group] = []
            grouped_schedule[group].append(detail)

        subjects = []
        condition = soup.find('font', color="#800000", string="เงื่อนไขรายวิชา:")
        if condition:
            table = condition.find_next('table', class_='normalDetail')
            if table:
                for row in table.find_all('tr'):
                    tds = row.find_all('td')
                    for td in tds:
                        text = td.get_text(strip=True)
                        if text:
                            subjects.append(text)

                    links = row.find_all('a')
                    for a in links:
                        href = a.get_text(strip=True)
                        if href:
                            subjects.append(href)

        subjects = list(dict.fromkeys(subjects))
        print("Subject:", subjects)

        data_container = []
        for group, details in grouped_schedule.items():
            group_info = grouped_data.get(group, {})
            data_container.append({
                "ปี": year,
                "เทอม": semester,
                "รหัสวิชา": data_specified[0],
                "ชื่อภาษาอังกฤษ" : data_specified[1],
                "ชื่อไทย": data_specified[2],
                "รหัสกลุ่ม": group,
                "ภาควิชา":data_specified[3],
                "หน่วยกิต" : data_specified[5],
                "รายละเอียดวัน-เวลา-ห้องเรียน-อาคาร-ประเภทการสอน": f"เรียนทั้งหมด {len(details)} คาบดังนี้ " + ", ".join(details),
                "รายละเอียดวิชา":thai_text,
                "เงื่อนไขรายวิชา":subjects,
                "อาจารย์": group_info.get("อาจารย์:", ""),
                "สอบกลางภาค": group_info.get("สอบกลางภาค:", ""),
                "สอบปลายภาค": group_info.get("สอบปลายภาค:", ""),
                "หมายเหตุ": group_info.get("หมายเหตุ:", ""),
                'อ้างอิง':link_web
            })
        data = pd.DataFrame(data_container)
        all_data = pd.concat([all_data,data])

        return {}
    except Exception as e:
        return {}

def scrape_course_data(year, semester):
    global all_data
    try:
        base_url = "https://reg.su.ac.th/registrar/"
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        course = soup.find_all('tr', class_='normalDetail')
        data_list = []
        for i in range(len(course)):
            td_tags = course[i].find_all('td', string=lambda text: text and text.strip() == "W")
            if td_tags:
                # print(f"พบ W ในแถวที่ {i}:", [td.get_text(strip=True) for td in td_tags])

                course_info = course[i].find('a')
                if course_info:
                    course_link = course_info.get('href')
                    course_id = course_info.text

                    if course_id in all_data['รหัสวิชา'].unique():
                        print(f"Found Duplicated Course ID: {course_id}, Skipping...")
                        pass
                    else:
                        details(course_link, year, semester)
                        print(f'Course ID: {course_id}')
                        all_data.to_csv(f'Data_{year}.csv',index=False, encoding='utf-8-sig')
                    data_list.append({'Id': course_id, 'Link': course_link, 'Status': td_tags})

                    Data = pd.DataFrame(data_list)
                    Data.to_csv('Status.csv',index = False,encoding = 'utf-8-sig')

                else:
                    # print(f"Link มีสถานะเป็น C ในช่อง : {i}")
                    pass
        
            if course_id in all_data['รหัสวิชา'].unique():
                print(f"Found Duplicated Course ID: {course_id}. Skipping...")
                pass
            else:
                details(course_link, year, semester)
                print(f'Course ID: {course_id}')
                all_data.to_csv(f'Data_{year}.csv',index=False, encoding='utf-8-sig')

        next_page = soup.find('a', string='[หน้าต่อไป]')
        if next_page:
            link_tadpai = next_page.get('href')
            next_link = urllib.parse.urljoin(base_url, link_tadpai)
            driver.get(next_link)
            scrape_course_data(year, semester)
        else:
            print("The next page was not found. Skipping...")
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return []

def filter_item(main_url, year, semester):

    if select_filters(year, semester):
        retry_limit = 5
        for retry in range(retry_limit):
            time.sleep(10)
            if driver.current_url != main_url:
                logger.info(f'Forwarding to {driver.current_url}')
                scrape_course_data(year, semester)
                break
            
            elif retry == retry_limit - 1:
                logger.error('Max retries reached.')
                break
            
            else:
                logger.error(f'{retry}/{retry_limit} Access denied. Retrying...')

if __name__ == '__main__':
    url_registration = 'https://reg.su.ac.th/registrar/class_info.asp?avs924956177=1'

    current_year = [datetime.now().year + 543, datetime.now().year + 542]

    for year in current_year:
        for semester in range(1, 4):
            if len(all_data) == 0 and semester == 2:
                logger.error(f'Data not found on year {year}, Searching on previous year instead.')
                break

            driver.get(url_registration)
            selected_categories = filter_item(url_registration, year, semester)