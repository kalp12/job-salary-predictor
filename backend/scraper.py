from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time 

def get_salary_selenium(job_title,driver):
    """Scrapes salary data using Selenium"""
    job_title = job_title.replace(" ", "-")
    url = f"https://www.indeed.com/salaries/{job_title}-Salary"
    print(url)
    driver.get(url)
    time.sleep(3)  # Allow time for the page to load

    try:
        # salary_element = driver.find_element(By.CLASS_NAME, "cmp-SalaryEstimate")
        # <div data-testid="avg-salary-value" class="css-1aa51kj eu4oa1w0">$105,682</div>
        # salary_element = driver.find("div", {"data-testid": "avg-salary-value"})
        salary_element = driver.find_element(By.XPATH, '//div[@data-testid="avg-salary-value"]')
        salary = salary_element.text.strip()
    except Exception:
        salary = "Salary data not found"

    driver.quit()
    return {"job_title": job_title.replace("-", " "), "estimated_salary": salary}

# Test
if __name__ == "__main__":
    chrome_driver_path = r"C:\Users\kp121\Documents\vs code project\chromedriver\chromedriver-win64\chromedriver.exe"

    service = Service(chrome_driver_path)

    driver = webdriver.Chrome(service=service)
    print(get_salary_selenium("Product Engineer",driver))
