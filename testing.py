import requests
from bs4 import BeautifulSoup
import re
url = 'https://www.mlb.com/mets/stats/'
response = requests.get(url)
lineup = ['Pete Alonso']
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    theTable = soup.find("table", class_="bui-table is-desktop-sKqjv9Sb") 
    [print(elem.text) for elem in theTable.find_all('tr')]
    
    pattern = r'(\d+)\s+([A-Za-z]+)\s+\1\s+([A-Za-z])\w*\s+([A-Za-z]+)\s+\3\w*'

    # Find all matches in the data
    matches = re.findall(pattern, "".join([elem.text for elem in theTable.find_all('tr')]))

    # Extract and print first and last names
    for match in matches:
        first_name, last_name = match
        print(f"First Name: {first_name}, Last Name: {last_name}")

#headers="tb-134-header-col17" scope="row">
