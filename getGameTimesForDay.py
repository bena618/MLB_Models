import requests
from bs4 import BeautifulSoup
from datetime import datetime

#For Github
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}

url = 'https://www.rotowire.com/baseball/daily-lineups.php'
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    times = soup.find_all('div', class_="lineup__time")
    times = [elem.text.strip() for elem in times if elem.text.strip()]

    if times:
        first_time = times[0]
        last_time = times[-1]
        
        with open('game_times.txt', 'w') as f:
            f.write(f"{first_time},{last_time}")
    else:
        print("No game times found.")
