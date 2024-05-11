import requests
from bs4 import BeautifulSoup
from datetime import datetime
import yaml

#For Github
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}

url = 'https://www.rotowire.com/baseball/daily-lineups.php'
response = requests.get(url,headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    game_times = soup.find_all('div',class_="lineup__time")[:-2]
    game_times = [" ".join(elem.text.split()[:-1]) for elem in game_times]
    game_times = [datetime.strptime(time, '%I:%M %p').strftime('%H:%M') for time in game_times]
    cron_expression = f"    - cron: '0 {min(int(game_times[0].split(':')[0])+4 -2,23)}-{max(int(game_times[-1].split(':')[0])+4,23)} * * *'"


    fname = '.github/workflows/withRotoIdea.yml'
#    fname = 'withRotoIdea.yml'
    with open(fname, 'r') as file:
        print(file)
        data = [line for line in file]
    
    # Update the 10th line
    data[9] = cron_expression
    [print(elem) for elem in enumerate(data)]
    print("DFGVHBJNHGJFHJ")
    # Write the updated content back to the file
    with open(fname, 'w') as file:
        yaml.dump(data, file)
