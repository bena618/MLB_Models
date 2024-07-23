# %%
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}

# %%
#Suppressing Warnings for better looking output if desired
import warnings
warnings.filterwarnings('ignore')

# %%
todaysDate = '7/23/2024'
yesterdaysDate = '7/22/2024'

# %%
from datetime import datetime, timedelta

# Function to get the date 7 days ago
def date_N_days_ago_str(date_str, n):
    # Parse the input date string to a datetime object
    date = datetime.strptime(date_str, '%m/%d/%Y')
 
    date_N_days_ago = date - timedelta(days=n)
    
    # Format the date back to 'MM/DD/YYYY'
    date_N_days_ago_str = date_N_days_ago.strftime('%m/%d/%Y')
    return date_N_days_ago_str

# %% [markdown]
# Data Grabbing Function

# %%
# Function to get pitcher data and return mean ERA, WHIP, and K/9
def get_pitcher_data(name):
    '''
    if name == 'DJ Herz':
        return {"Name": name,"whip": 1.044}
    elif name == 'Bowden Francis': 
        return {"Name": name,"whip": 1.7}
    elif name == 'Adam Mazur':
        return {"Name": name,"whip": 1.75}#made up cause dont see nay log last 3 years
    elif name == 'Valente Bellozo':
        return {"Name": name,"whip": 1.2}
    elif name == 'Gavin Williams':
        return {"Name": name,"whip": 1.5}
    elif name == 'Davis Daniel':
        return {"Name": name,"whip": 1.15}
    elif name == 'Cristian Mena':
        return {"Name": name,"whip": 1.8}
#    elif name == 'Keider Montero':
#        return {"Name": name,"whip": 1.9}
'''        
    
    url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={ids[name]}&stats=pitching"
    print(f"{name}:{url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        stats = response.json()
        print(url)
        try:
            total_pc_last_30_compare = int(stats['gamelog']['majors']['pitching']['footer'][1]['pc']['text'])
            whip_L30 = float(stats['gamelog']['majors']['pitching']['footer'][1]['whip']['text'])

            if total_pc_last_30_compare < 200:
                return {"Name": name,"whip": 1.313}
            else:
                return {"Name": name,"whip": whip_L30}
            
#            daily = stats['gamelog']['majors']['pitching']['body']
        except:
#            print(total_pc_last_30_compare)
            url = 'https://www.statmuse.com/mlb/ask/' + name.lower().replace(' ', '%20') + '%20stats%20last%2010%20games%20including%20whip'
            response = requests.get(url)
            try:
                tables = pd.read_html(response.text)
#                print(tables)
                df = tables[0].head(1)
                df = df.filter(items=["NAME","WHIP"])
                return {"Name": name,"whip": mean_values["WHIP"]}
            except:
                print(f"help-p: {url}")
                print(response.text)
                return {"Name": name,"whip": 1.313}
#                return None
'''            

        game = 0
        games = []
        while total_pc_last_30_compare > 0:
            cur_game = daily[game]
#            er = int(cur_game['er'])
            ip = float(cur_game['ip'])
            ip = int(ip) + (ip-int(ip)) * (10/3)
#            ks_per_9 = int(cur_game['k']) * 9 / ip
#            era = er / ip * 9 / ip
            whip = float(cur_game['whip'])
            pc = int(cur_game['pc'])
#            games.append([era, whip, ks_per_9])
            games.append([whip])
            total_pc_last_30_compare -= pc
            game += 1
#        df = pd.DataFrame(games, columns=['ERA', 'WHIP', 'K/9'])
        df = pd.DataFrame(games, columns=['WHIP'])
#        df = df.fillna({"ERA": df.loc[:, 'ERA'].mean(), "WHIP": df.loc[:, 'WHIP'].mean(), "K/9": df.loc[:, 'K/9'].mean()})
        df = df.fillna({"WHIP": df.loc[:, 'WHIP'].mean()})
        
        mean_values = df.mean()
#        return {"era": mean_values["ERA"], "whip": mean_values["WHIP"], "k9": mean_values["K/9"]}
        return {"Name": name,"whip": mean_values["WHIP"]}
    else:
        print(f"{name}: None - default 1.313 used")
        return {"Name": name,"whip": 1.313}
#        return None
'''

# %%
#url = "https://www.fangraphs.com/leaders/splits-leaderboards?splitArr=&splitArrPitch=&autoPt=false&splitTeams=false&statType=player&statgroup=2&startDate=2024-6-14&endDate=2024-7-13&players=&filter=&groupBy=season&wxTemperature=&wxPressure=&wxAirDensity=&wxElevation=&wxWindSpeed=&position=P&sort=21,1&pageitems=2000000000&pg=0"
#response = requests.get(url,headers=headers)
#soup = BeautifulSoup(response.text, "html.parser")

#print(soup)

# %%
# Function to get batter data and return mean OBP and SLG
#def get_batter_data(name, date):
def get_batter_data(name):
    
    if name == 'Luis Garcia':
        name += ' jr'
#    elif name == 'Brian Anderson':#batting 0.00, about 2 at bats a week
#        return {"Name": name, "avg": .050, "single_prob": 1.0, "double_prob": 0.0, "triple_prob": 0.0, "homerun_prob": 0.0}
   
    elif name == 'Will Smith':
        name += ' Dodgers'
        print(name)
#    elif name == 'Greg Jones':#batting 0.00, 5 pas this season
#        return {"Name": name, "avg": .050, "single_prob": 1.0, "double_prob": 0.0, "triple_prob": 0.0, "homerun_prob": 0.0}
#    elif name == 'Tyler Wade':
#        return {"Name": name, "avg": .250, "single_prob": 1.0, "double_prob": 0.0, "triple_prob": 0.0, "homerun_prob": 0.0}
    elif name == 'Enrique Hernandez':
        name = 'Kike Hernandez'
    elif name == 'Dairon Blanco':
        return {"Name": name, "avg": .050, "single_prob": 1.0, "double_prob": 0.0, "triple_prob": 0.0, "homerun_prob": 0.0}

    


#    url = 'https://www.statmuse.com/mlb/ask/' + name.lower().replace(' ', '%20') + '%20last%207%20days%20stats%20including%20obp%20avg%20and%20slg'
    url = 'https://www.statmuse.com/mlb/ask/' + name.lower().replace(' ', '%20') + '%20stats%20between%20' + date_N_days_ago_str(todaysDate,7) + '%20and%20' + yesterdaysDate + '%20stats%20including%20obp%20avg%20and%20slg'
    print(url)
#    print(f"{name}:{url}")

    response = requests.get(url)
    
    if response.status_code == 200:
        try:
            tables = pd.read_html(response.text)
        except:
            url = 'https://www.statmuse.com/mlb/ask/' + name.lower().replace(' ', '%20') + '%20stats%20including%20obp%20avg%20and%20slg'
            response = requests.get(url)
            try:
                tables = pd.read_html(response.text)
            except:
                print(f"help: {url}")
#                df = pd.DataFrame([[f'{name} no stats found so 0.0','0.0']],columns=["Name","avg"])                
#                return df
                return None

        df = tables[0].head(25)
#        if name == 'Will Smith':
#            print("Hi")
#            df = df.iloc[[0]]
#            print(df)
        
        if df['G'].iloc[0] < 3:
            url = 'https://www.statmuse.com/mlb/ask?q=' + name.lower().replace(' ', '%20') + '%20last%2010%20games%20%20obp%20avg%20and%20slg'
            print(url)
            response = requests.get(url)
            if response.status_code == 200:
                try:
                    print(url)
                    tables = pd.read_html(response.text)
                    df = tables[0].head(10)

                    df = df.filter(items=["NAME","AVG","H","2B","3B","HR"])
                    total_hits = df["H"].iloc[0]
                    if total_hits == 0:
                        if np.isnan(df["AVG"].iloc[0]):
                            return {"Name": name, "avg": .240, "single_prob": 1.0, "double_prob": 0.0, "triple_prob": 0.0, "homerun_prob": 0.0}
                        else:
                            return {"Name": name, "avg": df["AVG"].iloc[0], "single_prob": 1.0, "double_prob": 0.0, "triple_prob": 0.0, "homerun_prob": 0.0}
                    double_prob = df["2B"].iloc[0] / total_hits
                    triple_prob = df["3B"].iloc[0] / total_hits
                    homerun_prob = df["HR"].iloc[0] / total_hits
                    single_prob = 1 - (double_prob +  triple_prob + homerun_prob)

                    return {
                        "Name": df['NAME'].iloc[0],
                        "avg": df["AVG"].iloc[0],
                        "single_prob": single_prob,
                        "double_prob": double_prob,
                        "triple_prob": triple_prob,
                        "homerun_prob": homerun_prob
                    }

                except:
                    print(f"name:{name}, {df["AVG"]},{df}")
            else:
                print(f"{name}: < 3 none")
                return None
        else:
            df = df.filter(items=["NAME","AVG","H","2B","3B","HR"])
            total_hits = df["H"].iloc[0]
            if total_hits == 0:
                return {"Name": name, "avg": df["AVG"].iloc[0], "single_prob": 1.0, "double_prob": 0.0, "triple_prob": 0.0, "homerun_prob": 0.0}

            double_prob = df["2B"].iloc[0] / total_hits
            triple_prob = df["3B"].iloc[0] / total_hits
            homerun_prob = df["HR"].iloc[0] / total_hits
            if homerun_prob > .3 and total_hits < 15:
                removed_for_balance = homerun_prob - .3 #Trying to make it so no one is purely expected to hit a homerun
                homerun_prob = .3
                double_prob = .25 * removed_for_balance#75% rest goes to single_prob by default cause of the 1-everything else


            single_prob = 1 - (double_prob +  triple_prob + homerun_prob)

            return {
                "Name": df['NAME'].iloc[0],
                "avg": df["AVG"].iloc[0],
                "single_prob": single_prob,
                "double_prob": double_prob,
                "triple_prob": triple_prob,
                "homerun_prob": homerun_prob
            }


#            return {"Name": df['NAME'],"avg": df["AVG"], "slg": df["SLG"]}
#            return {"Name": df['NAME'],"avg": df["AVG"]}
    else:
        print(f"{name} none returned")
        print(url)
        return None

# %%
#%history


# %%
#print(get_batter_data('Byron Buxton'))

# %%

url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=1st-inning&subcategory=1st-inning-total-runs"
response = requests.get(url,headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
teamsAndLines = soup.find_all("div", class_="sportsbook-event-accordion__wrapper expanded")

awayTeams = [elem.text.split()[0] for elem in teamsAndLines]

teamsAndLines = [elem.text for elem in teamsAndLines]
odds_dict_nrfi = {}
odds = []
#YRFI,NRFI pattern
[odds.extend(line.split("0.5")[1:3]) for line in teamsAndLines]
odds = [elem[:4] for elem in odds]

#print(odds)

print(teamsAndLines)

#awayTeams[-2] = "LAA"
#awayTeams[2] = "NYY"
#awayTeams[1] = "WSH"
#awayTeams[4] = "CHC"
#awayTeams[-3] = "CWS"

odds_dict_nrfi = {awayTeams[i]: odds[2 * i: 2 * i + 2] for i in range(len(awayTeams))}
print(odds_dict_nrfi)

#indexForOdds = [index for index,elem in enumerate(awayTeams) if elem.startswith(awayTeam.split()[-1])]

# %%

'''
from selenium import webdriver
import json

pitcher_so_lines = {}

url = "https://sportsbook.fanduel.com/baseball"

# Set up the Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')

# Initialize the Chrome driver
driver = webdriver.Chrome(options=chrome_options) 

# Fetch the webpage
#driver.get(url)

# Get the HTML content
html = driver.page_source

# Close the driver
driver.quit()
'''

# %%
#print(html)

# %%
#print(str(html).find("toronto-blue-jays-@-arizona-diamondbacks-33411239"))

# %%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# Set up the WebDriver (ensure chromedriver is in your PATH)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')  # Run in headless mode
chrome_options.add_argument('--no-sandbox')  # Add no-sandbox argument

# Initialize the Chrome driver
driver = webdriver.Chrome(options=chrome_options)

# URL of the FanGraphs splits leaderboard page
url = "https://www.fangraphs.com/leaders/splits-leaderboards?splitArr=&splitArrPitch=&autoPt=false&splitTeams=false&statType=player&statgroup=2&startDate=2024-6-12&endDate=2024-7-11&players=&filter=&groupBy=season&wxTemperature=&wxPressure=&wxAirDensity=&wxElevation=&wxWindSpeed=&position=P&sort=21,1&pageitems=2000000000&pg=0"

# Load the webpage
print("Loading the webpage...")
driver.get(url)

try:
    # Wait for the table to be present (increase timeout to 30 seconds)
    print("Waiting for the table to be present...")
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div#LeaderBoard1_dg1 table.rgMasterTable')))

    # Extract table data
    print("Extracting table data...")
    table = driver.find_element(By.CSS_SELECTOR, 'div#LeaderBoard1_dg1 table.rgMasterTable')
    headers = [header.text for header in table.find_elements(By.TAG_NAME, 'th')]

    rows = table.find_elements(By.TAG_NAME, 'tr')[1:]  # Skip header row
    data = []
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, 'td')
        data.append([cell.text for cell in cells])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Close the WebDriver
    driver.quit()

    # Display the first few rows of the dataframe
    print(df.head())

    # Optionally, save the data to a CSV file
    df.to_csv('fangraphs_splits_leaderboard.csv', index=False)

    print("Script completed successfully.")
except Exception as e:
    print("An error occurred:", e)
    # Print page source for debugging
    print(driver.page_source)
    driver.quit()


# %%
url = 'https://www.rotowire.com/baseball/daily-lineups.php'
#url = 'https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow'
response = requests.get(url,headers=headers)

pitchers = []
batters = [] 
teams = []
game_times = []
ids = {}

if response.status_code == 200:
   soup = BeautifulSoup(response.text, 'html.parser')
   pitchers = soup.find_all('div',class_='lineup__player-highlight-name')
   ids = {a.text.strip()[:-2]: a.find('a').get('href').split('-')[-1] for a in pitchers}
   pitchers = [elem.find('a').text for elem in pitchers]
 #   pitchers[5] = 'Carlos Carrasco'       
 #   ids['Carlos Carrasco'] = 9449s
   #I think 1st inning and F5 different but use both

   pitchers = [get_pitcher_data(elem) for elem in pitchers]

   pitchers[0] = {"Name": 'Joey Wentz',"whip": 2} 
   pitchers[1] = {"Name": 'Xzavion Curry',"whip": .75}   


   pitchers_for_1st = pitchers
  
   ids['Shawn Armstrong'] = 13661
   '''
   ids['Brent Honeywell'] = 13442
   ids['Beau Brieske'] = 17745
   ids['Jared Shuster'] = 16728
   ids['Carson Fulmer'] = 13807


   
  '''
   pitchers_for_1st[10] = get_pitcher_data('Shawn Armstrong')   
   '''
   pitchers_for_1st[6] = get_pitcher_data('Brent Honeywell')   
   pitchers_for_1st[7] = get_pitcher_data('Beau Brieske')   
   pitchers_for_1st[15] = get_pitcher_data('Jared Shuster')   
   pitchers_for_1st[25] = get_pitcher_data('Carson Fulmer')   
   '''

   
   batters = soup.find_all('li',class_ = 'lineup__player')
   batters = [elem.find('a').get('title') for elem in batters]
   batters = [get_batter_data(elem) for elem in batters]

   teams = soup.find_all('div',class_= 'lineup__abbr')
   teams = [elem.text for elem in teams]

   game_times = soup.find_all('div',class_="lineup__time")
   game_times = [elem.text for elem in game_times][:-2]

   confirmedOrExpected = soup.find_all('li',class_="lineup__status")
   confirmedOrExpected = [elem.text.strip().split()[0][0] for elem in confirmedOrExpected]
   
  


# %%
#game_times[7:] = game_times[9:]
#teams[14:] = teams[18:]

#pitchers[1] = get_pitcher_data('Keider Montero')
#%history

#print(get_pitcher_data("Carlos Rodon"))

# %%
[print(elem) for elem in pitchers_for_1st]


# %%
names_For_first = [elem['Name'] for elem in pitchers_for_1st]
names = [
    "Pablo Lopez",
    "Carlos Rodon",
    "C. Sanchez",
    "Martin Perez",
    "Roddery Muñoz",
    "Ranger Suarez",
    "S. Schwellenbach",
    "Randy Vasquez",
    "Yariel Rodriguez",
    "G. Rodriguez",
    "Jose Soriano",
    "Jose Berrios"
    "S. Arrighetti", 
    "Reynaldo Lopez",
    "S. Woods Richardson",
    "J. Montgomery"

]

# Replacement mappings
replacements = {
    "Pablo Lopez": "Pablo LÃ³pez",
    "Carlos Rodon": "Carlos RodÃ³n",
    "C. Sanchez": "Cristopher SÃ¡nchez",
    "Martin Perez": "MartÃ­n PÃ©rez",
    "Roddery Muñoz": "Roddery MuÃ±oz",
    "Ranger Suarez": "Ranger SuÃ¡rez",
    "S. Schwellenbach" : "Spencer Schwellenbach",
    "Randy Vasquez" : "Randy VÃ¡squez",
    "Yariel Rodriguez":"Yariel RodrÃ­guez",
    "G. Rodriguez":"Grayson Rodriguez",
    "Jose Soriano":"JosÃ© Soriano",
    "Jose Berrios" : "JosÃ© BerrÃ­os",
    "S. Arrighetti": "Spencer Arrighetti",
    "Reynaldo Lopez": "Reynaldo LÃ³pez",
    "S. Woods Richardson":"Simeon Woods Richardson",
    "J. Montgomery":"Jordan Montgomery"

}

# Replace names
names_For_first = [replacements.get(name, name) for name in names_For_first]
#names_For_first = [name.replace(name.replace("Carlos Rodon","Carlos RodÃ³n"),LÃ³pez) for name in names_For_first] 
#names_For_first = [name.replace("Pablo Lopez","Carlos RodÃ³n") for name in names_For_first] 

print(names_For_first)
#[print(elem is None) for elem in batters]
batters_for_first = [elem['avg'] for elem in batters if elem is not None]
batters_avg = [elem['avg'] for elem in batters if elem is not None]

print(batters_for_first)
first_4_greater_than_300  = [sum(x >= 0.300 for x in batters_for_first[i:i+4]) for i in range(0, len(batters_for_first), 9)]
first_4_greater_than_300 = [first_4_greater_than_300[i+1] if i % 2 == 0 else first_4_greater_than_300[i-1] for i in range(len(first_4_greater_than_300))]

all_greater_than_300 = [sum(x >= 0.300 for x in batters_avg[i:i+9]) for i in range(0, len(batters_avg), 9)]
all_greater_than_300 = [all_greater_than_300[i+1] if i % 2 == 0 else all_greater_than_300[i-1] for i in range(len(all_greater_than_300))]


#print(first_4_greater_than_300)

[print(batters_for_first[i:i+4]) for i in range(0, len(batters_for_first), 9)]


# %%
def implied_odds(odds):
    # Convert probability to decimal odds
    decimal_odds = 1 / odds

    # Convert decimal odds to American odds
    if decimal_odds >= 2:
        american_odds = (decimal_odds - 1) * 100
    else:
        american_odds = -100 / (decimal_odds - 1)

    # Format the result
    res = int(american_odds)
    if res > 0:
        return f"+{res}"
    else:
        return str(res)

# Example usage:
odds = 0.44    # Decimal odds representing a probability of 0.44
print(implied_odds(odds))  # Output: +127

odds = 0.475    # Decimal odds representing a probability of 0.5
print(implied_odds(odds))  # Output: +100

odds = 0.25  # Decimal odds representing a probability of 0.25
print(implied_odds(odds))  # Output: +300

odds = 0.33  
print(implied_odds(odds))  # Output: +203

odds = 0.75  # Decimal odds representing a probability of 0.75
print(implied_odds(odds))  # Output: -300


# %%
from bs4 import BeautifulSoup
import re

url = 'https://www.baseball-reference.com/leagues/majors/2024-standard-pitching.shtml'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
}
# Send a GET request to the URL
response = requests.get(url, headers=headers)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Convert response content to string
    soupAsStr = response.text

    # Use regular expression to find player names and href attributes
    pattern = r'<a href="([^"]+)">([^<]+)<\/a>'
    links = {}

    # Find all matches in the string
    matches = re.findall(pattern, soupAsStr)

    # Extract href and sanitize player name from each match and store in dictionary
    for href, player_name in matches:
        # Replace &nbsp; with a regular space
        player_name = player_name.replace('&nbsp;', ' ')
        # Sanitize player name: Remove extra whitespace and newline characters
        sanitized_name = ' '.join(player_name.strip().split())
        links[sanitized_name] = href

#    print(len(links))
#    [print(asdasd) for asdasd in links]
    stats = []
    for i,elem in enumerate(names_For_first):
#        print(f"{elem}, {elem in links}")
        if elem in links:
#            print(links[elem])
            url = f'https://www.baseball-reference.com/players/split.fcgi?id={links[elem].split('/')[-1].split('.')[0]}&year=2024&t=p#all_innng'
#            print(url)
            response = requests.get(url, headers=headers)
            soupAsStr = response.text
            start = soupAsStr.find('1st inning')
            stats_for_player = BeautifulSoup(soupAsStr[start:start +1448], 'html.parser')
            try:

                ip = int(float(stats_for_player.find('td', {'data-stat': 'IP'}).text.strip()))
                h = int(stats_for_player.find('td', {'data-stat': 'H'}).text.strip())   
                bb = int(stats_for_player.find('td', {'data-stat': 'BB'}).text.strip()) 
                ibb = int(stats_for_player.find('td', {'data-stat': 'IBB'}).text.strip())
                era = float(stats_for_player.find('td', {'data-stat': 'earned_run_avg'}).text.strip())
                whip = (bb + h + ibb) / ip
            except:
                print(stats_for_player)

            stats.append((elem,teams[i],game_times[i//2],ip,round(whip,2),era,first_4_greater_than_300[i]))
        else:
            stats.append((elem,teams[i],game_times[i//2],'0','N/A','N/A',first_4_greater_than_300[i]))

else:
    print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
    if 'Retry-After' in response.headers:
        retry_after = int(response.headers['Retry-After'])
        print(f"Retry-After header value: {retry_after} seconds")
        time.sleep(retry_after)  # Wait for the specified time before retrying
    else:
        print("Retry-After header not found. Implementing exponential backoff.")

formatted_data = [
{'Pitcher': item[0],'Team': item[1],'Time':item[2],'Games': item[3], '1st inning Whip': item[4], '1st inning ERA': item[5], "OPP F4 >= .300": item[6]}
for item in stats]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'Pitchers 1st inning Stats {todaysDate}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.show()
print(datetime.now())

# %%
[print(elem) for elem in links]

# %%
f5_cheat_sheet = []
print(pitchers[0])
for i,elem in enumerate(pitchers):
    f5_cheat_sheet.append((elem['Name'],teams[i],game_times[i//2],round(elem['whip'],2),all_greater_than_300[i]))

formatted_data = [
{'Pitcher': item[0],'Team': item[1],'Time':item[2], 'Whip': item[3], "OPP >= .300": item[4]}
for item in f5_cheat_sheet]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'Pitchers 1st inning Stats {todaysDate}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/f5_cheat_sheet.png', bbox_inches='tight', dpi=300)
plt.show()


# %%
formatted_data = [
{'Pitcher': item[0],'Team': item[1],'Time':item[2],'Games': item[3], '1st inning Whip': item[4], '1st inning ERA': item[5], "OPP F4 >= .300": item[6]}
for item in stats]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'Pitchers 1st inning Stats {todaysDate}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/NRFI_YRFI_CheatSheet.png', bbox_inches='tight', dpi=300)
plt.show()



# %%
#pitchers[5] =

# %%
#game_times[0:] = game_times[1:]
#confirmedOrExpected[0:] = confirmedOrExpected[2:]
#teams[0:] = teams[2:]

print(len(game_times))

print(len(pitchers))
#batters[0:] = batters[18:] 
print(len(pitchers))
#game_times[6:] = game_times[7:]
#teams[12:] = teams[14:]
#pitchers[12:] = pitchers[14:]
#batters[12 * 9:] = batters[14 * 9:]
print(pitchers)
print(list(elem['Name'] for elem in pitchers))
print(game_times)

# %%
# Function to simulate an at-bat
def simulate_at_bat(batter_avg,single_prob,double_prob,triple_prob,hr_prob, pitcher_whip):
    hit_prob = max(batter_avg + (pitcher_whip-1.32) * .1,0)
    walk_prob = max((pitcher_whip - hit_prob)* .1 * batter_avg/.300,0)#hotter batters more likely to be walked
#    strikeout_prob = pitcher_k9 / 27
    out_prob = 1 - (hit_prob + walk_prob)
    if out_prob < 0:
        walk_prob = min((pitcher_whip - hit_prob)* .1 * batter_avg/.350,0.3)
        out_prob = 1 - (hit_prob + walk_prob)
        print("adj1")

    if out_prob < 0:
        hit_prob = .7
        out_prob = 1 - (hit_prob + walk_prob)
        print("adj2")
    if out_prob < 0:
        walk_prob = .25
        hit_prob = .5
        out_prob = .25
        print("force")         

    outcome = np.random.choice(['hit', 'walk', 'out'], p=[hit_prob, walk_prob, out_prob])

    if outcome == 'hit':
        outcome = np.random.choice(['single', 'double', 'triple','home_run'], p=[single_prob, double_prob, triple_prob,hr_prob])
    return outcome

# %%
def simulate_inning(batter_stats, pitcher_stats,batter_index=0):
    runs_scored = 0
    outs = 0
    bases = [False, False, False]
#    batter_index = 0
    hits = 0
    while outs < 3:
        try:
#            print(batter_stats[batter_index]['avg'])
            batter_avg = batter_stats[batter_index]['avg']
            batter_1b_prob = batter_stats[batter_index]['single_prob']
            batter_2b_prob = batter_stats[batter_index]['double_prob']
            batter_3b_prob = batter_stats[batter_index]['triple_prob']
            batter_HR_prob = batter_stats[batter_index]['homerun_prob']

            #pitcher_k9 = pitcher_stats['k9']
            pitcher_whip = pitcher_stats['whip']
            
 #           outcome = simulate_at_bat(batter_avg, batter_slg, pitcher_k9, pitcher_era, pitcher_whip)
            outcome = simulate_at_bat(batter_avg,batter_1b_prob,batter_2b_prob,batter_3b_prob,batter_HR_prob, pitcher_whip)
        except Exception as e:
            hp = max(min(.700, batter_avg + (pitcher_whip-1.32) * .1),0)
            wp = max((pitcher_whip - hp)* .1 * batter_avg/.300,0)

            hp = max(batter_avg + (pitcher_whip-1.32) * .1,0)
            wp = max((pitcher_whip - hp)* .1 * batter_avg/.300,0)#hotter batters more likely to be walked
#    strikeout_prob = pitcher_k9 / 27
            print(f"Exception: {batter_stats[batter_index]}")
            print(hp)
            print(wp)
            print(1 - (hp + wp))
            print(e)
            raise SyntaxError
#            print(batter_stats[batter_index]['Name'])
#            print(pd.DataFrame(batter_stats[batter_index]))
#            print("Skipped")
#            continue
#            raise SyntaxError


#        print(batter_stats[batter_index],outcome,batter_index)
        if outcome == 'single':
            if bases[2]: runs_scored += 1
            bases = [True] + bases[:2]
            hits += 1
        elif outcome == 'double':
            runs_scored += bases[2] + bases[1]
            bases = [False, True, True]
            hits += 1
        elif outcome == 'triple':
            runs_scored += bases[2] + bases[1] + bases[0]
            bases = [False, False, True]
            hits += 1
        elif outcome == 'home_run':
            name = batter_stats[batter_index]['Name']
#            print(f"name: {name}")
            if name in home_run_cur:
                home_run_cur[name] += 1
            else:
                home_run_cur[name] = 1
                hrs_cur.append(name)

            runs_scored += 1 + bases[2] + bases[1] + bases[0]
            bases = [False, False, False]
            hits += 1
        elif outcome == 'walk':
            #bases loaded
            if bases == [True, True, True]: 
                runs_scored += 1
            #if 1st empty, batter goes there everything else stays the same
            elif bases[0] == False:
                bases =  [True] + bases[1:]
            #if 1st has someone on it but 2nd does not
            elif bases[1] == False:
                bases =  [True, True] + [bases[2]]
            #if 1st has someone on it(checked 2 before) and 2nd has somoeone on it(checked 1 before), and not bases loaded(checked 1st)
            #ie [True,True,False]
            else: 
                bases = [True,True,True]
        else:
            outs += 1    
        
        batter_index = (batter_index + 1) % len(batter_stats)
    return runs_scored,hits,batter_index

# %%
'''
def simulate_inning(batter_stats, pitcher_stats):
    runs_scored = 0
    outs = 0
    bases = [False, False, False]
    batter_index = 0
    while outs < 3:
        try:
#            print(batter_stats[batter_index]['avg'])
            batter_avg = batter_stats[batter_index]['avg']
            batter_1b_prob = batter_stats[batter_index]['single_prob']
            batter_2b_prob = batter_stats[batter_index]['double_prob']
            batter_3b_prob = batter_stats[batter_index]['triple_prob']
            batter_HR_prob = batter_stats[batter_index]['homerun_prob']

            #pitcher_k9 = pitcher_stats['k9']
            pitcher_whip = pitcher_stats['whip']
            
 #           outcome = simulate_at_bat(batter_avg, batter_slg, pitcher_k9, pitcher_era, pitcher_whip)
            outcome = simulate_at_bat(batter_avg,batter_1b_prob,batter_2b_prob,batter_3b_prob,batter_HR_prob, pitcher_whip)
        except:
            hp = max(min(.700, batter_avg + (pitcher_whip-1.32) * .1),0)
            wp = max((pitcher_whip - hp)* .1 * batter_avg/.300,0)
            print(f"Exception: {batter_stats[batter_index]}")
            print(hp)
            print(wp)
            print(1 - (hp + wp))
            raise SyntaxError
#            print(batter_stats[batter_index]['Name'])
#            print(pd.DataFrame(batter_stats[batter_index]))
#            print("Skipped")
            continue
#            raise SyntaxError


#        print(batter_stats[batter_index],outcome)
        if outcome == 'single':
            if bases[2]: runs_scored += 1
            bases = [True] + bases[:2]
        elif outcome == 'double':
            runs_scored += bases[2] + bases[1]
            bases = [False, True, True]
        elif outcome == 'triple':
            runs_scored += bases[2] + bases[1] + bases[0]
            bases = [False, False, True]
        elif outcome == 'home_run':
            runs_scored += 1 + bases[2] + bases[1] + bases[0]
            bases = [False, False, False]
        elif outcome == 'walk':
            #bases loaded
            if bases == [True, True, True]: 
                runs_scored += 1
            #if 1st empty, batter goes there everything else stays the same
            elif bases[0] == False:
                bases =  [True] + bases[1:]
            #if 1st has someone on it but 2nd does not
            elif bases[1] == False:
                bases =  [True, True] + [bases[2]]
            #if 1st has someone on it(checked 2 before) and 2nd has somoeone on it(checked 1 before), and not bases loaded(checked 1st)
            #ie [True,True,False]
            else: 
                bases = [True,True,True]
        else:
            outs += 1    
        
        batter_index = (batter_index + 1) % len(batter_stats)
    return runs_scored
'''

# %%
def simulate_first_five_innings(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats):

    away_runs_total = 0
    home_runs_total = 0

    away_bttp = 0
    home_bttp = 0
    for i in range(5):    
        away_runs,_,away_bttp = simulate_inning(away_batter_stats, home_pitcher_stats,away_bttp)
        home_runs,_,home_bttp = simulate_inning(home_batter_stats, away_pitcher_stats,home_bttp)
        
        away_runs_total += away_runs
        home_runs_total += home_runs

#        print(f"End of inning {i}, score {away_runs_total}-{home_runs_total}, next batters {away_bttp},{home_bttp}")
    return away_runs_total,home_runs_total

# %%
def simulate_first_inning(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats):
    away_runs,away_hits,away_bttp = simulate_inning(away_batter_stats, home_pitcher_stats)
    home_runs,home_hits,home_bttp = simulate_inning(home_batter_stats, away_pitcher_stats)
    return away_runs,away_hits,away_bttp, home_runs,home_hits,home_bttp

# %%


# %%
# Main simulation
NUM_SIMULATIONS = 10000
#NUM_SIMULATIONS = 100
#todaysDate = "06/02/2023"

preds = {}
pred_games = []
pred_team_hits = []
pred_team_bttp = []
pred_game_hits = []
home_run_cur = {}
hrs_cur=[]#not used but makes it so no error

nrfi_preds_with_implied_odds = []


#iter = game_times[:1]
#tmp = batters[36:]
#print(iter)

print(odds_dict_nrfi)
print(game_times)
print(len(teams))
print(len(game_times))
for i in range(len(game_times)): 

#    i += 11
    away_batter_stats = batters[18 * i: 18 * i + 9]
    home_batter_stats = batters[18 * i + 9: 18 * i + 18]

    away_batter_stats = [away_batter_stats[i] for i in range(len(away_batter_stats)) if away_batter_stats[i] is not None]
    home_batter_stats = [home_batter_stats[i] for i in range(len(home_batter_stats)) if home_batter_stats[i] is not None]
 
    try:
#        away_pitcher_stats = get_pitcher_data(pitchers[2 * i])
#        home_pitcher_stats = get_pitcher_data(pitchers[2 * i + 1])
        away_pitcher_stats = pitchers_for_1st[2 * i]
        home_pitcher_stats = pitchers_for_1st[2 * i + 1]

        print(home_pitcher_stats)
        print(away_pitcher_stats)

    except:
        print("huh")
        continue

    runs_first_inning = 0
    total_away_runs = 0 
    total_home_runs = 0

    total_away_hits = 0
    total_home_hits = 0

    runs_first_inning_away = 0
    runs_first_inning_home = 0

    hits_first_inning_away = 0
    hits_first_inning_home = 0

    total_away_bttp = 0
    total_home_bttp = 0

    bttp_first_inning_away = 0
    bttp_first_inning_home = 0

    hits_first_inning_comp_one_and_a_half = 0

    for _ in range(NUM_SIMULATIONS):
        away_runs,away_hits,away_bttp, home_runs,home_hits,home_bttp = simulate_first_inning(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats)
        total_away_runs += away_runs
        total_home_runs += home_runs
        if away_runs + home_runs > 0:
            runs_first_inning += 1
            if away_runs > 0:
                runs_first_inning_away += 1 
            if home_runs > 0:
                runs_first_inning_home += 1 

        total_away_hits += away_hits
        total_home_hits += home_hits


        if away_hits + home_hits > 1.5:
            hits_first_inning_comp_one_and_a_half += 1
            
        if away_hits > 0:
            hits_first_inning_away += 1 
        if home_hits > 0:
            hits_first_inning_home += 1 

        total_away_bttp += away_bttp
        total_home_bttp += home_bttp
        
        if away_bttp > 3.5:
            bttp_first_inning_away += 1 
        if home_bttp > 3.5:
            bttp_first_inning_home += 1 



    # Calculate probability
    probability_run_first_inning = float(runs_first_inning / NUM_SIMULATIONS)
    average_away_runs = float(total_away_runs / NUM_SIMULATIONS)
    average_home_runs = float(total_home_runs / NUM_SIMULATIONS)
    average_total_runs = average_away_runs + average_home_runs
    prob_away_yrfi = float(runs_first_inning_away / NUM_SIMULATIONS)
    prob_home_yrfi = float(runs_first_inning_home / NUM_SIMULATIONS)

    average_away_hits = float(total_away_runs / NUM_SIMULATIONS)
    average_home_hits = float(total_home_runs / NUM_SIMULATIONS)    
    prob_away_hit = float(hits_first_inning_away / NUM_SIMULATIONS)
    prob_home_hit = float(hits_first_inning_home / NUM_SIMULATIONS)

    average_away_bttp = float(total_away_bttp / NUM_SIMULATIONS)
    average_home_bttp = float(total_home_bttp / NUM_SIMULATIONS)    
    prob_away_o3_bttp = float(bttp_first_inning_away / NUM_SIMULATIONS)
    prob_home_o3_bttp = float(bttp_first_inning_home / NUM_SIMULATIONS)

    prob_o1and5_hits = float(hits_first_inning_comp_one_and_a_half / NUM_SIMULATIONS)
    '''
    print(f"Probability of scoring more than 0.5 runs in the first inning: {probability_run_first_inning}")
    print(f"Average runs scored by {teams[2 * i]} team in the first inning: {average_away_runs:}")
    print(f"Average runs scored by {teams[2 * i + 1]} in the first inning: {average_home_runs}")
    print(f"Average runs scored in the first inning of game between {teams[2 * i]} and {teams[2 * i + 1]}: {average_away_runs + average_home_runs}")
    print(f"Average hits by {teams[2 * i]} team in the first inning: {average_away_hits}")
    print(f"Average hits by {teams[2 * i + 1]} in the first inning: {average_home_hits}")
    print(f"Probability away hit: {prob_away_hit}")
    print(f"Probability home hit: {prob_home_hit}")
    '''



    preds[teams[2 * i]] = average_away_runs
    preds[teams[2 * i + 1]] = average_home_runs
    
#    pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,prob_away_yrfi,prob_home_yrfi))

    c_or_e = confirmedOrExpected[2 * i]
    c_or_e = 'C' if c_or_e == 'C' and  c_or_e == confirmedOrExpected[2 * i + 1] else 'E'

    pred_team_hits.append((teams[2 * i],game_times[i],prob_away_hit,average_away_hits))
    pred_team_hits.append((teams[2 * i + 1],game_times[i],prob_home_hit,average_home_hits))

    pred_game_hits.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],prob_o1and5_hits,average_away_hits + average_home_hits))

    pred_team_bttp.append((teams[2 * i],game_times[i],prob_away_o3_bttp,average_away_bttp))
    pred_team_bttp.append((teams[2 * i + 1],game_times[i],prob_home_o3_bttp,average_home_bttp))

    print(f"{teams[2 * i]} @ {teams[2 * i + 1]}")
    print(game_times[i])
    if teams[2 * i] in odds_dict_nrfi:
        print(True)
        print(odds_dict_nrfi[teams[2 * i]])
    else:
        print(False)

    if teams[2 * i] in odds_dict_nrfi:
#        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",average_away_runs + average_home_runs,game_times[i],probability_run_first_inning,odds_dict[teams[2 * i]][probability_run_first_inning < .5],average_away_runs,average_home_runs))
        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,odds_dict_nrfi[teams[2 * i]][probability_run_first_inning < .5],prob_away_yrfi,prob_home_yrfi,c_or_e))
#        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,odds_dict[teams[2 * i]][probability_run_first_inning < .5],average_away_runs + average_home_runs,prob_away_yrfi,average_away_runs,prob_home_yrfi,average_home_runs))
        nrfi_preds_with_implied_odds.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,odds_dict_nrfi[teams[2 * i]][probability_run_first_inning < .5],c_or_e))
    else:
#        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",average_away_runs + average_home_runs,game_times[i],probability_run_first_inning,"N/A",average_away_runs,average_home_runs))
        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,"N/A",prob_away_yrfi,prob_home_yrfi,c_or_e))
#        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,"N/A",average_away_runs + average_home_runs,prob_away_yrfi,average_away_runs,prob_home_yrfi,average_home_runs))
        nrfi_preds_with_implied_odds.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,"N/A",c_or_e))
        
    print()


# %%
#batters[18*8 + 16] = get_batter_data('Dairon Blanco')

# %% [markdown]
# Model

# %% [markdown]
# Demo

# %%
[print(elem) for elem in pred_games]

# %%
pred_games = sorted(pred_games,key=lambda x :x[1])

'''
formatted_data = [
    {'Game': item[0], 'Prediction': round(item[1],2),'Time': item[2], 'Prob YRFI': item[3],'NRFI/YRFI line': item[4], 'Away Proj': item[5],'Home Proj': item[6]}
    for item in pred_games
]
'''

formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3], 'Prob Away YRFI': item[4],'Prob Home YRFI': item[5],'C/E': item[6]}
    for item in pred_games
]

'''
formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3],'Total pred':item[4] ,'Prob Away YRFI': item[5],'Away Proj': item[6],'Prob Home YRFI': item[7],'Home Proj': item[8]}
    for item in pred_games
]
'''
'''
formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'Prob Away YRFI': item[3],'Prob Home YRFI': item[4]}
    for item in pred_games
]
'''

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'NRFI/YRFI Chart {todaysDate}', fontsize=14)
ax.set_title(f'NRFI/YRFI Chart {datetime.now()}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/NRFIs.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
df = df.drop(['Time', 'NRFI/YRFI line','Prob Away YRFI','Prob Home YRFI','C/E'], axis=1)
df['Game'] = df['Game'].apply(lambda x: "".join(x.split()))
print(df.to_string(index=False, header=False))



# %%
pred_team_hits = sorted(pred_team_hits,key=lambda x :x[2],reverse=True)

'''
formatted_data = [
    {'Game': item[0], 'Prediction': round(item[1],2),'Time': item[2], 'Prob YRFI': item[3],'NRFI/YRFI line': item[4], 'Away Proj': item[5],'Home Proj': item[6]}
    for item in pred_games
]
'''
formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob Hit': item[2],'Proj Hits': item[3]}
    for item in pred_team_hits
]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'1st Inning Team Hit {todaysDate}', fontsize=14)
ax.set_title(f'1st Inning Team Hit {datetime.now()}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/1stInningHit.png', bbox_inches='tight', dpi=300)
plt.show()

# %% [markdown]
# 

# %%
#df = df.drop(['Prediction','Time', 'Away Proj','Home Proj'], axis=1)
#df = df.drop(['Time', 'NRFI/YRFI line','Prob Away YRFI','Prob Home YRFI'], axis=1)
df = df.drop(['Time'], axis=1)
df.insert(0,'Date',todaysDate)
#df['Game'] = df['Game'].apply(lambda x: "".join(x.split()))
print(df.to_string(index=False, header=False))



# %%
pred_team_bttp = sorted(pred_team_bttp,key=lambda x :x[2],reverse=True)

formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob O3.5 bttp': item[2],'Proj bttp': item[3]}
    for item in pred_team_bttp
]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'1st Inning Team Hit {todaysDate}', fontsize=14)
ax.set_title(f'1st Inning Team BTTP {datetime.now()}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/1stInningbttp.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
df = df.drop(['Time'], axis=1)
print(df.to_string(index=False, header=False))



# %%
#pred_games = sorted(pred_games,key=lambda x :x[3],reverse=True)
#pred_games = sorted(pred_games,key=lambda x :x[1])
pred_games = sorted(pred_games,key=lambda x :x[2],reverse=True)


pick_implied_odds = [implied_odds(elem[2]) if elem[2] > 0.5 else implied_odds(1 - elem[2]) for elem in pred_games]

formatted_data = [
#    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3], 'Prob Away YRFI': item[4],'Prob Home YRFI': item[5],'C/E':item[6]}
    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3], 'Imp. Odds (Pick)': pick_implied_odds[i],'Imp. Odds(NRFI)': implied_odds(1-item[2]), 'Prob Away YRFI': item[4],'Prob Home YRFI': item[5],'C/E':item[6]}    for i,item in enumerate(pred_games)
]


# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(14, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')


tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')


highlight_color = '#65fe08'
# Highlight rows where the 6th or 7th column value is greater than 50
for i in range(len(df)):
    if df.iloc[i, 3] != "N/A" and int(df.iloc[i, 3].replace('−', '-')) > int(df.iloc[i, 4].replace('−', '-')): 
        for j in range(len(df.columns)):
            tbl[(i + 1, j)].set_facecolor(highlight_color)  # +1 to account for header row



#ax.set_title(f'NRFI/YRFI Chart {todaysDate}', fontsize=14)
ax.set_title(f'NRFI/YRFI Chart {datetime.now()}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.figtext(0.5, 0.01, '+EV', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color, edgecolor='black'))

plt.savefig('images/NRFIs.png', bbox_inches='tight', dpi=300)
plt.show()

# %%


# %%
#pred_games = sorted(pred_games,key=lambda x :x[3],reverse=True)
#pred_games = sorted(pred_games,key=lambda x :x[1])
nrfi_preds_with_implied_odds = sorted(nrfi_preds_with_implied_odds,key=lambda x :x[2],reverse=True)

pick_implied_odds = [implied_odds(elem[2]) if elem[2] > 0.5 else implied_odds(1 - elem[2]) for elem in nrfi_preds_with_implied_odds]

formatted_data = [
#    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3], 'Prob Away YRFI': item[4],'Prob Home YRFI': item[5],'C/E':item[6]}
    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3], 'Imp. Odds (Pick)': pick_implied_odds[i],'Imp. Odds(NRFI)': implied_odds(1-item[2]),'C/E':item[4]}  for i,item in enumerate(nrfi_preds_with_implied_odds)
]


# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(11, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

ax.set_title(f'NRFI/YRFI Chart {todaysDate} with implied odds', fontsize=14)
#ax.set_title(f'NRFI/YRFI Chart {datetime.now()} with implied odds', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

#plt.savefig('images/NRFIs.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
'''
df['Game'] = df['Game'].apply(lambda x: "".join(x.split()))
df = df.drop(['Time', 'NRFI/YRFI line','Prob Away YRFI','Prob Home YRFI'], axis=1)
df = df.drop(['C/E'], axis=1)
df.insert(0,'Date',todaysDate)

print(df.to_string(index=False, header=False))

'''

# %%
pred_game_hits = sorted(pred_game_hits,key=lambda x :x[2],reverse=True)

'''
formatted_data = [
    {'Game': item[0], 'Prediction': round(item[1],2),'Time': item[2], 'Prob YRFI': item[3],'NRFI/YRFI line': item[4], 'Away Proj': item[5],'Home Proj': item[6]}
    for item in pred_games
]
'''
'''
formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3], 'Prob Away YRFI': item[5],'Prob Home YRFI': item[7]}
    for item in pred_games
]
'''
'''
formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3],'Total pred':item[4] ,'Prob Away YRFI': item[5],'Away Proj': item[6],'Prob Home YRFI': item[7],'Home Proj': item[8]}
    for item in pred_games
]
'''

formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob O1.5 hits': item[2],'Proj Hits': round(item[3],2)}
    for item in pred_game_hits
]


# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'O/U 1.5 hits in the 1st Chart {todaysDate}', fontsize=14)
ax.set_title(f'O/U 1.5 hits in the 1st Chart {datetime.now()}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/ou1_5hits1st.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
#df = df.drop(['Prediction','Time', 'Away Proj','Home Proj'], axis=1)
#df = df.drop(['Time', 'NRFI/YRFI line','Prob Away YRFI','Prob Home YRFI'], axis=1)
df = df.drop(['Time'], axis=1)
df['Game'] = df['Game'].apply(lambda x: "".join(x.split()))
print(df.to_string(index=False, header=False))



# %%
#url = 'https://sportsbook.draftkings.com/leagues/baseball/mlb?category=innings&subcategory=1st-5-innings'
url = 'https://sportsbook.draftkings.com/leagues/baseball/mlb?category=1st-x-innings'
response = requests.get(url,headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
teamsAndLines = soup.find_all("div", class_="sportsbook-event-accordion__wrapper expanded")

awayTeams = [elem.text.split()[0] for elem in teamsAndLines]

teamsAndLines = [elem.text for elem in teamsAndLines]
odds_dict_f5 = {}
odds = []

for elem in teamsAndLines:
    line = elem.split('Both')[0]
    print(line)    

    try:
        plus = line.find('+')
        minus = line.find('−')
    except:
        print(odds)

    if plus == -1:
        odds.append(elem[minus : minus + 4])
        minus = line.find('−',minus+1)
        odds.append(elem[minus : minus + 4])
    elif minus < plus:
        odds.append(elem[minus : minus + 4])
        odds.append(elem[plus : plus + 4])
    else:
        odds.append(elem[plus : plus + 4])
        odds.append(elem[minus : minus + 4])

    odds_dict_f5 = {awayTeams[i]: odds[2 * i: 2 * i + 2] for i in range(len(awayTeams))}


print(odds_dict_f5)


# %%
from collections import Counter

NUM_SIMULATIONS = 10000


home_run_idea = Counter()
home_run_idea_top3 = Counter()


# Main simulation
#NUM_SIMULATIONS = 10000
#NUM_SIMULATIONS = 20
#odds_dict={}
hrs = []

preds_f5s = {}
pred_games_f5 = []

pred_games_f5_no_betting_odds = []

everything_for_chart = []
hrs_cur = []

#iter = game_times[:1]
#tmp = batters[36:]
#print(iter)

print(game_times)
print(odds_dict_f5)

for i in range(len(game_times)): 

#    i = 1
#    i += 11
    away_batter_stats = batters[18 * i: 18 * i + 9]
    home_batter_stats = batters[18 * i + 9: 18 * i + 18]

    away_batter_stats = [away_batter_stats[i] for i in range(len(away_batter_stats)) if away_batter_stats[i] is not None]
    home_batter_stats = [home_batter_stats[i] for i in range(len(home_batter_stats)) if home_batter_stats[i] is not None]
 
    try:
#        away_pitcher_stats = get_pitcher_data(pitchers[2 * i])
#        home_pitcher_stats = get_pitcher_data(pitchers[2 * i + 1])
        away_pitcher_stats = pitchers[2 * i]
        home_pitcher_stats = pitchers[2 * i + 1]

    except:
        print("huh")
        continue

#    home_run_cur = {}

    runs_first_inning_more_than_4_5 = 0
    total_away_runs_f5 = 0 
    total_home_runs_f5 = 0

    away_wins_f5 = 0
    away_wins_f5_by_at_least2 = 0

    home_wins_f5 = 0
    home_wins_f5_by_at_least2 = 0

    ties_f5 = 0

    c_or_e = confirmedOrExpected[2 * i]
    c_or_e = 'C' if c_or_e == 'C' and  c_or_e == confirmedOrExpected[2 * i + 1] else 'E'

    home_run_cur = {}


    for _ in range(NUM_SIMULATIONS):

#        home_run_cur = {}
        hrs_cur = []

        away_runs, home_runs = simulate_first_five_innings(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats)
        total_away_runs_f5 += away_runs
        total_home_runs_f5 += home_runs

        if away_runs > home_runs:
            if away_runs - 1.5 > home_runs:
                away_wins_f5_by_at_least2 += 1
            away_wins_f5 += 1
        elif away_runs < home_runs:
            if home_runs - 1.5 > away_runs:
                home_wins_f5_by_at_least2 += 1
            home_wins_f5 += 1
        else:
            ties_f5 += 1

        if away_runs + home_runs > 4.5:
            runs_first_inning_more_than_4_5 += 1
        hrs.extend(hrs_cur)

    print(len(hrs))   

    sorted_home_run_cur = dict(sorted(home_run_cur.items(), key=lambda item: item[1], reverse=True))
    # Print the sorted dictionary
    home_run_idea += Counter(sorted_home_run_cur)

    home_run_idea_top3 += Counter(dict(list(sorted_home_run_cur.items())[:3]))
    print("Sorted dictionary by value:", sorted_home_run_cur)
    print(len(home_run_idea_top3))
    print(home_run_idea_top3)


    # Calculate probability
#    probability_o_4_5_runs_first_5_innings = round(float(runs_first_inning_more_than_4_5 / NUM_SIMULATIONS),2)
    probability_o_4_5_runs_first_5_innings = float(runs_first_inning_more_than_4_5 / NUM_SIMULATIONS)
    average_away_runs_f5 = round(float(total_away_runs_f5 / NUM_SIMULATIONS),2)
    average_home_runs_f5 = round(float(total_home_runs_f5 / NUM_SIMULATIONS),2)
    average_total_runs_f5 = round(average_home_runs_f5 + average_away_runs_f5,2)

    away_win_pct_f5 = round(away_wins_f5/NUM_SIMULATIONS  * 100,2)
    home_win_pct_f5 = round(home_wins_f5/NUM_SIMULATIONS  * 100,2)
    ties_f5_pct = round(ties_f5/NUM_SIMULATIONS * 100,2)

        
    away_wins_by_at_least2_pct = round(away_wins_f5_by_at_least2/NUM_SIMULATIONS  * 100,2)
    home_wins_by_at_least2_pct = round(home_wins_f5_by_at_least2/NUM_SIMULATIONS  * 100,2)

    print(f"Probability of scoring more than 4.5 runs in the first 5 innings: {probability_o_4_5_runs_first_5_innings}")
    print(f"Average runs scored by {teams[2 * i]} in the first 5 innings: {average_away_runs_f5}")
    print(f"Average runs scored by {teams[2 * i + 1]} in the first 5 innings: {average_home_runs_f5}")
    print(f"Average runs scored in the first 5 innings of game between {teams[2 * i]} and {teams[2 * i + 1]}: {average_total_runs_f5}")


#    sorted_home_run_cur = dict(sorted(home_run_cur.items(), key=lambda item: item[1], reverse=True))

    # Print the sorted dictionary
    
#    print(home_run_cur)


    preds_f5s[teams[2 * i]] = average_away_runs_f5
    preds_f5s[teams[2 * i + 1]] = average_home_runs_f5
    
#    pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,prob_away_yrfi,prob_home_yrfi))

    c_or_e = confirmedOrExpected[2 * i]
    c_or_e = 'C' if c_or_e == 'C' and  c_or_e == confirmedOrExpected[2 * i + 1] else 'E'

    pred_games_f5_no_betting_odds.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],average_away_runs_f5,average_home_runs_f5,away_win_pct_f5,home_win_pct_f5,away_wins_by_at_least2_pct,home_wins_by_at_least2_pct,c_or_e))

    '''
    if teams[2 * i] in odds_dict:
#        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",average_away_runs + average_home_runs,game_times[i],probability_run_first_inning,odds_dict[teams[2 * i]][probability_run_first_inning < .5],average_away_runs,average_home_runs))
        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,odds_dict[teams[2 * i]][probability_run_first_inning < .5],prob_away_yrfi,prob_home_yrfi,c_or_e))
#        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,odds_dict[teams[2 * i]][probability_run_first_inning < .5],average_away_runs + average_home_runs,prob_away_yrfi,average_away_runs,prob_home_yrfi,average_home_runs))
    else:
#        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",average_away_runs + average_home_runs,game_times[i],probability_run_first_inning,"N/A",average_away_runs,average_home_runs))
        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,"N/A",prob_away_yrfi,prob_home_yrfi,c_or_e))
#        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,"N/A",average_away_runs + average_home_runs,prob_away_yrfi,average_away_runs,prob_home_yrfi,average_home_runs))
    '''

    print()
    everything_for_chart.append((todaysDate,f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_o_4_5_runs_first_5_innings,away_win_pct_f5,home_win_pct_f5,ties_f5_pct,away_wins_by_at_least2_pct,home_wins_by_at_least2_pct)) 
    if teams[2 * i] in odds_dict_f5:
        pred_games_f5.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_o_4_5_runs_first_5_innings,average_total_runs_f5,average_away_runs_f5,average_home_runs_f5,away_win_pct_f5,home_win_pct_f5,ties_f5_pct,odds_dict_f5[teams[2 * i]][home_win_pct_f5 > away_win_pct_f5]))
    else:
        pred_games_f5.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_o_4_5_runs_first_5_innings,average_total_runs_f5,average_away_runs_f5,average_home_runs_f5,away_win_pct_f5,home_win_pct_f5,ties_f5_pct,'N/A'))
    print()

# %%
print(odds_dict_f5)


# %%
'''
# Main simulation
from collections import Counter

NUM_SIMULATIONS = 10000


home_run_idea = Counter()
home_run_idea_top3 = Counter()


for i in range(len(game_times)): 
    
    away_batter_stats = batters[18 * i: 18 * i + 9]
    home_batter_stats = batters[18 * i + 9: 18 * i + 18]

    away_batter_stats = [away_batter_stats[i] for i in range(len(away_batter_stats)) if away_batter_stats[i] is not None]
    home_batter_stats = [home_batter_stats[i] for i in range(len(home_batter_stats)) if home_batter_stats[i] is not None]
 
    try:
        away_pitcher_stats = pitchers[2 * i]
        home_pitcher_stats = pitchers[2 * i + 1]

    except:
        print("huh")
        continue

    home_run_cur = {}


    for _ in range(NUM_SIMULATIONS):
        away_runs, home_runs = simulate_first_five_innings(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats)
    
    sorted_home_run_cur = dict(sorted(home_run_cur.items(), key=lambda item: item[1], reverse=True))
    # Print the sorted dictionary
    home_run_idea += Counter(sorted_home_run_cur)

    home_run_idea_top3 += Counter(dict(list(sorted_home_run_cur.items())[:3]))
    print("Sorted dictionary by value:", sorted_home_run_cur)

#home_run_idea = dict(sorted(home_run_idea.items(), key=lambda item: item[1], reverse=True))
'''
formatted_data = [{'Name': name} for name,pred in home_run_idea_top3.items()]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'F5 Chart {todaysDate}', fontsize=14)
#ax.set_title(f'Home Run Chart {datetime.now()}', fontsize=14)


tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/HRs.png', bbox_inches='tight')

plt.show()

# %%
print(df.to_string(index=False, header=False))

# %%
formatted_data = [{'Name': name, 'Pred': pred/1000}   for name, pred in list(sorted(home_run_idea.items(), key=lambda item: item[1], reverse=True))[:45]]


# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'F5 Chart {todaysDate}', fontsize=14)
ax.set_title(f'Home Run Chart by pred ignore num idk {datetime.now()}', fontsize=14)


tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.show()
plt.savefig('images/hr_by_pred')

# %%
print(df.to_string(index=False, header=False))

# %%
'''
# Main simulation
from collections import Counter

NUM_SIMULATIONS = 10000
hrs = []


for i in range(len(game_times)): 
    
    away_batter_stats = batters[18 * i: 18 * i + 9]
    home_batter_stats = batters[18 * i + 9: 18 * i + 18]

    away_batter_stats = [away_batter_stats[i] for i in range(len(away_batter_stats)) if away_batter_stats[i] is not None]
    home_batter_stats = [home_batter_stats[i] for i in range(len(home_batter_stats)) if home_batter_stats[i] is not None]
 
    try:
        away_pitcher_stats = pitchers[2 * i]
        home_pitcher_stats = pitchers[2 * i + 1]

    except:
        print("huh")
        continue

    for _ in range(NUM_SIMULATIONS):
        home_run_cur = {}
        hrs_cur = []
        away_runs, home_runs = simulate_first_five_innings(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats)
#        print(hrs_cur)
        hrs.extend(hrs_cur)
    print(len(hrs))
'''

# %%
print(hrs)

# %%
print(Counter(hrs))

# %%
hrs_by_pct = Counter(hrs)
hrs_by_pct = dict(list(sorted(hrs_by_pct.items(), key=lambda item: item[1], reverse=True))[:45])

#    print("Sorted dictionary by %:", hrs_by_pct)


formatted_data = [{'Name': name,'Prob': prob/NUM_SIMULATIONS} for name,prob in hrs_by_pct.items()]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')


tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

#plt.savefig('images/HRs.png', bbox_inches='tight')
ax.set_title(f'Home Run Chart by prob idk {datetime.now()}', fontsize=14)


plt.show()
plt.savefig('images/hr_by_prob')

# %%
print(df.to_string(index=False, header=False))


# %%
'''
df = df.drop(['Time', 'Avg Total','Away Avg','Home Avg','ML Odds'], axis=1)
df['Game'] = df['Game'].apply(lambda x: "".join(x.split()))
print(df.to_string(index=False, header=False))
'''



# %%

print(pred_games_f5[0])
#pred_games_f5 = sorted(pred_games_f5,key=lambda x :x[2],reverse=True)
pred_games_f5 = sorted(pred_games_f5,key=lambda x :x[1])

formatted_data = [
    {'Game': item[0],'Time': item[1],'Avg Total': item[3], 'Away Avg': item[4],'Home Avg': item[5],'Away Win%': item[6],'Home Win%': item[7],'Tie %': item[8],'ML Odds': item[9]}
    for item in pred_games_f5
]


# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'F5 Chart {todaysDate}', fontsize=14)
ax.set_title(f'F5 Chart {datetime.now()}', fontsize=14)


highlight_color = '#65fe08'
for i in range(len(df)):
#    if df.iloc[i, 6] > 50 or df.iloc[i, 7] > 50: 
    if df.iloc[i, 5] > 50 or df.iloc[i, 6] > 50:
        home_team = df.iloc[i, 0].split()
        away_team = home_team[0] 
        home_team = home_team[-1]
        if df.iloc[i, 5] > 50:
            print(f"{df.iloc[i, 0]} - {away_team}")
        elif df.iloc[i, 6] > 50:
            print(f"{df.iloc[i, 0]} - {home_team}")
                 

        for j in range(len(df.columns)):
            tbl[(i + 1, j)].set_facecolor(highlight_color)  # +1 to account for header row
            
# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.figtext(0.5, 0.01, 'Green = >50%', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color, edgecolor='black'))

plt.savefig('images/F5.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
df = df.drop(['Time', 'Avg Total','Away Avg','Home Avg','ML Odds'], axis=1)
df['Game'] = df['Game'].apply(lambda x: "".join(x.split()))
print(df.to_string(index=False, header=False))



# %%

print(pred_games_f5[0])
pred_games_f5 = sorted(pred_games_f5_no_betting_odds,key=lambda x :x[1],reverse=True)
#pred_games_f5 = sorted(pred_games_f5,key=lambda x :x[1])

formatted_data = [
    #{'Game': item[0],'Time': item[1], 'Away -1.5 %': item[6],'Home -1.5% ': item[7]}
    {'Game': item[0],'Time': item[1], 'Away Avg': item[2],'Home Avg': item[3],'Away Win%': item[4],'Home Win%': item[5],'Away -1.5 %': item[6],'Home -1.5 %': item[7],'C/E': item[8]}

    #for item in pred_games_f5_1_5
    for item in pred_games_f5_no_betting_odds
]


# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'F5 Chart {todaysDate}', fontsize=14)
#ax.set_title(f'F5 ML and -1.5 Chart {todaysDate}', fontsize=14)
ax.set_title(f'F5 ML and -1.5 Chart {datetime.now()}', fontsize=14)


highlight_color_1 = '#65fe08'
highlight_color_2 = '#FFD700'

for i in range(len(df)):

    if df.iloc[i, 6] > 66 or df.iloc[i, 7] > 66: 
        for j in range(len(df.columns)):
            tbl[(i + 1, j)].set_facecolor(highlight_color_2)  

    elif df.iloc[i, 4] > 50 or df.iloc[i, 5] > 50: 
        for j in range(len(df.columns)):
            tbl[(i + 1, j)].set_facecolor(highlight_color_1)  
            
# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.figtext(0.5, 0.046, 'Gold = -1.5 >66% ', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color_2, edgecolor='black'))
plt.figtext(0.5, 0.01, 'Green = ml >50% ', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color_1, edgecolor='black'))

plt.savefig('images/F5__ML_1_5.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
df = df.drop(['Time','Away Avg','Home Avg','C/E'], axis=1)
#df = df.drop(['C/E'], axis=1)
df['Game'] = df['Game'].apply(lambda x: "".join(x.split()))
df.insert(0,'Date',todaysDate)
#df.insert(4,'Space','')
#df.insert(5,'Space2','')

print(df.to_string(index=False, header=False))



# %%

#print(pred_games_f5[0])
everything_for_chart = sorted(everything_for_chart,key=lambda x :x[3],reverse=True)
#pred_games_f5 = sorted(pred_games_f5,key=lambda x :x[1])

formatted_data = [
    #{'Game': item[0],'Time': item[1], 'Away -1.5 %': item[6],'Home -1.5% ': item[7]}
    {'Date':item[0],'Game': item[1],'Time': item[2],'Prob O4.5':item[3],'Away Win%': item[4],'Home Win%': item[5],'tie%':item[6],'Away -1.5 %': item[7],'Home -1.5 %': item[8]}

    #for item in pred_games_f5_1_5
    for item in everything_for_chart
]
#    everything_for_chart.append((away_win_pct_f5,home_win_pct_f5,ties_f5_pct,away_wins_by_at_least2_pct,home_wins_by_at_least2_pct)) 


# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'F5 Chart {todaysDate}', fontsize=14)
ax.set_title(f'everything for chart {todaysDate}', fontsize=14)
#ax.set_title(f'F5 ML and -1.5 Chart {datetime.now()}', fontsize=14)


highlight_color_1 = '#65fe08'
highlight_color_2 = '#FFD700'

for i in range(len(df)):

    if df.iloc[i, 7] > 50 or df.iloc[i, 8] > 50: 
        for j in range(len(df.columns)):
            tbl[(i + 1, j)].set_facecolor(highlight_color_2)  

    elif df.iloc[i, 4] > 50 or df.iloc[i, 5] > 50: 
        for j in range(len(df.columns)):
            tbl[(i + 1, j)].set_facecolor(highlight_color_1)  
            
# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.figtext(0.5, 0.046, 'Gold = -1.5 >50% ', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color_2, edgecolor='black'))
plt.figtext(0.5, 0.01, 'Green = ml >50% ', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color_1, edgecolor='black'))

plt.show()

# %%
df = df.drop(['Time'], axis=1)
df['Game'] = df['Game'].apply(lambda x: "".join(x.split()))

df['tie%'] = df['tie%'].apply(lambda x: f'{x}  ')

print(df.to_string(index=False, header=False))
