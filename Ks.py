# %%
import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}

# %%
#Suppressing Warnings for better looking output if desired
import warnings
warnings.filterwarnings('ignore')

# %%
todaysDate = (datetime.now() - timedelta(hours=4))
todaysDateHour = todaysDate.hour
yesterdaysDate = (todaysDate - timedelta(1)).strftime('%m/%d/%Y')
todaysDate = todaysDate.strftime('%m/%d/%Y')

# %%
# Function to get the date 7 days ago
def date_30_days_ago_str(date_str):
    # Parse the input date string to a datetime object
    date = datetime.strptime(date_str, '%m/%d/%Y')
 
    date_30_days_ago = date - timedelta(days=30)
    
    # Format the date back to 'MM/DD/YYYY'
    date_30_days_ago_str = date_30_days_ago.strftime('%m/%d/%Y')
    
    return date_30_days_ago_str
    
# %%
def get_player_data(name, date):
    df = None
    thirty_days_ago = date_30_days_ago_str(date)

    url = 'https://www.statmuse.com/mlb/ask?q=' + name.lower().replace(' ', '+') + '+k%2F9+for+each+game+between+' + thirty_days_ago.replace('/', '%2F') + '+and+' + date.replace('/', '%2F')

    response = requests.get(url)

    print(url,response.status_code)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using Pandas read_html function
        try:
            tables = pd.read_html(response.text)
            df = tables[0].head(25)

            #Just dont think enough data if less than 3 games with any at bats last 7 days
            if len(df) < 3:
                 return None
            df = df.filter(items=["NAME", "DATE","K/9","OPP","IP","SO"])

        except ValueError as ve:
            print(f"pd.read_html failed: {ve}")
            # Use BeautifulSoup as a fallback
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')

            if table:
                    # Extract table headers
                    hders = [th.get_text(strip=True) for th in table.find_all('th')]
                    
                    # Extract table rows
                    rows = []
                    for tr in table.find_all('tr')[1:]:  # Skip header row
                        cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                        if len(cells) == len(hders):
                            rows.append(cells)
                    
                    df = pd.DataFrame(rows, columns=hders).head(25)
            else:
                print("No table found with BeautifulSoup either")
                return None
            
        expected_columns = ["NAME", "DATE","K/9","OPP","IP","SO"]
        df = df.filter(items=expected_columns)
            
        df = df.fillna({"NAME": "ERROR", "DATE": "ERROR","K/9": df.loc[:,'K/9'].mean(),\
                                "OPP": "ERROR", "IP": df.loc[:,'IP'].mean(),"SO": df.loc[:,'SO'].mean()
                                })
#        if df.max()['IP'] - df.min()['IP']:
    if df is not None:
        if df['K/9'].isna().any():
            return None
        #Just modified because otherwise code may think any decimal amount of innings possible for example
#        df['IP'] = df['IP'].apply(lambda ip: int(ip) + (ip - int(ip)) * (10 / 3))
#        if df['IP'].max() - df['IP'].min() < 2 + 1/3:#Try to prevent outliers
    return df

# %%
print('todaysDateHour:',todaysDateHour)
if todaysDateHour > 21 or todaysDateHour < 3 :
    url = 'https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow'
else:
    url = 'https://www.rotowire.com/baseball/daily-lineups.php'

response = requests.get(url,headers=headers)

pitchers = []
batters = []
teams = []
game_times = []
#confirmedOrExpected = []

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    pitchers = soup.find_all('div',class_='lineup__player-highlight-name')
    pitchers = [elem.find('a').text for elem in pitchers]

        
    batters = soup.find_all('li',class_ = 'lineup__player')
#    batters = [elem.find('a').get('title') for elem in batters]

    teams = soup.find_all('div',class_= 'lineup__abbr')
    teams = [elem.text for elem in teams]
#    teams[6:] = teams[8:]

    game_times = soup.find_all('div',class_="lineup__time")
    game_times = [elem.text for elem in game_times]
#    game_times[3:] = game_times[4:]

    team_strikeout_rate = {}
    team_so_rate = 0
    num_players = 0
    team = None
    for i,batter in enumerate(batters):
        try:
#            print(batter)
            batter = batter.find('a')
            url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={batter.get('href').split('-')[-1]}&stats=batting"
            response = requests.get(url,headers=headers)
            if response.status_code == 200:
                stats = response.json()
#                if i % 9 == 0:
#                    team = stats['basic']['batting']['body'][-1]['team']

                if team is None or len(team) > 3:
                    team = stats['basic']['batting']['body'][-1]['team']

                last7 = stats['gamelog']['majors']['batting']['footer'][0]

                pas = float(last7['pa']['text'])
                sos = float(last7['so']['text'])
                team_so_rate += sos/pas
                num_players += 1
#                print(i,num_players,team_so_rate,team)
                if (i - 8) % 9 == 0 and i != 0:
#                    print("Adding to dict")
                    team_strikeout_rate[team] = [team_so_rate/num_players]
                    team_so_rate = 0   
                    num_players = 0 
                    team = None
        except Exception as e:
            print(url)
            print(e)
            continue

# %%
url = 'https://www.baseball-reference.com/leagues/majors/bat.shtml'
response = requests.get(url,headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('tbody')
row = table.find('tr')

so = float(row.find('td', {'data-stat': 'SO'}).text.strip())
pa = float(row.find('td', {'data-stat': 'PA'}).text.strip())

league_average_so_per_ab = so/pa
print('league_average_so_per_ab: ',league_average_so_per_ab)

# %%
def model(name, date, opponent, projection, so_rate,league_average_so_rate, demo_mode=False):
    
    if opponent not in so_rate:
        return None

    player_data = get_player_data(name, date)

    if player_data is None:
        print(name)
        return None    

    k_per_9 = player_data.loc[:, 'K/9'].to_numpy()
    so = player_data.loc[:, 'SO'].to_numpy()
    ip = player_data.loc[:, 'IP'].to_numpy()
    opponent_teams = player_data.loc[:, 'OPP'].to_numpy()
    
    mean_k_per_9 = np.mean(k_per_9)
    std_dev_k_per_9 = np.std(k_per_9)

    mean_so = np.mean(so)
    std_dev_so = np.std(so)

    mean_ip = np.mean(ip)
    std_dev_ip = np.std(ip)

    num_simulations = 10000

    actual_range = int(num_simulations/ len(so))
    simulated_SOs = np.zeros((actual_range, len(so)))

    # Calculate R-squared values for each variable
    r2_values = []
    for variable in [k_per_9, so, ip]:
        X = np.array(variable).reshape(-1, 1)
        linear_model = LinearRegression()
        linear_model.fit(X, so)
        y_pred = linear_model.predict(X)
        r2_values.append(r2_score(so, y_pred))

    # Normalize R-squared values to create custom weights
    weights = np.array(r2_values) / np.sum(r2_values)

    # Simulate linear regression and prediction for points
    for i in range(actual_range):

        # Add some random noise to the true, usage, and point values to simulate variability
        simulated_k_per_9 = np.array(k_per_9) + np.random.normal(mean_k_per_9, std_dev_k_per_9, len(k_per_9))
        simulated_so = np.array(so) + np.random.normal(mean_so, std_dev_so, len(so))
        simulated_ip = np.array(ip) + np.random.normal(mean_ip, std_dev_ip,len(ip))

        # Perform linear regression for points with custom weights
        X = np.array([simulated_k_per_9, simulated_so, simulated_ip]).T
        linear_model = LinearRegression()

        # Apply custom weights to each variable
        X_weighted = X * weights
    
        # Fit the model and predict points
        linear_model.fit(X_weighted, so)
        simulated_SOs[i, :] = linear_model.predict(X_weighted)

    # Now simulated_points has the shape (num_simulations, len(point))
    adjusted_matchup = so_rate[opponent][0]/ league_average_so_rate
    simulated_SOs *= adjusted_matchup

    median_predicted_Ks = np.median(simulated_SOs)
    if demo_mode == False:
        print(f"Name: {name}:")
        print(f"Line: {projection}")
        print(f"Median value of predicted Ks: {median_predicted_Ks:.2f}")

    mean_predicted_Ks = np.mean(simulated_SOs)
    if demo_mode == False:
        print(f"Mean value of predicted Ks: {mean_predicted_Ks:.2f}")

    max_predicted_Ks = np.max(simulated_SOs)
    if demo_mode == False:
        print(f"Ceiling value of predicted Ks: {max_predicted_Ks:.2f}")
    
    min_predicted_Ks = np.min(simulated_SOs)
    if demo_mode == False:
        print(f"Floor value of predicted Ks: {min_predicted_Ks:.2f}")
    
    #Finding over/under chances
    num_over = 0
    num_under = 0
    num_push = 0
    for set in simulated_SOs:
        for num in set:
            if num > projection:
                num_over = num_over + 1
            elif num < projection:
                num_under = num_under + 1
            else:
                num_push = num_push + 1

    over_chance = 100 * num_over/(num_over + num_under + num_push)
    under_chance = 100 * num_under/(num_over + num_under + num_push)

    if demo_mode == False:
        print(f"Over Odds from my model: {over_chance}")
        print(f"Under Odds from my model: {under_chance}")
        print()

    #Giving return to demo
    if demo_mode == True:
        return (round(mean_predicted_Ks,4), round(over_chance,4), round(under_chance,4))
    

# %%

pitcher_so_lines = {}

preds = []
preds_dict = {}
for i,elem in enumerate(pitchers):
    if elem != "SKIP":
        oppTeamIndex = 0
        if i % 2 == 0:
            oppTeamIndex = i + 1
        else:
            oppTeamIndex = i - 1

        try:
            line = pitcher_so_lines[elem][0]
        except KeyError:
            line = 4.5
   
        pred = model(elem, todaysDate, teams[oppTeamIndex], line, team_strikeout_rate, league_average_so_per_ab,True)
        if pred is not None:
            try:
                
                if pred[0] > line:
                    odds = pitcher_so_lines[elem][1][0]
                else:
                    odds = pitcher_so_lines[elem][1][1]
                '''
                if pred[1] > 50:
                    odds = pitcher_so_lines[elem][1][0]
                else:
                    odds = pitcher_so_lines[elem][1][1]
                '''
            except KeyError:
                    odds = 'N/A' #Idk

            preds_dict[elem] = pred
            pred = (elem,pred,teams[i],game_times[i//2],line,odds)
            preds.append(pred)

# %%
preds = sorted(preds,key=lambda x :x[1][0],reverse=True)
[print(elem) for elem in preds]

# %%
preds = sorted(preds,key=lambda x :x[1][1],reverse=True)
[print(elem) for elem in preds]

# %%
preds = sorted(preds,key=lambda x :x[3])
[print(elem) for elem in preds]

# %%
preds = sorted(preds,key=lambda x :float(x[1][0])-float(x[4]),reverse=True)

formatted_data = [
    {'Name': item[0], 'Team': item[2], 'Prediction': round(item[1][0],2), 'Line': item[4], 'Difference': round(float(item[1][0])-float(item[4]),2), 'Odds': item[5]}
    for item in preds
]

df = pd.DataFrame(formatted_data)

plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

highlight_color = '#65fe08'
# Highlight rows where the 6th or 7th column value is greater than 50
for i in range(len(df)):
    if abs(df.iloc[i, 4]) >= 1: 
        for j in range(len(df.columns)):
            tbl[(i + 1, j)].set_facecolor(highlight_color)  # +1 to account for header row



ax.set_title(f'Strikeouts Chart {todaysDate}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.figtext(0.5, 0.01, 'Diff >= 1', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color, edgecolor='black'))


plt.savefig('Outputs/Ks.png', bbox_inches='tight', dpi=300)
plt.show()
