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
todaysDate = (datetime.now()).strftime('%m/%d/%Y')

# Function to get the date 7 days ago
def get_date_7_days_ago(date_str):
    # Parse the input date string to a datetime object
    date = datetime.strptime(date_str, '%m/%d/%Y')
 
    date_7_days_ago = date - timedelta(days=7)
    
    # Format the date back to 'MM/DD/YYYY'
    date_7_days_ago_str = date_7_days_ago.strftime('%m/%d/%Y')
    
    return date_7_days_ago_str

# %%
#Returns player info for past 50 games before the input date, from StatsMuse
def get_player_data(name, date):
    df = None

    seven_days_ago = get_date_7_days_ago(date)
    # can replace name to any player name, can change date to match any games before that date
    url = 'https://www.statmuse.com/mlb/ask?q=' + name.lower().replace(' ', '+') + '+stats+between+' + seven_days_ago.replace('/', '%2F') + '+and+' + date.replace('/', '%2F') + '+per+game+including+ops%2Cavg%2C+and+slg'
    response = requests.get(url)

    print(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using Pandas read_html function
        try:
            tables = pd.read_html(response.text)
            df = tables[0].head(25)

            #Just dont think enough data if less than 3 games with any at bats last 7 days
            if len(df) < 3:
                 return None

            df = df.filter(items=["NAME", "DATE","AVG","OPP","H","PA"])

        except  ValueError as ve:
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
                    
                    # Create DataFrame manually
                    df = pd.DataFrame(rows, columns=hders).head(25)
            else:
                print("No table found with BeautifulSoup either")
                return None
            
        expected_columns = ["NAME", "DATE", "AVG", "OPP","H","PA"]
        df = df.filter(items=expected_columns)
        
        df = df.fillna({"NAME": "ERROR", "DATE": "ERROR","AVG": 0,\
                                "OPP": "ERROR", "H": df.loc[:,'H'].mean(),"PA": df.loc[:,'PA'].mean()
                                })
        return df

# %%
todaysDate = (datetime.now() - timedelta(hours=4))
todaysDateHour = todaysDate.hour
yesterdaysDate = (todaysDate - timedelta(1)).strftime('%m/%d/%Y')
todaysDate = todaysDate.strftime('%m/%d/%Y')


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
    batters = soup.find_all('li',class_ = 'lineup__player')
    batters = [elem.find('a').get('title') for elem in batters]

    teams = soup.find_all('div',class_= 'lineup__abbr')
    teams = [elem.text for elem in teams]

    game_times = soup.find_all('div',class_="lineup__time")[:-2]
    game_times = [elem.text for elem in game_times]

    pitchers_stats = {}
    for i,pitcher in enumerate(pitchers):
        try:
            pitcher = pitcher.find('a')
            url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={pitcher.get('href').split('-')[-1]}&stats=pitching"
            response = requests.get(url,headers=headers)
            if response.status_code == 200:
                stats = response.json()
                
                team = stats['basic']['pitching']['body'][-1]['team']
                h_per_9 = stats['gamelog']['majors']['pitching']['footer'][1]
                ip = float(h_per_9['ip']['text'])
                partial_inning_part = ip - int(ip)
                ip = int(ip) + partial_inning_part * (10/3)
                h_per_9 = int(h_per_9['h']['text'])/ip * 9
                pitchers_stats[f"{team} @ {game_times[i//2]}"] = [h_per_9]    
        except:
            print(url)
            batters[(i * 9) -9: (i * 9)] = ["SKIP" for i in enumerate(batters[(i * 9) -9: (i * 9)])]
            continue

# %%
url = 'https://www.baseball-reference.com/leagues/majors/pitch.shtml'
response = requests.get(url,headers=headers)

#league_average_h_per_9 = None
print(response.status_code)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    league_average_h_per_9 = float(soup.find('td', attrs={'data-stat': 'hits_per_nine'})['csk'])
print(league_average_h_per_9)

# %%
#TODO: Come back to
players_hit_lines = {}
odds = [base[0]['oddsAmerican'], base[1]['oddsAmerican']]
players_hit_lines[f"{player_name} @ {game_times[indices[0]//18]}"] = line,odds


print(len(players_hit_lines))
[print(elem,players_hit_lines[elem]) for elem in players_hit_lines]

# %%
def model(name, date, opponent, projection, h_per_9_allowed,league_average_h_per_9, demo_mode=False):

#    print(f"name:{name},date:{date}, opponent:{opponent},projection:{projection},h_per_9_allowed:{h_per_9_allowed},{league_average_h_per_9},demo_mode:{demo_mode}")


    opponent = f"{opponent} @ {game_times[teams.index(opponent)//2]}"
    if opponent not in h_per_9_allowed:
        return None

    player_data = get_player_data(name, date)

    if player_data is None:
        print(name)
        return None
    
    avg = player_data.loc[:, 'AVG'].to_numpy()
    h = player_data.loc[:, 'H'].to_numpy()
    pa = player_data.loc[:, 'PA'].to_numpy()
    opponent_teams = player_data.loc[:, 'OPP'].to_numpy()
    
    mean_avg = np.mean(avg)
    std_dev_avg = np.std(avg)

    mean_h = np.mean(h)
    std_dev_h = np.std(h)
    mean_pa = np.mean(pa)
    std_dev_pa = np.std(pa)

    num_simulations = 10000
    actual_range = int(num_simulations/ len(h))
    
    # Arrays to store simulated results
    simulated_hits = np.zeros((actual_range, len(h)))

    # Calculate R-squared values for each variable
    r2_values = []
    for variable in [avg, h, pa]:
        X = np.array(variable).reshape(-1, 1)
        linear_model = LinearRegression()
        linear_model.fit(X, h)
        y_pred = linear_model.predict(X)
        r2_values.append(r2_score(h, y_pred))

    # Normalize R-squared values to create custom weights
    weights = np.array(r2_values) / np.sum(r2_values)

    # Simulate linear regression and prediction for points
    for i in range(actual_range):
        # Add some random noise to the true, usage, and point values to simulate variability
        simulated_avg = np.array(avg) + np.random.normal(mean_avg, std_dev_avg, len(avg))
        simulated_h = np.array(h) + np.random.normal(mean_h, std_dev_h, len(h))
        simulated_pa = np.array(pa) + np.random.normal(mean_pa, std_dev_pa,len(pa))

        # Perform linear regression for points with custom weights
        X = np.array([simulated_avg, simulated_h, simulated_pa]).T
        linear_model = LinearRegression()

        # Apply custom weights to each variable
        X_weighted = X * weights
    
        # Fit the model and predict points
        linear_model.fit(X_weighted, h)
        simulated_hits[i, :] = linear_model.predict(X_weighted)

    # Now simulated_points has the shape (num_simulations, len(point))
    adjusted_matchup = h_per_9_allowed[opponent][0]/ league_average_h_per_9
    simulated_hits *= adjusted_matchup

    median_predicted_hits = np.median(simulated_hits)
    if demo_mode == False:
        print(f"Name: {name}:")
        print(f"Line: {projection}")
        print(f"Median value of predicted hits: {median_predicted_hits:.2f}")

    mean_predicted_hits = np.mean(simulated_hits)
    if demo_mode == False:
        print(f"Mean value of predicted hits: {mean_predicted_hits:.2f}")

    max_predicted_hits = np.max(simulated_hits)
    if demo_mode == False:
        print(f"Ceiling value of predicted hits: {max_predicted_hits:.2f}")
    
    min_predicted_hits = np.min(simulated_hits)
    if demo_mode == False:
        print(f"Floor value of predicted hits: {min_predicted_hits:.2f}")
    
    #Finding over/under chances
    num_over = 0
    num_under = 0
    num_push = 0

    print(num_over)
    for set in simulated_hits:
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
        return (round(mean_predicted_hits,4), round(over_chance,4), round(under_chance,4))
    
# %%
[print(i,elem) for i,elem in enumerate(batters)]

# %%
preds = []
preds_to_record_a_hit = []

for i, elem in enumerate(batters):
    if elem != "SKIP":
        curTeamIndex = i // 9
        oppTeamIndex = 0
        if curTeamIndex % 2 == 0:
            oppTeamIndex = curTeamIndex + 1
        else:
            oppTeamIndex = curTeamIndex - 1
        
        # Adjust name for Luis García/Garcia
        if elem.lower() == "luis garcía" or elem.lower() == "luis garcia":
            elem += " jr"

        try:
            line = players_hit_lines[elem][0]
        except KeyError:
            line = 0.5

        # Original prediction with the given line
        pred = model(elem, todaysDate, teams[oppTeamIndex], line, pitchers_stats, league_average_h_per_9, True)
        if pred is not None:
            try:
                if pred[1] > 50:
                    odds = players_hit_lines[elem][1][0]
                else:
                    odds = players_hit_lines[elem][1][1]
            except KeyError:
                odds = 'N/A'  # Unknown odds

            pred_entry = (elem, pred, teams[curTeamIndex], game_times[curTeamIndex // 2], line, odds)
            preds.append(pred_entry)

            # Add to preds_to_record_a_hit if line is 0.5
            if line == 0.5:
                preds_to_record_a_hit.append(pred_entry)

        # Calculate another prediction with line set to 0.5 (for preds_to_record_a_hit) if the original line isn't 0.5
        if line != 0.5:
            pred_hit = model(elem, todaysDate, teams[oppTeamIndex], 0.5, pitchers_stats, league_average_h_per_9, True)
            if pred_hit is not None:
                try:
                    if pred_hit[1] > 50:
                        odds_hit = players_hit_lines[elem][1][0]
                    else:
                        odds_hit = players_hit_lines[elem][1][1]
                except KeyError:
                    odds_hit = 'N/A'  # Unknown odds

                pred_hit_entry = (elem, pred_hit, teams[curTeamIndex], game_times[curTeamIndex // 2], 0.5, odds_hit)
                preds_to_record_a_hit.append(pred_hit_entry)


[print(elem) for elem in preds]

preds = sorted(preds,key=lambda x :x[1][0],reverse=True)
[print(elem) for elem in preds]
include_in_image = preds[:10]

preds = sorted(preds,key=lambda x :(x[1][1],x[1][0]),reverse=True)
[print(elem) for elem in preds]

for elem in preds[:10]:
    if elem not in include_in_image:
        include_in_image.append(elem)

for elem in preds[10:]:
    if len(include_in_image) < 20 or elem[1][1] == 100.0:
        if elem not in include_in_image:
            include_in_image.append(elem)

print(include_in_image)
print(len(include_in_image))

[print(include_in_image[i]) for i in range(len(include_in_image))] 

formatted_data = [
    {'Name': item[0], 'Team': item[2],'Time':item[3], 'Prediction': round(item[1][0],2), 'Odds Over': round(item[1][1],2),  'Line': item[4], 'Odds': item[5]}
    for item in include_in_image
]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

ax.set_title(f'Hits Chart {todaysDate}', fontsize=14)


# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('Output/MostHits.png', bbox_inches='tight', dpi=300)
plt.show()

sorted_preds = sorted(preds,key=lambda x :(x[1][1],x[1][0]),reverse=True)
formatted_data = [
    {'Name': item[0], 'Team': item[2],'Time':item[3], 'Prediction': round(item[1][0],2), 'Odds 1+ hits': round(item[1][1],2)}
    for item in sorted_preds[:20]
]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

ax.set_title(f'Most likely to record a hit {todaysDate}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('Output/MostLikelyHit.png', bbox_inches='tight', dpi=300)
plt.show()


formatted_data = [
    {'Name': item[0], 'Team': item[2],'Time':item[3], 'Prediction': round(item[1][0],2), 'Odds 1+ hits': round(item[1][1],2)}
    for item in sorted_preds[-20:]
]

df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

ax.set_title(f'Least likely to record a hit {datetime.now()}', fontsize=14)


# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('Outputs/LeastLikelyHit.png', bbox_inches='tight', dpi=300)
plt.show()

