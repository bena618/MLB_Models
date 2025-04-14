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
#todaysDate = '9/19/2024'
todaysDate = (datetime.now()).strftime('%m/%d/%Y')


# %%
from datetime import datetime, timedelta

# Function to get the date 7 days ago
def get_date_7_days_ago(date_str):
    # Parse the input date string to a datetime object
    date = datetime.strptime(date_str, '%m/%d/%Y')
 
    date_7_days_ago = date - timedelta(days=7)
    
    # Format the date back to 'MM/DD/YYYY'
    date_7_days_ago_str = date_7_days_ago.strftime('%m/%d/%Y')
    
    return date_7_days_ago_str

# %% [markdown]
# Data Grabbing Function

# %%
#Returns player info for past 50 games before the input date, from StatsMuse
def get_player_data(name, date):
    df = None
#    if name == 'Levi Jordan':
#        return None
#el
    if name == 'Enrique Hernandez':
        name = 'kike Hernandez'
    seven_days_ago = get_date_7_days_ago(date)
    # can replace name to any player name, can change date to match any games before that date
    url = 'https://www.statmuse.com/mlb/ask?q=' + name.lower().replace(' ', '+') + '+stats+between+' + seven_days_ago.replace('/', '%2F') + '+and+' + date.replace('/', '%2F') + '+per+game+including+ops%2Cavg%2C+and+slg'
    # get page content
    response = requests.get(url)

    print(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using Pandas read_html function
        try:
            tables = pd.read_html(response.text)
#            print(url)
            df = tables[0].head(25)

            #Just dont think enough data if less than 3 games with any at bats last 7 days
            if len(df) < 3:
                 return None

    #        df = df.filter(items=["NAME", "DATE","OPS","AVG","SLG", "OPP","AB", "H", "HR", "SH", "SF", "IBB", "BB", "HBP", "SO", "PA", "TB"])
#            df = df.filter(items=["NAME", "DATE","OPS","AVG","SLG", "OPP","AB", "H", "SO", "PA"])
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
            
#        expected_columns = ["NAME", "DATE", "OPS", "AVG", "SLG", "OPP", "AB", "H", "SO", "PA"]
        expected_columns = ["NAME", "DATE", "AVG", "OPP","H","PA"]
        df = df.filter(items=expected_columns)
            
        '''
        # filling NaNs
        df = df.fillna({"NAME": "ERROR", "DATE": "ERROR", "OPS": df.loc[:,'OPS'].mean(),\
                                "AVG": df.loc[:,'AVG'].mean(),"SLG": df.loc[:,'SLG'].mean(),\
                                "OPP": "ERROR","AB": df.loc[:,'AB'].mean(), "H": df.loc[:,'H'].mean(),\
                                "HR": df.loc[:,'HR'].mean(), \
                                "SH": df.loc[:,'SH'].mean(), "SF": df.loc[:,'SF'].mean(),\
                                "IBB": df.loc[:,'IBB'].mean(), "BB": df.loc[:,'BB'].mean(),\
                                "HBP": df.loc[:,'HBP'].mean(), "SO": df.loc[:,'SO'].mean(),\
                                "PA": df.loc[:,'PA'].mean(), "TB": df.loc[:,'TB'].mean(),\
                                })
        
        df = df.fillna({"NAME": "ERROR", "DATE": "ERROR", "OPS": df.loc[:,'OPS'].mean(),\
                                "AVG": df.loc[:,'AVG'].mean(),"SLG": df.loc[:,'SLG'].mean(),\
                                "OPP": "ERROR","AB": df.loc[:,'AB'].mean(), "H": df.loc[:,'H'].mean(),\
                                "SO": df.loc[:,'SO'].mean(),"PA": df.loc[:,'PA'].mean()
                                })
        '''                                             #was df.loc[:,'AVG'].mean()
        df = df.fillna({"NAME": "ERROR", "DATE": "ERROR","AVG": 0,\
                                "OPP": "ERROR", "H": df.loc[:,'H'].mean(),"PA": df.loc[:,'PA'].mean()
                                })
        return df

# %%
'''
name = 'Rob Refsnyder'
date = '7/3/2024'
print(get_player_data(name, date))
#print(get_player_data(name, date).loc[:, 'PTS'].to_numpy())
'''

# %%
url = 'https://www.rotowire.com/baseball/daily-lineups.php'
#url = 'https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow'

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

#    confirmedOrExpected = soup.find_all('li',class_="lineup__status")
#    confirmedOrExpected = [elem.text.strip().split()[0][0] for elem in confirmedOrExpected]

#    pitchers = pitchers[2:]
    pitchers_stats = {}
    for i,pitcher in enumerate(pitchers):
        try:
            pitcher = pitcher.find('a')
            url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={pitcher.get('href').split('-')[-1]}&stats=pitching"
            response = requests.get(url,headers=headers)
            if response.status_code == 200:
                stats = response.json()
                team = stats['basic']['pitching']['body'][-1]['team']

#                if stats['gamelog']:
                h_per_9 = stats['gamelog']['majors']['pitching']['footer'][1]
#                else:
#                    h_per_9 = stats['gl2024']['majors']['pitching']['footer'][1]
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
tmp = game_times
#game_times = game_times[:-1]

#confirmedOrExpected[0:] = confirmedOrExpected[1:]

print(len(game_times))

# %%
print(len(batters))
#batters[0:] = batters[18:] 
print(len(batters))
#game_times[7:] = game_times[9:]
#teams[14:] = teams[18:]

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
players_hit_lines = {}
with open(f"batterprops/batterprops{"_".join(todaysDate.split('/')[0:2])}.json", "r") as batter_props_file:
        print(f"batterprops{"_".join(todaysDate.split('/')[0:2])}.json")

        # Convert the JSON string to a Python dictionary
        data = json.load(batter_props_file)

        oc = 1
        os = 4
        try:
#                [print(a) for a in enumerate(data['eventGroup']['offerCategories'][0]['offerSubcategoryDescriptors'][5])]
                batter_props = data['eventGroup']['offerCategories'][oc]['offerSubcategoryDescriptors'][os]['offerSubcategory']['offers']
        except KeyError:
                try:
                        batter_props = data['eventGroup']['offerCategories'][oc]['offerSubcategoryDescriptors'][os + 1]['offerSubcategory']['offers']
                except KeyError:
                        try:
                                batter_props = data['eventGroup']['offerCategories'][oc]['offerSubcategoryDescriptors'][os + 2]['offerSubcategory']['offers']
                        except KeyError:
                                batter_props = data['eventGroup']['offerCategories'][oc]['offerSubcategoryDescriptors'][os + 3]['offerSubcategory']['offers']

        time_to_use_for_orig = ''
        
        #[print(elem) for elem in enumerate(batter_props)]

        #print(len(batter_props))
        for offer in batter_props:
                for elem in offer:
                        try:
                                base = elem['outcomes']
#                                print(base['line'])
#                                print(len(base))
                                
                                if len(base) < 2:
                                        continue
                                line = base[1]['line']
#                                print(base[1].keys())
                                odds = [base[0]['oddsAmerican'], base[1]['oddsAmerican']]
                        #        print(odds)
                                player_name = base[1]['participant'].strip()
                                print(player_name)

                                indices = None
                                if player_name in batters:
                                        indices = [i for i, x in enumerate(batters) if x == player_name]
#                                        print(player_name,indices)
                                print(player_name,indices)
                                if indices is not None:
                                        if player_name in players_hit_lines:
                                                players_hit_lines[f"{player_name} @ {game_times[indices[-1]//18]}"] = line,odds
                                        else:
                                                players_hit_lines[f"{player_name} @ {game_times[indices[0]//18]}"] = line,odds

                        except:
                                print(base)
#                                print(base[1])
                                print(len(base))
                                print(elem)
                                print(player_name)
                                raise SyntaxError

#print(players_hit_lines)
print(len(players_hit_lines))
[print(elem,players_hit_lines[elem]) for elem in players_hit_lines]

# %%
print(batters[1])

# %%
#players_hit_lines = {} #commented out so that anything we got earlier and is in json will be default and msot recnet stuff will be used for others

'''
from selenium import webdriver
from bs4 import BeautifulSoup

import json

url = "https://sportsbook-nash.draftkings.com/sites/US-SB/api/v5/eventgroups/84240/categories/743/subcategories/6719"

# Set up the Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')

# Initialize the Chrome driver
driver = webdriver.Chrome(options=chrome_options) 

# Fetch the webpage
driver.get(url)

# Get the HTML content
html = driver.page_source

# Close the driver
driver.quit()


start_index = html.find("{")
end_index = html.rfind("}") + 1
json_str = html[start_index:end_index]

# Convert the JSON string to a Python dictionary
data = json.loads(json_str)


batter_props = data['eventGroup']['offerCategories'][2]['offerSubcategoryDescriptors'][3]['offerSubcategory']['offers']

#[print(elem) for elem in enumerate(batter_props)]

#print(len(batter_props))
for offer in batter_props:
    for elem in offer:
        base = elem['outcomes']
        line = base[1]['line']
        odds = [base[0]['oddsAmerican'], base[1]['oddsAmerican']]
#        print(odds)
        player_name = base[1]['participant']
        players_hit_lines[player_name] = line,odds

#print(players_hit_lines)
print(len(players_hit_lines))

with open(f"batterprops{"_".join(todaysDate.split('/')[0:2])}.json", "a") as fileWithLines:
        #Hardcoded line will manually change since overwhelming majority of people line .5
        fileWithLines.write(str(data))
'''


# %% [markdown]
# Model

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
    
    #ops = player_data.loc[:, 'OPS'].to_numpy()
    avg = player_data.loc[:, 'AVG'].to_numpy()
    #slg = player_data.loc[:, 'SLG'].to_numpy()
    #ab = player_data.loc[:, 'AB'].to_numpy()
    h = player_data.loc[:, 'H'].to_numpy()
    #so = player_data.loc[:, 'SO'].to_numpy()
    pa = player_data.loc[:, 'PA'].to_numpy()
    opponent_teams = player_data.loc[:, 'OPP'].to_numpy()
    
#    mean_ops = np.mean(ops)
#    std_dev_ops = np.std(ops)
    mean_avg = np.mean(avg)
    std_dev_avg = np.std(avg)
    '''
    mean_slg = np.mean(slg)
    std_dev_slg = np.std(slg)
    mean_ab = np.mean(ab)
    std_dev_ab = np.std(ab)
    '''
    mean_h = np.mean(h)
    std_dev_h = np.std(h)
#    mean_so = np.mean(so)
#    std_dev_so = np.std(so)
    mean_pa = np.mean(pa)
    std_dev_pa = np.std(pa)

    # Number of simulations

    num_simulations = 10000
    actual_range = int(num_simulations/ len(h))
    # Arrays to store simulated results
    simulated_hits = np.zeros((actual_range, len(h)))
    # Calculate R-squared values for each variable
    r2_values = []
#    for variable in [ops, avg, slg, ab, h, so, pa]:
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
#        simulated_slg = np.array(slg) + np.random.normal(mean_slg, std_dev_slg, len(slg))
        simulated_avg = np.array(avg) + np.random.normal(mean_avg, std_dev_avg, len(avg))
#        simulated_ops = np.array(ops) + np.random.normal(mean_ops, std_dev_ops, len(ops))
#        simulated_ab = np.array(ab) + np.random.normal(mean_ab, std_dev_ab, len(ab))
        simulated_h = np.array(h) + np.random.normal(mean_h, std_dev_h, len(h))
#        simulated_so = np.array(so) + np.random.normal(mean_so, std_dev_so, len(so))
        simulated_pa = np.array(pa) + np.random.normal(mean_pa, std_dev_pa,len(pa))
        # Perform linear regression for points with custom weights
#        X = np.array([simulated_slg, simulated_avg, simulated_ops, simulated_ab, simulated_h, simulated_so, simulated_pa]).T
        X = np.array([simulated_avg, simulated_h, simulated_pa]).T
        linear_model = LinearRegression()

#        print(X)
#        print(weights)
#        print("Next")

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


    '''
    #Plot the distribution of weights
    if demo_mode == False:
        plt.hist(simulated_points.flatten(), bins=30, color='blue', alpha=0.7)
        plt.title(name + ' Distribution of Predicted Points')
        plt.xlabel('Predicted Points')
        plt.ylabel('Frequency')
    '''
    
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
#res = model('Matt Vierling','7/30/2024','CLE',0.5,pitchers_stats,league_average_h_per_9,demo_mode = True)
#print(res)

# %%
print(len(batters))
#print(teams)

# %%
#teams[12:] = teams[14:]
[print(elem) for elem in enumerate(teams)]

# %%
print(len(teams))
#teams = teams[:-2]

# %%
[print(i,elem) for i,elem in enumerate(batters)]


# %%
#pred = model("Rafael Devers", todaysDate, teams[13], 0.5, pitchers_stats, league_average_h_per_9,True)
#print(pred)
#pred = model("Paul Goldschmidt", todaysDate, teams[20], 0.5, pitchers_stats, league_average_h_per_9,True)
#print(pred)
#print(teams[13])
#print(teams[20])


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


# %%
#print(teams[23])

# %%
#predsForEachGame = [preds[9 * i: (9 * i) + 9] for i in range(0,len(teams))]
[print(elem) for elem in preds]

# %%
'''
for i,elem in enumerate(batters):

    if elem != "SKIP":
        curTeamIndex = i // 9
        oppTeamIndex = 0
        
        if elem.lower() == "luis garcía" or elem.lower() == "luis garcia":
            elem += " jr"

        if get_player_data(elem,todaysDate) is not None:
            url = 'https://www.statmuse.com/mlb/ask?q=' + elem.lower().replace("'", "").replace(' ', '+') + '+stats+' + todaysDate.replace('/', '%2F')
            # get page content
            response = requests.get(url)
            print(url)

            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                stats = soup.find_all('td', class_="text-right")
    #            print(stats)
    #            print(url)
    #            print(hits)
            
    #            [print(i,elem) for i,elem in enumerate(soup[:6])]
                if stats != []:
                    hits = stats[3].text.strip()
                    abs = int(stats[1].text.strip())

                    #Default line of .5
                    try:
                        line = players_hit_lines[elem][0]

                    except KeyError:
                        line = 0.5

    #                print(abs)
                    if curTeamIndex % 2 == 0:   
                        oppTeamIndex = curTeamIndex + 1
                    else:
                        oppTeamIndex = curTeamIndex - 1
                    print(line)
                    try:
                        (p,o,u) = preds_dict[elem] 
                        o_or_u = max(o,u)
                    except KeyError:
                        continue

    #                print(hits[1].text.strip())
                    if abs < 3:

                        print(f"{elem}: {abs}, {url}")
                        abs = f"#only {abs} abs"
                    else:
                        abs = "" 
                        with open(f"mlb_{"_".join(todaysDate.split('/')[0:2])}_3ab.csv", "a") as file3ab:
                            #Hardcoded line will manually change since overwhelming majority of people line .5
                            file3ab.writelines(f"{elem},{todaysDate},{teams[curTeamIndex]},{teams[oppTeamIndex]},{hits},{line},{p},{o_or_u},{abs}\n")#[ACTUAL RESULT]

                
                    with open(f"mlb_{"_".join(todaysDate.split('/')[0:2])}.csv", "a") as file:
                        #Hardcoded line will manually change since overwhelming majority of people line .5
                        file.writelines(f"{elem},{todaysDate},{teams[curTeamIndex]},{teams[oppTeamIndex]},{hits},{line},{p},{o_or_u},{abs}\n")#[ACTUAL RESULT]
            
        
        if curTeamIndex % 2 == 0:   
            oppTeamIndex = curTeamIndex + 1
        else:
            oppTeamIndex = curTeamIndex - 1
        with open("mlb_5_24.csv", "a") as file:
            #Hardcoded line will manually change since overwhelming majority of people line .5
            file.writelines(f"{elem},5/24/2024,{teams[curTeamIndex]},{teams[oppTeamIndex]},ACTUAL_RESULT,0.5,\n")#[ACTUAL RESULT]
    '''

# %%
preds = sorted(preds,key=lambda x :x[1][0],reverse=True)
[print(elem) for elem in preds]
include_in_image = preds[:10]

# %%
#pred = model('Anthony Santander', todaysDate, teams[29], 0.5, pitchers_stats, league_average_h_per_9,True)
#print(pred)

# %%
preds = sorted(preds,key=lambda x :(x[1][1],x[1][0]),reverse=True)
[print(elem) for elem in preds]

#model('Alec Burleson')

# %%
for elem in preds[:10]:
    if elem not in include_in_image:
        include_in_image.append(elem)

for elem in preds[10:]:
    if len(include_in_image) < 20 or elem[1][1] == 100.0:
        if elem not in include_in_image:
            include_in_image.append(elem)

#include_in_image = [include_in_image.append(elem) for elem in preds[:10] if elem not in include_in_image]
print(include_in_image)
print(len(include_in_image))
#include_in_image = preds

# %%
[print(include_in_image[i]) for i in range(len(include_in_image))] 

# %%

formatted_data = [
    {'Name': item[0], 'Team': item[2],'Time':item[3], 'Prediction': round(item[1][0],2), 'Odds Over': round(item[1][1],2),  'Line': item[4], 'Odds': item[5]}
    for item in include_in_image
]
#formatted_data = formatted_data[:10] + [{'Name': 'Remaining', 'Team': 'Top', 'Prediction': 'Over', 'Odds Over': 'Batters', 'Line': 'By', 'Odds':'%'}] + formatted_data[10:]
#print(type(formatted_data))

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

ax.set_title(f'Hits Chart {todaysDate}', fontsize=14)
#ax.set_title(f'Hits Chart {datetime.now()}', fontsize=14)
#plt.tight_layout()


# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/table_image_hits.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
#print(df.to_string(index=False, header=False))
df = df.drop('Time',axis=1)
df.insert(0,'Date',todaysDate)
print('\n'.join(df.apply(lambda row: ' '.join(map(str, row)), axis=1)))


# %%
[print(elem) for elem in preds]

# %% [markdown]
# 

# %%
'''
preds = []
preds_dict = {}


for i,elem in enumerate(batters):
    if elem != "SKIP":
        curTeamIndex = i // 9
        oppTeamIndex = 0
        if curTeamIndex % 2 == 0:
            oppTeamIndex = curTeamIndex + 1
        else:
            oppTeamIndex = curTeamIndex - 1

        if elem.lower() == "luis garcía" or elem.lower() == "luis garcia" or elem.lower() == "fernando tatis":
            elem += " Jr."
   
        pred = model(elem, todaysDate, teams[oppTeamIndex], 0.5, pitchers_stats, league_average_h_per_9,True)
        if pred is not None:

            preds_dict[elem] = pred
            pred = (elem,pred,teams[curTeamIndex],game_times[curTeamIndex//2])
            preds.append(pred)
            '''


# %%
preds = preds_to_record_a_hit
#stl_game = preds[9 : 24]

#print(stl_game)


sorted_preds = sorted(preds,key=lambda x :(x[1][1],x[1][0]),reverse=True)
formatted_data = [
    {'Name': item[0], 'Team': item[2],'Time':item[3], 'Prediction': round(item[1][0],2), 'Odds 1+ hits': round(item[1][1],2)}
    for item in sorted_preds[:20]
]
#formatted_data = formatted_data[:10] + [{'Name': 'Remaining', 'Team': 'Top', 'Prediction': 'Over', 'Odds Over': 'Batters', 'Line': 'By', 'Odds':'%'}] + formatted_data[10:]
#print(type(formatted_data))

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

ax.set_title(f'Most likely to record a hit {todaysDate}', fontsize=14)
#ax.set_title(f'Most likely to record a hit {datetime.now()}', fontsize=14)
#plt.tight_layout()


# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/table_image_to_Record_a_hit.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
[print(elem) for elem in sorted_preds]

# %%
df = df.drop(['Time'],axis=1)
df.insert(0,'Date',todaysDate)
print('\n'.join(df.apply(lambda row: ' '.join(map(str, row)), axis=1)))


# %%

#print(stl_game)


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

plt.savefig('images/table_image_to_not_Record_a_hit.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
df = df.drop(['Time'],axis=1)
df.insert(0,'Date',todaysDate)
print('\n'.join(df.apply(lambda row: ' '.join(map(str, row)), axis=1)))


# %%
'''
formatted_data = [
    {'Name': item[0], 'Team': item[2], 'Prediction': round(item[1][0],2), 'Odds 1+ hits': round(item[1][1],2)}
    for item in sorted(stl_game,key=lambda x :(x[1][1],x[1][0]),reverse=True)
]
#formatted_data = formatted_data[:10] + [{'Name': 'Remaining', 'Team': 'Top', 'Prediction': 'Over', 'Odds Over': 'Batters', 'Line': 'By', 'Odds':'%'}] + formatted_data[10:]
#print(type(formatted_data))

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a table plot
ax = plt.gca()
ax.axis('off')
tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

#ax.set_title(f'Most likely to record a hit {todaysDate}', fontsize=14)
ax.set_title(f'Most likely to record a hit in Cards game {datetime.now()}', fontsize=14)
#plt.tight_layout() 


# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.savefig('images/stlgame.png', bbox_inches='tight', dpi=300)
plt.show()
'''


# %%
#print(df.to_string(index=False, header=False))
#df.insert(0,'Date',todaysDate)
#print('\n'.join(df.apply(lambda row: ' '.join(map(str, row)), axis=1)))


# %%
[print(elem) for elem in preds]

# %%
[print(elem) for elem in sorted(preds_to_record_a_hit,key=lambda x :x[1][1],reverse=True)]


# %% [markdown]
# Demo

# %%
'''
#Running model for 100 players, projecting for their points scored for the 4/12/2024 games
#Extracting info from input file
#demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}.csv")
demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}_3ab.csv")

print(len(demo_input))


BCE_losses = []
correct_over_under = 0
ses = []


overs = 0
correct_overs = 0
unders = 0
correct_unders = 0

preds = []

valid_over_under = 0 #Count for non-push over/under outcomes

#projection = 30 #Threshold for over/under bets

for index, row in demo_input.iterrows():
    name = row[0]
    date = row[1]
    opponent = row[3]
    actual_score = row[4]
    prop = row[5] #Threshold for over/under bets

    try:
        prop = float(prop)
        actual_score = int(actual_score)
    except:
        print(f"ERROR: {name}")
        continue
        
    #Running model to get point prediction w/ over under chances
    res = model(name.replace("'", ""), date, opponent, prop, pitchers_stats,league_average_h_per_9, demo_mode=True)
    if res is not None:
        (prediction, over, under) = res
        pred = (name,pred)
        preds.append(pred)

    else:
        print(f"{name}: NONE")
        continue

    #Calculating point prediction error using model prediction and actual points scored value
    ses = ses + [(actual_score - prediction)*(actual_score - prediction)]

    #Logging if over/under prediction from model was correct and calculating BCE Loss
    #Skipping this metric for players that yielded a push rather than an over or under
    if actual_score != prop:
        valid_over_under = valid_over_under + 1

        if prediction > prop:
            overs += 1

            #over
            over_under_result = 1
        elif prediction < prop:
            unders += 1
            #under
            over_under_result = 0

        if (((over_under_result == 1) & (prediction > actual_score)) or ((over_under_result == 0) & (prediction < actual_score))):
            correct_over_under = correct_over_under + 1
            if prediction > actual_score:
                correct_overs += 1
            else:
                correct_unders += 1
        BCE_losses = BCE_losses + [(over_under_result*(np.log(over/100)) + (1 - over_under_result)*(np.log(1 - over/100)))]

#Reporting MSE, BCE Loss, Over/Under Accuracy
mse = np.sum(ses)/demo_input.shape[0]
BCE_loss = np.sum(BCE_losses)/-demo_input.shape[0]
over_under_accuracy = correct_over_under/valid_over_under

print(mse)
print(BCE_loss)
print(over_under_accuracy)
'''


# %%
'''
#Modified version
demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}.csv")
#demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}_3ab.csv")

print(len(demo_input))


correct_over_under = 0
ses = []


overs = 0
correct_overs = 0
unders = 0
correct_unders = 0

preds = []

valid_over_under = 0 #Count for non-push over/under outcomes

#projection = 30 #Threshold for over/under bets

for index, row in demo_input.iterrows():

    name = row[0]
    date = row[1]
    opponent = row[3]
    actual_score = row[4]
    prop = row[5] #Threshold for over/under bets
    prediction = row[6]
    confidence = row[7]
    try:
        prop = float(prop)
        actual_score = int(actual_score)
    except:
        print(f"ERROR: {name}")
        continue
        
    if actual_score != prop:
        valid_over_under = valid_over_under + 1

        if prediction > prop:
            overs += 1
            over_under_result = 1
        elif prediction < prop:
            unders += 1
            #under
            over_under_result = 0

        if (((over_under_result == 1) & (prediction > actual_score)) or ((over_under_result == 0) & (prediction < actual_score))):
            correct_over_under = correct_over_under + 1
            if prediction > actual_score:
                correct_overs += 1
            else:
                correct_unders += 1

over_under_accuracy = correct_over_under/valid_over_under

print(over_under_accuracy)

print(f"Unders: {correct_unders}/{unders}({round(float(100 * correct_unders/unders),2)}%)")
print(f"Overs: {correct_overs}/{overs}({round(float(100 * correct_overs/overs),2)}%)")
print(f"Overall: {correct_unders + correct_overs}/{unders + overs}({round(100 * float((correct_unders) + float(correct_overs))/(float(unders) + float(overs)),2)}%)")
'''


# %%
'''
print(correct_unders)
print(unders)
print(correct_overs)
print(overs)
print(unders+overs)
'''

# %%
'''
print(f"Unders: {correct_unders}/{unders}({round(float(100 * correct_unders/unders),2)}%)")
print(f"Overs: {correct_overs}/{overs}({round(float(100 * correct_overs/overs),2)}%)")
print(f"Overall: {correct_unders + correct_overs}/{unders + overs}({round(100 * float((correct_unders) + float(correct_overs))/(float(unders) + float(overs)),2)}%)")
'''

# %%
'''
#this code only counts for players who have more than a 10% edge against the implied probability

demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}.csv")
#BCE_losses = []
#ses = []

thresholds =[67.38,100]

correct_over_under = [0] * len(thresholds)
overs = [0] * len(thresholds)
correct_overs = [0] * len(thresholds)
unders = [0] * len(thresholds)
correct_unders = [0] * len(thresholds) 
valid_over_under = [0] * len(thresholds) 
over_under_accuracy = [0] * len(thresholds) 

for index, row in demo_input.iterrows():
    name = row[0]
    date = row[1]
    opponent = row[3]
    actual_score = row[4]
    prop = row[5] #Threshold for over/under bets

    try:
        actual_score = int(actual_score)
        prop = float(prop)
    except:
        print(f"Exception occured: {name},{actual_score},{prop}")
        continue

    # Running model to get point prediction w/ over under chances
    res = model(name.replace("'", ""), date, opponent, prop, pitchers_stats,league_average_h_per_9, demo_mode=True)
    if res is not None:
        (prediction, over, under) = res
        print(f"name: {name},  pred: {prediction}")
    else:
        print(name)
        continue

#    print(f"{name}--3")
    # Calculating point prediction error using model prediction and actual points scored value
#    ses = ses + [(actual_score - prediction)*(actual_score - prediction)]

#    percent_dif = prediction / prop

    # Logging if over/under prediction from model was correct and calculating BCE Loss
    # Skipping this metric for players that yielded a push rather than an over or under

    for i,thold in enumerate(thresholds):
        if ((actual_score != prop) and ((over >= thold) or (under >= thold))):
            valid_over_under[i] += 1

            if actual_score != prop:
                if prediction > prop:
                    overs[i] += 1

                    # over
                    over_under_result = 1
                elif prediction < prop:
                    unders[i] += 1
                    # under
                    over_under_result = 0

                if (((over_under_result == 1) & (prediction > actual_score)) or ((over_under_result == 0) & (prediction < actual_score))):
                    correct_over_under[i] += 1
                    
                    if prediction > actual_score:
                        correct_overs[i] += 1
                    else:
                        correct_unders[i] += 1
            else:
                print(f"{name}, pred: {prediction},act:{actual_score}")

#                BCE_losses = BCE_losses + [(over_under_result*(np.log(over/100)) + (1 - over_under_result)*(np.log(1 - over/100)))]
    

# Reporting MSE, BCE Loss, Over/Under Accuracy
#mse = np.sum(ses) / demo_input.shape[0]
#BCE_loss = np.sum(BCE_losses) / -demo_input.shape[0]
'''

# %%
'''
#this code only counts for players who have more than a 10% edge against the implied probability MODIFIED

demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}.csv")

thresholds =[67.38,100]

correct_over_under = [0] * len(thresholds)
overs = [0] * len(thresholds)
correct_overs = [0] * len(thresholds)
unders = [0] * len(thresholds)
correct_unders = [0] * len(thresholds) 
valid_over_under = [0] * len(thresholds) 
over_under_accuracy = [0] * len(thresholds) 

for index, row in demo_input.iterrows():
    name = row[0]
    date = row[1]
    opponent = row[3]
    actual_score = row[4]
    prop = row[5] #Threshold for over/under bets
    prediction = row[6]
    confidence = row[7]
    try:
        actual_score = int(actual_score)
        prop = float(prop)
    except:
        print(f"Exception occured: {name},{actual_score},{prop}")
        continue

    for i,thold in enumerate(thresholds):
        if ((actual_score != prop) and confidence >= thold):
            valid_over_under[i] += 1

            if actual_score != prop :
                if prediction > prop:
                    overs[i] += 1

                    # over
                    over_under_result = 1
                elif prediction < prop:
                    unders[i] += 1
                    # under
                    over_under_result = 0

                if (((over_under_result == 1) & (prediction > actual_score)) or ((over_under_result == 0) & (prediction < actual_score))):
                    correct_over_under[i] += 1
                    
                    if prediction > actual_score:
                        correct_overs[i] += 1
                    else:
                        correct_unders[i] += 1
            else:
                print(f"{name}, pred: {prediction},act:{actual_score}")
'''

# %%
'''
for i,thold in enumerate(thresholds):

    over_under_accuracy[i] = correct_over_under[i] / valid_over_under[i]
#        print(mse)
#        print(BCE_loss)
    print(f"Threshold: {round(thold,2)}%")
#    print(over_under_accuracy[i])
#    print(valid_over_under[i])
#    print(correct_over_under[i])
    print(f"Unders: {correct_unders[i]}/{unders[i]}({round(float(100 * correct_unders[i]/unders[i]),2)}%)")
    print(f"Overs: {correct_overs[i]}/{overs[i]}({round(float(100 * correct_overs[i]/overs[i]),2)}%)")
    print(f"Overall: {correct_unders[i] + correct_overs[i]}/{unders[i] + overs[i]}({round(100 * float((correct_unders[i]) + float(correct_overs[i]))/(float(unders[i]) + float(overs[i])),2)}%)")
    print()
'''

# %%
'''
print(correct_unders)
print(unders)
print(correct_overs)
print(overs)
[print(elem) for elem in enumerate(thresholds)]
'''

# %%
'''
#Saving results from demo in CSVs
demo_ses = pd.DataFrame(ses, columns=['ses'])
demo_ses.to_csv("demo_mlb_hits_ses.csv")

demo_bce = pd.DataFrame(BCE_losses, columns=['bces'])
demo_bce.to_csv("demo_mlb_hits_bce.csv")

demo_stats = pd.DataFrame([[mse, BCE_loss, over_under_accuracy]], columns=['mse', 'bce_loss', 'over_under_acc'])
demo_stats.to_csv("demo_mlb_hits_stats.csv")
'''

# %%
'''
#Printing stats resulting from demo
mse_output = pd.read_csv("outputs\demo_stats.csv").loc[0, 'mse']
print(f"Mean Squared Error of Score Prediction: {mse_output}")

BCE_loss_output = pd.read_csv("outputs\demo_stats.csv").loc[0, 'bce_loss']
print(f"Binary Cross Entropy Loss for Over/Under Probabilities: {BCE_loss_output}")

over_under_accuracy_output = pd.read_csv("outputs\demo_stats.csv").loc[0, 'over_under_acc']
print(f"Over/Under Accuracy:{over_under_accuracy_output*100}%")
'''

# %%
'''
#Graphing BCE losses for individual players
BCE_losses_graphing = pd.read_csv("outputs\demo_bce.csv")
plt.hist(- BCE_losses_graphing.loc[:,'bces'], bins=10, color='blue')
plt.title('Binary Cross Entropy Losses (Individual Players)')
plt.xlabel('Binary Cross Entropy Loss')
plt.ylabel('Frequency')
'''

# %%
'''
#Graphing SEs for individual players
ses_graphing = pd.read_csv("outputs\demo_ses.csv")
plt.hist(ses_graphing.loc[:,'ses'], bins=10, color='blue')
plt.title('Squared Errors (Individual Players)')
plt.xlabel('Squared Error')
plt.ylabel('Frequency')
'''
