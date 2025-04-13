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
def date_30_days_ago_str(date_str):
    # Parse the input date string to a datetime object
    date = datetime.strptime(date_str, '%m/%d/%Y')
 
    date_30_days_ago = date - timedelta(days=30)
    
    # Format the date back to 'MM/DD/YYYY'
    date_30_days_ago_str = date_30_days_ago.strftime('%m/%d/%Y')
    
    return date_30_days_ago_str

# %% [markdown]
# Data Grabbing Function

# %%
#Returns player info for past 50 games before the input date, from StatsMuse
def get_player_data(name, date):
    df = None
    thirty_days_ago = date_30_days_ago_str(date)

    url = 'https://www.statmuse.com/mlb/ask?q=' + name.lower().replace(' ', '+') + '+k%2F9+for+each+game+between+' + thirty_days_ago.replace('/', '%2F') + '+and+' + date.replace('/', '%2F')

    # get page content
    response = requests.get(url)

    print(url,response.status_code)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using Pandas read_html function
        try:
            tables = pd.read_html(response.text)
#            print(url)
            # assumes we want only the first 25 rows of table since statmuse only shows 25 rows for free
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
                    
                    # Create DataFrame manually
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
    return None

# %%
name = 'Garrett Cole'
date = '10/05/2024'
print(get_player_data(name, date))
'''
name = 'Carlos Rodon'
print(get_player_data(name, date))
'''

# %%


# %%
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
#game_times = game_times[:-1]
#confirmedOrExpected[0:] = confirmedOrExpected[1:]

print(len(game_times))

# %%
print(len(pitchers))
#batters[0:] = batters[18:] 
print(len(pitchers))
print(pitchers)

# %%
url = 'https://www.baseball-reference.com/leagues/majors/bat.shtml'
response = requests.get(url,headers=headers)

#league_average_h_per_9 = None

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('tbody')
row = table.find('tr')

so = float(row.find('td', {'data-stat': 'SO'}).text.strip())
pa = float(row.find('td', {'data-stat': 'PA'}).text.strip())

league_average_so_per_ab = so/pa
print(league_average_so_per_ab)

# %%
#print(team_strikeout_rate['COL'])

# %%
#print(get_player_data('George Kirby',todaysDate))

# %%
#print(team_strikeout_rate['MIN'][0]/league_average_so_per_ab)

# %%
pitcher_so_lines = {}
with open(f"pitcherprops/pitcherprops{"_".join(todaysDate.split('/')[0:2])}.json", "r") as pitcher_props_file:
        print(f"pitcherprops{"_".join(todaysDate.split('/')[0:2])}.json")

        # Convert the JSON string to a Python dictionary
        data = json.load(pitcher_props_file)


        pitcher_props = data['eventGroup']['offerCategories'][2]['offerSubcategoryDescriptors'][1]['offerSubcategory']['offers']

        #print(len(batter_props))
        for offer in pitcher_props:
                for elem in offer:
                        base = elem['outcomes']
                        line = base[1]['line']
                        odds = [base[0]['oddsAmerican'], base[1]['oddsAmerican']]
                        player_name = base[1]['participant'].strip()
                        pitcher_so_lines[player_name] = line,odds

#print(players_hit_lines)
print(len(pitcher_so_lines))
[print(elem,pitcher_so_lines[elem]) for elem in pitcher_so_lines]

# %%

'''
from selenium import webdriver
from bs4 import BeautifulSoup
import json

pitcher_so_lines = {}

url = "https://sportsbook-nash-usmd.draftkings.com/sites/US-MD-SB/api/v5/eventgroups/84240/categories/1031/subcategories/15221?format=json"

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


for i in range(len(data['eventGroup']['offerCategories'][2]['offerSubcategoryDescriptors'])):
        try:
                pitcher_props = data['eventGroup']['offerCategories'][2]['offerSubcategoryDescriptors'][i]['offerSubcategory']['offers']
        except:
              print(i)

for offer in pitcher_props:
    for elem in offer:
        base = elem['outcomes']
        line = base[1]['line']
        odds = [base[0]['oddsAmerican'], base[1]['oddsAmerican']]
#        print(odds)
        player_name = base[1]['participant']
        pitcher_so_lines[player_name] = line,odds

#print(players_hit_lines)
print(len(pitcher_so_lines))

#with open(f"pitcherprops{"_".join(todaysDate.split('/')[0:2])}.json", "a") as fileWithLines:
#        #Hardcoded line will manually change since overwhelming majority of people line .5
#        fileWithLines.write(str(data))


'''

# %% [markdown]
# Model

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
res = model('Gerrit Cole','10/5/2024','KC',5.5,team_strikeout_rate,league_average_so_per_ab,demo_mode = True)

print(team_strikeout_rate)
#res = model('Hunter Greene','7/11/2024','COL',8.5,team_strikeout_rate,league_average_so_per_ab,demo_mode = True)
#res = model('Tarik Skubal','7/12/2024','LAD',6.5,team_strikeout_rate,league_average_so_per_ab,demo_mode = True)
print(res)

#res = model('Tanner Houck','7/11/2024','OAK',6.5,team_strikeout_rate,league_average_so_per_ab,demo_mode = True)
#print(res)

#print(team_strikeout_rate['CIN'])
#print(league_average_so_per_ab)
#print(team_strikeout_rate['CIN'][0]/league_average_so_per_ab)


# %%
print(len(pitcher_so_lines))
#print(teams)

# %%
#teams[0:] = teams[:-2]
[print(elem) for elem in enumerate(teams)]
print(len(pitchers))

# %%

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
#            pred = (elem,pred,teams[curTeamIndex],game_times[curTeamIndex//2],'''confirmedOrExpected[curTeamIndex],'''line,odds)
            preds.append(pred)
#            print(pred)    

# %%
#predsForEachGame = [preds[9 * i: (9 * i) + 9] for i in range(0,len(teams))]
[print(elem) for elem in preds]

# %%
'''
for i,elem in enumerate(pitchers):

    if elem != "SKIP":
        curTeamIndex = i
        oppTeamIndex = 0
        
        if get_player_data(elem,todaysDate) is not None:
            url = 'https://www.statmuse.com/mlb/ask?q=' + elem.lower().replace("'", "").replace(' ', '+') + '+stats+' + todaysDate.replace('/', '%2F')
            # get page content
            response = requests.get(url)
            print(url)

            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                stats = soup.find_all('td', class_="text-right")

                if stats != []:
                    Ks = stats[15].text.strip()
                    tbf = int(stats[8].text.strip())
                    print(f"Ks: {Ks}, tbf:{tbf}")
#                    [print(elem) for elem in enumerate(stats)]

                    #Default line of .5
                    try:
                        line = pitcher_so_lines[elem][0]

                    except KeyError:
                        line = 4.5#idk whats good default

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
                    '''
'''
    #                print(hits[1].text.strip())
                    if tbf < 12:

                        print(f"{elem}: {tbf}, {url}")
                        tbf = f"#only {tbf} batters faced"
                    else:
                        tbf = "" 
                        with open(f"mlb_{"_".join(todaysDate.split('/')[0:2])}_pitchers_12tbf.csv", "a") as file12tbf:
                            file12tbf.writelines(f"{elem},{todaysDate},{teams[curTeamIndex]},{teams[oppTeamIndex]},{Ks},{line},{p},{o_or_u},{tbf}\n")#[ACTUAL RESULT]

                    '''
#                    with open(f"mlb_{"_".join(todaysDate.split('/')[0:2])}_pitchers.csv", "a") as file:
#                        file.writelines(f"{elem},{todaysDate},{teams[curTeamIndex]},{teams[oppTeamIndex]},{Ks},{line},{p},{o_or_u},{tbf}\n")#[ACTUAL RESULT]
            
'''
        '''
#        if curTeamIndex % 2 == 0:   
#            oppTeamIndex = curTeamIndex + 1
#        else:
#            oppTeamIndex = curTeamIndex - 1
#        with open("mlb_5_24.csv", "a") as file:
            #Hardcoded line will manually change since overwhelming majority of people line .5
#            file.writelines(f"{elem},5/24/2024,{teams[curTeamIndex]},{teams[oppTeamIndex]},ACTUAL_RESULT,0.5,\n")#[ACTUAL RESULT]
'''      
        '''

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
#preds = sorted(preds,key=lambda x :abs(float(x[1][0])-float(x[4])),reverse=True)
preds = sorted(preds,key=lambda x :float(x[1][0])-float(x[4]),reverse=True)
'''
formatted_data = [
    {'Name': item[0], 'Team': item[2], 'Prediction': round(item[1][0],2), 'Odds O/U': max(item[1][1],item[1][2]) , 'Line': item[4], 'Difference': round(float(item[1][0])-float(item[4]),2), 'Odds': item[5]}
    for item in preds[:-1]
]
'''
formatted_data = [
    {'Name': item[0], 'Team': item[2], 'Prediction': round(item[1][0],2), 'Line': item[4], 'Difference': round(float(item[1][0])-float(item[4]),2), 'Odds': item[5]}
    for item in preds
]

# Create a DataFrame
df = pd.DataFrame(formatted_data)

# Set up the matplotlib figure
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



#ax.set_title(f'Strikeouts Chart {todaysDate}', fontsize=14)
ax.set_title(f'Strikeouts Chart {datetime.now()}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.figtext(0.5, 0.01, 'Diff >= 1', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color, edgecolor='black'))


plt.savefig('images/table_image_ks.png', bbox_inches='tight', dpi=300)
plt.show()

# %%
#print(df.to_string(index=False, header=False))
df.insert(0,'Date',todaysDate)
print('\n'.join(df.apply(lambda row: ' '.join(map(str, row)), axis=1)))


# %% [markdown]
# Demo

# %%

'''
#Running model for 100 players, projecting for their points scored for the 4/12/2024 games
#Extracting info from input file
#demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}.csv")
demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}_pitchers.csv")

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
    res = model(name.replace("'", ""), date, opponent, prop, team_strikeout_rate,league_average_so_per_ab, demo_mode=True)
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
#demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}.csv")
demo_input = pd.read_csv(f"mlb_{"_".join(todaysDate.split('/')[0:2])}_pitchers.csv")

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
            unders += 1.png
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


