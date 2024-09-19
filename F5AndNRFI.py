# %%
import pandas as pd 
import requests
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
#import pytz

import warnings
warnings.filterwarnings('ignore')

headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}

# %%
#todaysDate = '9/18/2024'
#yesterdaysDate = '9/17/2024'
todaysDate = datetime.today().strftime('%m/%d/%Y')
yesterdaysDate = (datetime.now() - timedelta(1)).strftime('%m/%d/%Y')

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

# %%
# Function to get pitcher data and return mean ERA, WHIP, and K/9
def get_pitcher_data(name):
    
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
        except:
            url = 'https://www.statmuse.com/mlb/ask/' + name.lower().replace(' ', '%20') + '%20stats%20last%2010%20games%20including%20whip'
            response = requests.get(url)
            try:
                tables = pd.read_html(response.text)
                df = tables[0].head(1)
                df = df.filter(items=["NAME","WHIP"])
                return {"Name": name,"whip": mean_values["WHIP"]}
            except:
                print(f"help-p: {url}")
                print(response.text)
                return {"Name": name,"whip": 1.313}

# %%
# Function to get batter data and return mean OBP and SLG
#def get_batter_data(name, date):
def get_batter_data(name):
    
    if name == 'Luis Garcia':
        name += ' jr'   
    elif name == 'Will Smith':
        name += ' Dodgers'
    elif name == 'Enrique Hernandez':
        name = 'Kike Hernandez'

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
            #except:
            except Exception as e:
                print(f"Error parsing HTML on second try: {e}")
                print(f"Help: {url}")
#            print(f"help: {url}")
#                df = pd.DataFrame([[f'{name} no stats found so 0.0','0.0']],columns=["Name","avg"])                
#                return df
                return None

        df = tables[0].head(25)
                
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
                    print(f"name:{name}, {df['AVG']},{df}")
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

   pitchers = [get_pitcher_data(elem) for elem in pitchers]
    
   pitchers[0]['whip'] = .9
   pitchers[14]['whip'] = 1.45
   pitchers[15]['whip'] = 1.7
   pitchers[24]['whip'] = 1.35


   pitchers_for_1st = pitchers[:] 
   pitchers_for_1st[4]['whip'] = 1.4
   batters = soup.find_all('li',class_ = 'lineup__player')
   batters = [elem.find('a').get('title') for elem in batters]

   batters = [get_batter_data(elem) for elem in batters]

   teams = soup.find_all('div',class_= 'lineup__abbr')
   teams = [elem.text for elem in teams]
#   teams[8:] = teams[10:]

   game_times = soup.find_all('div',class_="lineup__time")
   game_times = [elem.text for elem in game_times][:-2]
#   game_times[4:] = game_times[5:]

   confirmedOrExpected = soup.find_all('li',class_="lineup__status")
   confirmedOrExpected = [elem.text.strip().split()[0][0] for elem in confirmedOrExpected]


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

odds_dict_nrfi = {awayTeams[i]: odds[2 * i: 2 * i + 2] for i in range(len(awayTeams))}
print(odds_dict_nrfi)

#indexForOdds = [index for index,elem in enumerate(awayTeams) if elem.startswith(awayTeam.split()[-1])]

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

            pitcher_whip = pitcher_stats['whip']
            
            outcome = simulate_at_bat(batter_avg,batter_1b_prob,batter_2b_prob,batter_3b_prob,batter_HR_prob, pitcher_whip)
        except Exception as e:
            hp = max(min(.700, batter_avg + (pitcher_whip-1.32) * .1),0)
            wp = max((pitcher_whip - hp)* .1 * batter_avg/.300,0)

            hp = max(batter_avg + (pitcher_whip-1.32) * .1,0)
            wp = max((pitcher_whip - hp)* .1 * batter_avg/.300,0)#hotter batters more likely to be walked

            print(f"Exception: {batter_stats[batter_index]}")
            print(hp)
            print(wp)
            print(1 - (hp + wp))
            print(e)
            raise SyntaxError


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
    return runs_scored,batter_index

# %%
def simulate_first_five_innings(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats):

    away_runs_total = 0
    home_runs_total = 0

    away_bttp = 0
    home_bttp = 0
    for i in range(5):    
        away_runs,away_bttp = simulate_inning(away_batter_stats, home_pitcher_stats,away_bttp)
        home_runs,home_bttp = simulate_inning(home_batter_stats, away_pitcher_stats,home_bttp)
        
        away_runs_total += away_runs
        home_runs_total += home_runs

#        print(f"End of inning {i}, score {away_runs_total}-{home_runs_total}, next batters {away_bttp},{home_bttp}")
    return away_runs_total,home_runs_total

# %%
def simulate_first_inning(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats):
    away_runs,away_bttp = simulate_inning(away_batter_stats, home_pitcher_stats)
    home_runs,home_bttp = simulate_inning(home_batter_stats, away_pitcher_stats)
    return away_runs,home_runs,

# %%
NUM_SIMULATIONS = 10000

preds = {}
pred_games = []
nrfi_preds_with_implied_odds = []


for i in range(len(game_times)): 

    away_batter_stats = batters[18 * i: 18 * i + 9]
    home_batter_stats = batters[18 * i + 9: 18 * i + 18]

    away_batter_stats = [away_batter_stats[i] for i in range(len(away_batter_stats)) if away_batter_stats[i] is not None]
    home_batter_stats = [home_batter_stats[i] for i in range(len(home_batter_stats)) if home_batter_stats[i] is not None]
 
    try:
        away_pitcher_stats = pitchers_for_1st[2 * i]
        home_pitcher_stats = pitchers_for_1st[2 * i + 1]

    except:
        print("huh")
        continue

    runs_first_inning = 0
    total_away_runs = 0 
    total_home_runs = 0

    runs_first_inning_away = 0
    runs_first_inning_home = 0


    for _ in range(NUM_SIMULATIONS):
        away_runs,home_runs = simulate_first_inning(away_batter_stats, away_pitcher_stats, home_batter_stats, home_pitcher_stats)
        total_away_runs += away_runs
        total_home_runs += home_runs
        if away_runs + home_runs > 0:
            runs_first_inning += 1
            if away_runs > 0:
                runs_first_inning_away += 1 
            if home_runs > 0:
                runs_first_inning_home += 1 


    # Calculate probability
    probability_run_first_inning = float(runs_first_inning / NUM_SIMULATIONS)
    average_away_runs = float(total_away_runs / NUM_SIMULATIONS)
    average_home_runs = float(total_home_runs / NUM_SIMULATIONS)
    average_total_runs = average_away_runs + average_home_runs
    prob_away_yrfi = float(runs_first_inning_away / NUM_SIMULATIONS)
    prob_home_yrfi = float(runs_first_inning_home / NUM_SIMULATIONS)


    preds[teams[2 * i]] = average_away_runs
    preds[teams[2 * i + 1]] = average_home_runs
    

    c_or_e = confirmedOrExpected[2 * i]
    c_or_e = 'C' if c_or_e == 'C' and  c_or_e == confirmedOrExpected[2 * i + 1] else 'E'

    print(f"{teams[2 * i + 1]} pitching L30 whip({confirmedOrExpected[2 * i + 1]}): {home_pitcher_stats['whip']}")
    print(f"{teams[2 * i]} batting({confirmedOrExpected[2 * i]}): {away_batter_stats[0]['avg']},{away_batter_stats[1]['avg']},{away_batter_stats[2]['avg']},{away_batter_stats[3]['avg']},{away_batter_stats[4]['avg']},{away_batter_stats[5]['avg']},{away_batter_stats[6]['avg']}")
    
    print(f"{teams[2 * i]} pitching L30 whip({confirmedOrExpected[2 * i]}): {away_pitcher_stats['whip']}")
    print(f"{teams[2 * i + 1]} batting({confirmedOrExpected[2 * i + 1]}): {home_batter_stats[0]['avg']},{home_batter_stats[1]['avg']},{home_batter_stats[2]['avg']},{home_batter_stats[3]['avg']},{home_batter_stats[4]['avg']},{home_batter_stats[5]['avg']},{home_batter_stats[6]['avg']}")


    if teams[2 * i] in odds_dict_nrfi and len(odds_dict_nrfi[teams[2 * i]]) > 1:
        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,odds_dict_nrfi[teams[2 * i]][probability_run_first_inning < .5],prob_away_yrfi,prob_home_yrfi,c_or_e))
        nrfi_preds_with_implied_odds.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,odds_dict_nrfi[teams[2 * i]][probability_run_first_inning < .5],c_or_e))
    else:
        pred_games.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,"N/A",prob_away_yrfi,prob_home_yrfi,c_or_e))
        nrfi_preds_with_implied_odds.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],probability_run_first_inning,"N/A",c_or_e))
        
    print()


# %%
[print(elem) for elem in pred_games]

# %% [markdown]
# 

# %%
pred_games = sorted(pred_games,key=lambda x :x[2],reverse=True)


pick_implied_odds = [implied_odds(elem[2]) if elem[2] > 0.5 else implied_odds(1 - elem[2]) for elem in pred_games]

formatted_data = [
    {'Game': item[0],'Time': item[1], 'Prob YRFI': item[2],'NRFI/YRFI line': item[3], 'Imp. Odds (Pick)': pick_implied_odds[i], 'Prob Away YRFI': item[4],'Prob Home YRFI': item[5],'C/E':item[6]}    for i,item in enumerate(pred_games)
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


#eastern_time = datetime.now(pytz.timezone('US/Eastern'))
#ax.set_title(f'NRFI/YRFI Chart {todaysDate}', fontsize=14)
ax.set_title(f'NRFI/YRFI Chart {datetime.now()}', fontsize=14)

# Adjust the table and save as an image
tbl.scale(1, 1.5)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

plt.figtext(0.5, 0.01, '+EV', wrap=True, horizontalalignment='center', fontsize=12, bbox=dict(facecolor=highlight_color, edgecolor='black'))

plt.savefig('NRFIs.png', bbox_inches='tight', dpi=300)
plt.show()

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


#print(odds_dict_f5)


# %%
NUM_SIMULATIONS = 10000

pred_games_f5 = []

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

    total_away_runs_f5 = 0 
    total_home_runs_f5 = 0

    away_wins_f5 = 0
    away_wins_f5_by_at_least2 = 0

    home_wins_f5 = 0
    home_wins_f5_by_at_least2 = 0

    ties_f5 = 0

    c_or_e = confirmedOrExpected[2 * i]
    c_or_e = 'C' if c_or_e == 'C' and  c_or_e == confirmedOrExpected[2 * i + 1] else 'E'



    for _ in range(NUM_SIMULATIONS):


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

    average_away_runs_f5 = round(float(total_away_runs_f5 / NUM_SIMULATIONS),2)
    average_home_runs_f5 = round(float(total_home_runs_f5 / NUM_SIMULATIONS),2)
    average_total_runs_f5 = round(average_home_runs_f5 + average_away_runs_f5,2)

    away_win_pct_f5 = round(away_wins_f5/NUM_SIMULATIONS  * 100,2)
    home_win_pct_f5 = round(home_wins_f5/NUM_SIMULATIONS  * 100,2)
    ties_f5_pct = round(ties_f5/NUM_SIMULATIONS * 100,2)
    

    c_or_e = confirmedOrExpected[2 * i]
    c_or_e = 'C' if c_or_e == 'C' and  c_or_e == confirmedOrExpected[2 * i + 1] else 'E'

    if teams[2 * i] in odds_dict_f5:
        pred_games_f5.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],average_total_runs_f5,average_away_runs_f5,average_home_runs_f5,away_win_pct_f5,home_win_pct_f5,ties_f5_pct,odds_dict_f5[teams[2 * i]][home_win_pct_f5 > away_win_pct_f5]))
    else:
        pred_games_f5.append((f"{teams[2 * i]} @ {teams[2 * i + 1]}",game_times[i],average_total_runs_f5,average_away_runs_f5,average_home_runs_f5,away_win_pct_f5,home_win_pct_f5,ties_f5_pct,'N/A'))
    print()

# %%

print(pred_games_f5[0])
pred_games_f5 = sorted(pred_games_f5,key=lambda x :x[1])

formatted_data = [
    {'Game': item[0],'Time': item[1],'Avg Total': item[2], 'Away Avg': item[3],'Home Avg': item[4],'Away Win%': item[5],'Home Win%': item[6],'Tie %': item[7],'ML Odds': item[8]}
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

#eastern_time = datetime.now(pytz.timezone('US/Eastern'))

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

print('try save')
plt.savefig('F5.png', bbox_inches='tight')
plt.show()


