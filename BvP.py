# %%
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}

# %%
#Suppressing Warnings for better looking output if desired
import warnings
warnings.filterwarnings('ignore')

# %%
#%history

# %%
def get_pitcher_data(name):

    print(name,pitcher_ids[name])
    url_for_stats_vs_hand = f"https://www.rotowire.com/baseball/player/{name.lower().replace(' ','-')}-{pitcher_ids[name]}"

    response = requests.get(url_for_stats_vs_hand, headers=headers)
    soup = BeautifulSoup(response.text,'html.parser')

    baa_2025_vs_left = None
    baa_2025_vs_right = None

    # Iterate through all rows in the table
    rows = soup.find_all('tr')
    for row in rows:
        if row.find('b'):
            year = row.find('b').get_text()
            if year == '2025':
                span_text = row.find('span').get_text()
                baa = row.find_all('td')[1].get_text()
                if span_text == 'vs Left':
                    baa_2025_vs_left = float(baa)
                elif span_text == 'vs Right':
                    baa_2025_vs_right = float(baa)
    



    url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={pitcher_ids[name]}&stats=pitching"


    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        stats = response.json()
        try:
            whip_L30 = float(stats['gamelog']['majors']['pitching']['footer'][1]['whip']['text'])
        except:
            whip_L30 = None

    return [
    name,whip_L30,baa_2025_vs_left,baa_2025_vs_right
    ]    

# %% [markdown]
# Data Grabbing Function

# %%
def get_batter_data(name,i):
#    print(name)
    print(name,batter_ids[name],i)

    b_h = batter_handedness[i]
    
    p_h_index =  i//9
    if p_h_index % 2 == 0:
        p_h_index += 1
    else:
        p_h_index -= 1
    try:
        p_h = pitcher_handedness[p_h_index]
    except:
        print(p_h_index,pitcher_handedness[28])
        raise SyntaxError
#        p_h = None

#    print(p_h_index)

    url_for_stats_vs_hand = f'https://www.rotowire.com/baseball/tables/player-splits.php?playerID={batter_ids[name]}&position=OF&type=position'
    url_for_stats = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={batter_ids[name]}&stats=batting"
    response = requests.get(url_for_stats_vs_hand,headers=headers)
    response2 = requests.get(url_for_stats,headers=headers)
    #print(url_for_stats)
    historic_stats = response2.json()

    try:
        l_7 =  historic_stats['gamelog']['majors']['batting']['footer'][0]['avg']['text']
    except:
        l_7 = None


    try:
        avg_vs_p = historic_stats['matchup']['batting'][0]['avg']
    except:
        avg_vs_p = None
#    print(avg_vs_p,historic_stats['matchup']['batting'][0]['ab'], int(historic_stats['matchup']['batting'][0]['ab']) == 0)
    
    if avg_vs_p == ".000" and historic_stats['matchup']['batting'][0]['ab'] == "0":
        avg_vs_p = 'N/A'

    if response.status_code == 200:
        stats = response.json()
#        print(stats)

        '''
        p_h_index =  i//9
        if p_h_index % 2 == 0:
            p_h_index += 1
        else:
            p_h_index -= 1
        '''

        try:
            pitcher_stats = pitchers_stats[p_h_index] 
        except:
            print(pitchers_stats,p_h_index)
            pitcher_stats = ['',0,None,None]
            raise SyntaxError

#        print(pitcher_stats)
        if b_h == 'S':
            try:
                as_Left = float(pitcher_stats[2])
                as_Right = float(pitcher_stats[3])
                vs_Left = float(stats[0]["AVG"])
                vs_Right = float(stats[1]["AVG"])
            except:
                as_Left = 0
                as_Right = 0
                vs_Left = 0
                vs_Right = 0



            if as_Left - vs_Left > as_Right - vs_Right:
                b_h = 'L'
            else:
                b_h = 'R'
        
        stats_vs_hand = None
        if len(stats) == 2:
            if p_h == 'R':
                stats_vs_hand = float(stats[1]['AVG'])
            elif p_h == 'L':
                stats_vs_hand = float(stats[0]['AVG'])
            else:
                print(p_h,pitcher_handedness,p_h_index,pitcher_handedness[p_h_index],)
                raise ValueError("Handedness must be 'R', 'L', or 'B'")



    return {
        "Name": name,
        "handedness": b_h,
        "l7_avg": l_7,
        "avg_vs_pitcher": avg_vs_p,
        "avg_vs_pitcher_handedness": stats_vs_hand,
        "pitcher_name": pitcher_stats[0],
        "pitcher_handedness": pitcher_handedness[p_h_index],
        "pitcher_whip_l30": pitcher_stats[1],
        "pitching baa_vs_handedness": pitcher_stats[2 + (b_h == 'R')]
        }    

# %%
#print(ids)
#ids = {}
#print(get_batter_data('Steven Kwan',0))

#[print(elem) for elem in enumerate(pitcher_handedness)]

# %%
todaysDate = (datetime.now() - timedelta(hours=4)).hour

#Between 9pm and 3am look at what roto has as tommorow because it switches at 3am
if todaysDate > 21 or todaysDate < 3 :
    url = 'https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow'
else:
    url = 'https://www.rotowire.com/baseball/daily-lineups.php'
response = requests.get(url,headers=headers)

pitchers = []
pitchers_stats = []

batters = [] 
teams = []
game_times = []
batter_handedness = []
pitcher_handedness = []

#game_num = 2

if response.status_code == 200:
   soup = BeautifulSoup(response.text, 'html.parser')

   todaysDate = soup.find('main').get('data-gamedate')
   todaysDate = datetime.strptime(todaysDate, '%Y-%m-%d').strftime('%m/%d/%Y')


   pitchers = soup.find_all('div',class_='lineup__player-highlight-name')
   pitcher_handedness = [elem.find('span',class_= 'lineup__throws').text  for elem in pitchers]

   pitcher_ids = {elem.find('a').text: elem.find('a').get('href').split('-')[-1] for elem in pitchers}
   pitchers_stats = [get_pitcher_data(name.find('a').text) for name in pitchers]

   batters = soup.find_all('li',class_ = 'lineup__player')
   batter_handedness = [elem.find('span',class_= 'lineup__bats').text  for elem in batters]


   teams = soup.find_all('div',class_= 'lineup__abbr')
   teams = [elem.text for elem in teams]
#   teams[2:] = teams[4:]    
#   teams[6:] = teams[8:]    
   
   game_times = soup.find_all('div',class_="lineup__time")
   game_times = [elem.text for elem in game_times]
#   game_times[1:] = game_times[2:]    
#   game_times[3:] = game_times[4:]    

   confirmedOrExpected = soup.find_all('li',class_="lineup__status")
   confirmedOrExpected = [elem.text.strip().split()[0][0] for elem in confirmedOrExpected]
#   confirmedOrExpected = confirmedOrExpected[:2]

# %%
batter_ids = {elem.find('a').get('title') : elem.find('a').get('href').split('-')[-1] for elem in batters}
batters = [elem.find('a').get('title') for elem in batters]
stats_for_chart = [get_batter_data(name,i) for i,name in enumerate(batters)]

print(game_times)


# %%
#%history
#stats_for_chart = [get_batter_data(name,i) for i,name in enumerate(batters)]


# %%
for i in range(0,len(batters),18):

    
    formatted_data = [
    {'Batter': item['Name'],'B-Hand': item['handedness'],'L7 avg':item['l7_avg'],'BvP': item['avg_vs_pitcher'],'BvP-Hand': item['avg_vs_pitcher_handedness'],'Pitcher': item['pitcher_name'], 'P-Hand': item['pitcher_handedness'], "Whip L30": item['pitcher_whip_l30'],"BAA-B-Hand": item['pitching baa_vs_handedness']}
    for item in stats_for_chart[i:i+18]]

    # Create a DataFrame
    df = pd.DataFrame(formatted_data)

    plt.figure(figsize=(8, 6), dpi=100)

    # Create a table plot
    ax = plt.gca()
    ax.axis('off')
    tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    ax.set_title(f'Pitcher vs Batter for {teams[i//9]}({confirmedOrExpected[i//9]}) @ {teams[i//9 + 1]}({confirmedOrExpected[i//9 + 1]}) @ {game_times[i//18]} on {todaysDate}', fontsize=14)

    # Adjust the table and save as an image
    tbl.scale(2, 2)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    plt.savefig(f'Outputs/BVP-Charts/bvp_{i // 18}.png', bbox_inches='tight')
    plt.close()
