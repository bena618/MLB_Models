import requests
from bs4 import BeautifulSoup
from selenium import webdriver

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
#For Github
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}


team_abbreviations = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


url = 'https://www.rotowire.com/baseball/daily-lineups.php'
response = requests.get(url)

avgwhip = 1.313	

NRFIs = []
YRFIs = []

chrome_options = webdriver.ChromeOptions() 
chrome_options.add_argument('--headless') 
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=chrome_options) 
url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=1st-inning&subcategory=1st-inning-total-runs"
driver.get(url)
html = driver.page_source    
soup = BeautifulSoup(html, "html.parser")
teamsAndLines = soup.find_all("div", class_="sportsbook-event-accordion__wrapper expanded")

awayTeams = [elem.text.split()[1] for elem in teamsAndLines]
#awayTeams = [elem[:elem.index('at')] for elem in awayTeams]

#[print(elem) for elem in awayTeams]
teamsAndLines = [elem.text for elem in teamsAndLines]
odds = []
#YRFI,NRFI pattern
[odds.extend(line.split("0.5")[1:3]) for line in teamsAndLines]
odds = [elem[:4] for elem in odds]
#print(odds)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a')
#    links = links[476:-54]
    links = links[480:-54]
#    links = links[485:-54]
#    print(links[0:5])
#    print(len(links))
#x    [print(elem) for elem in enumerate(links[:5])]

    for index in range(0,len(links),23):
        awaystats = []
        homestats = []
        awayTeam = " ".join(links[index].text.split()[:-3])
        print(f"awayteam:{awayTeam}")


        url = f"https://www.rotowire.com{links[index+1].get('href')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        awayWhip = soup.find_all(class_="p-card__stat-value")[2].text
        if awayWhip == '0.00':
            awayWhip = avgwhip
        print(f"awayWhip: {awayWhip}")

        for i in range(index +2,index + 10,1):
#            print(links[i])
            url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={links[i].get('href').split('-')[-1]}&stats=batting"
#            print(url)
            response = json.loads(requests.get(url,headers=headers).text)
            #avg,obo,slg, and ops
            statsForPlayer2024 = response['basic']['batting']['body'][-1]
            last7DaysStats = response['gamelog']['majors']['batting']['footer'][0]
            statsVsOpposingPitcher = response['matchup']['batting'][0]
            print(last7DaysStats)

            vsLHPorRHP = None
            url = f"https://www.rotowire.com{links[i].get('href')}"
#            print(url)
            response = requests.get(url,headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            vsLHPorRHP = soup.find_all('td',class_= "hide-until-sm split-end")
            if statsVsOpposingPitcher['throws'] == 'R':
                vsLHPorRHP = vsLHPorRHP[2].text
            else:
                vsLHPorRHP = vsLHPorRHP[10].text            

            if last7DaysStats.get('ab', 0).get('text') > 10:
                if int(statsVsOpposingPitcher['ab']) > 4:
                    awaystats.append((float(last7DaysStats.get('obp', 0).get('text')) * .7 + float(statsVsOpposingPitcher['obp']) * .25 + float(vsLHPorRHP) * .05))
                else:
                    awaystats.append((float(last7DaysStats.get('obp', 0).get('text')) * .7 + float(vsLHPorRHP) * .3))
            else:
                if int(statsVsOpposingPitcher['ab']) > 4:
                    awaystats.append((float(statsForPlayer2024['obp']) * .7 + float(statsVsOpposingPitcher['obp']) * .25 + float(vsLHPorRHP) * .05))
                else:
                    awaystats.append((float(statsForPlayer2024['obp']) * .7 +  float(vsLHPorRHP) * .3))

        print(awaystats)


        homeTeam = " ".join(links[index+10].text.split()[:-3])

        url = f"https://www.rotowire.com{links[index+11].get('href')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        homeWhip = soup.find_all(class_="p-card__stat-value")[2].text
        if homeWhip == '0.00':
            homeWhip = avgwhip
        print(f"homeWhip: {homeWhip}")

        for i in range(index +12,index + 20,1):
            print(links[i])
            url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={links[i].get('href').split('-')[-1]}&stats=batting"
#            print(url)
            response = json.loads(requests.get(url,headers=headers).text)
            #avg,obo,slg, and ops
            statsForPlayer2024 = response['basic']['batting']['body'][-1]
            last7DaysStats = response['gamelog']['majors']['batting']['footer'][0]
            statsVsOpposingPitcher = response['matchup']['batting'][0]
#            print(last7DaysStats)

            vsLHPorRHP = None
            url = f"https://www.rotowire.com{links[i].get('href')}"
            print(url)
            response = requests.get(url,headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            vsLHPorRHP = soup.find_all('td',class_= "hide-until-sm split-end")
            if statsVsOpposingPitcher['throws'] == 'R':
                vsLHPorRHP = vsLHPorRHP[2].text
            else:
                vsLHPorRHP = vsLHPorRHP[10].text            

            if last7DaysStats.get('ab', 0).get('text') > 10:
                if int(statsVsOpposingPitcher['ab']) > 4:
                    homestats.append((float(last7DaysStats.get('obp', 0).get('text')) * .7 + float(statsVsOpposingPitcher['obp']) * .25 + float(vsLHPorRHP) * .05))
                else:
                    homestats.append((float(last7DaysStats.get('obp', 0).get('text')) * .7 + float(vsLHPorRHP) * .3))
            else:
                if int(statsVsOpposingPitcher['ab']) > 4:
                    homestats.append((float(statsForPlayer2024['obp']) * .7 + float(statsVsOpposingPitcher['obp']) * .25 + float(vsLHPorRHP) * .05))
                else:
                    homestats.append((float(statsForPlayer2024['obp']) * .7 +  float(vsLHPorRHP) * .3))


        print(homestats)
        print(awayTeam,homeTeam)

        print(awayWhip)
        print(homeWhip)

        indexForOdds = [index for index,elem in enumerate(awayTeams) if awayTeam.split()[-1].startswith(elem)][0]             

        awayTeam = team_abbreviations[awayTeam]
        homeTeam = team_abbreviations[homeTeam]

        awayScore = 0
        batterNum = 0
        numOuts = 0
        oddsAtBatHappens = 0
        
        print(awaystats)
        print(homestats)
        print(awayTeam,homeTeam)

        while numOuts < 3 and batterNum < len(awaystats):
            curBatter = float(awaystats[batterNum])
            adjCurBatter = curBatter + (.1 * (homeWhip - avgwhip) / avgwhip)
            print(f"Batter Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{awayScore}")
            
            if batterNum < 3:
                oddsAtBatHappens += adjCurBatter * adjCurBatter/.300
                if adjCurBatter < 0.300:
                    numOuts += 1
                else:
                    awayScore += adjCurBatter
                if batterNum == 2:
                    oddsAtBatHappens = min(oddsAtBatHappens,1)
            else:
                awayScore += adjCurBatter * adjCurBatter/.350 * oddsAtBatHappens
                if adjCurBatter < 0.300:
                    oddsAtBatHappens *= (1 - adjCurBatter)
                    numOuts += 1
                else:
                    oddsAtBatHappens *= min(adjCurBatter/.400,1)
                oddsAtBatHappens *= (batterNum / (batterNum+1))

            batterNum += 1        
        awayScore /= 2

        homeScore = 0
        batterNum = 0
        numOuts = 0

        oddsAtBatHappens = 0

        while numOuts < 3 and batterNum < len(homestats):
            curBatter = float(homestats[batterNum])
            adjCurBatter = curBatter + (.1 * (homeWhip - avgwhip) / avgwhip)
            print(f"Batter Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{homeScore}")
            
            if batterNum < 3:
                oddsAtBatHappens += adjCurBatter * adjCurBatter/.300
                if adjCurBatter < 0.300:
                    numOuts += 1
                else:
                    homeScore += adjCurBatter
                if batterNum == 2:
                    oddsAtBatHappens = min(oddsAtBatHappens,1)
            else:
                homeScore += adjCurBatter * adjCurBatter/.350 * oddsAtBatHappens
                if adjCurBatter < 0.300:
                    oddsAtBatHappens *= (1 - adjCurBatter)
                    numOuts += 1
                else:
                    oddsAtBatHappens *= min(adjCurBatter/.400,1)
                oddsAtBatHappens *= (batterNum / (batterNum+1))

            batterNum += 1        
        homeScore /= 2


        print(f"{awayTeam} predicted runs: {awayScore}")
        print(f"{homeTeam} predicted runs: {homeScore}")
        print(f"Predicted total runs: {homeScore + awayScore}")

        if indexForOdds:
            indexForOdds = indexForOdds[0]
            if homeScore + awayScore < 1:
                NRFIs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({odds[(2 * indexForOdds)+1]})"] + [homeScore + awayScore])
            else: 
                YRFIs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({odds[2 * indexForOdds]})"] + [homeScore + awayScore])


        else:
            if homeScore + awayScore < 1:
                NRFIs.append([-1] + [f"{awayTeam} @ {homeTeam}"] + [homeScore + awayScore]) 
            else:
                YRFIs.append([-1] + [f"{awayTeam} @ {homeTeam}"] + [homeScore + awayScore])

    
NRFIs = sorted(NRFIs,key=lambda x: x[2],reverse=True)
YRFIs = sorted(YRFIs,key=lambda x: x[2],reverse=True)


print("|--------------------------------------------------|")
print("|                      YRFIs                       |")
print(("|--------------------------------------------------|"))
for elem in YRFIs:
    print(f"|{elem[1].center(50, '-')}|")
    print("|--------------------------------------------------|")
print("\n")

print("|--------------------------------------------------|")
print("|                      NRFIs                       |")
print(("|--------------------------------------------------|"))
for elem in NRFIs:
    print(f"|{elem[1].center(50, '-')}|")
    print("|--------------------------------------------------|")


print("YRFIs")
for elem in YRFIs:
    print(elem)
print()
print("NRFIs")
for elem in NRFIs:
    print(elem)
print()

YRFIs.extend(NRFIs)


YRFIs = [elem[1:3] for elem in YRFIs]

df = pd.DataFrame(YRFIs, columns=['Game', 'numPoints'])

print(df)

plt.figure(figsize=(30, 6)) 
plt.bar(df['Game'], df['numPoints'], linestyle='-')

plt.axhline(y=1, color='r', linestyle='--')

plt.xlabel('Game')
plt.ylabel('Points in 1st inning')
plt.title('NRFI/YRFI Chart(With New Site)')

plt.savefig('newSite.png')
plt.clf()
