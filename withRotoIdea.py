import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
import sys
#For Github
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}


team_abbreviations = {
    "Diamondbacks": "ARI",
    "Braves": "ATL",
    "Orioles": "BAL",
    "Red Sox": "BOS",
    "Cubs": "CHC",
    "White Sox": "CHW",
    "Reds": "CIN",
    "Guardians": "CLE",
    "Rockies": "COL",
    "Tigers": "DET",
    "Astros": "HOU",
    "Royals": "KC",
    "Angels": "LAA",
    "Dodgers": "LAD",
    "Marlins": "MIA",
    "Brewers": "MIL",
    "Twins": "MIN",
    "Mets": "NYM",
    "Yankees": "NYY",
    "Athletics": "OAK",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Padres": "SDP",
    "Giants": "SF",
    "Mariners": "SEA",
    "Cardinals": "STL",
    "Rays": "TB",
    "Rangers": "TEX",
    "Blue Jays": "TOR",
    "Nationals": "WSH",
}

url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=1st-inning&subcategory=1st-inning-total-runs"
response = requests.get(url,headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
teamsAndLines = soup.find_all("div", class_="sportsbook-event-accordion__wrapper expanded")

awayTeams = [elem.text.split()[1] for elem in teamsAndLines]

teamsAndLines = [elem.text for elem in teamsAndLines]
odds = []
#YRFI,NRFI pattern
[odds.extend(line.split("0.5")[1:3]) for line in teamsAndLines]
odds = [elem[:4] for elem in odds]

url = 'https://www.rotowire.com/baseball/daily-lineups.php'
response = requests.get(url,headers=headers)

avgwhip = 1.313	

half_innings = []
NRFIs = []
YRFIs = []
GameAgreeBothHalfs = []

#NRFIs2 = []
#YRFIs2 = []

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a')
    
#    print("Starting at 470 and so can tell if 23 or 24")
#    [print(elem) for elem in enumerate(links[470:520])]
    matchuplocs = [index for index, link in enumerate(links) if 'lineup__matchup' in link.get('class', [])]
    links = links[matchuplocs[0]:-54]

    game_times = soup.find_all('div',class_="lineup__time")[:-2]
    game_times = [elem.text for elem in game_times]
    confirmedOrExpected = soup.find_all('li',class_="lineup__status")
    confirmedOrExpected = [elem.text.strip().split()[0][0] for elem in confirmedOrExpected]

    print(matchuplocs)
    for i in range(len(matchuplocs)):
        [print(elem) for elem in links[matchuplocs[i]:matchuplocs[i+1]]]
        index = matchuplocs[i]-matchuplocs[0]
    
        awaystats = []
        homestats = []
        
        awaystats2 = []
        homestats2 = []

        split = links[index].text.split("(")

        awayTeam = split[0].strip()
        print(f"awayteam: {awayTeam}")


        url = f"https://www.rotowire.com{links[index+1].get('href')}"
        print("A",url)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        awayWhip = soup.find_all(class_="p-card__stat-value")[2].text
        if awayWhip == '0.00':
            awayWhip = avgwhip
        awayWhip = float(awayWhip)

#        print(f"awayWhip: {awayWhip}")

        for i in range(index +2,index + 11,1):
#            print(links[i])
            try:
                url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={links[i].get('href').split('-')[-1]}&stats=batting"
                print("B",url)
                response = json.loads(requests.get(url,headers=headers).text)
                #avg,obo,slg, and ops
                statsForPlayer2024 = response['basic']['batting']['body'][-1]
            except KeyError:
                print(f"Key error for: {url}")
                sys.exit()

            if statsForPlayer2024['season'] == '2024' and statsForPlayer2024['league_level'] == 'MAJ' :
                last7DaysStats = response['gamelog']['majors']['batting']['footer'][0]
                statsVsOpposingPitcher = response['matchup']['batting'][0]

                vsLHPorRHP = None
                vsLHPorRHP2 = None
                url = f"https://www.rotowire.com{links[i].get('href')}"
                print("C",url)
                response = requests.get(url,headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')

                vsLHPorRHP = soup.find_all('td',class_= "hide-until-sm split-end")
                if statsVsOpposingPitcher['throws'] == 'R':
                    vsLHPorRHP = vsLHPorRHP[2].text
                else:
                    vsLHPorRHP = vsLHPorRHP[10].text            
    
                vsLHPorRHP2 = soup.find_all('td',class_= "split-end")
                if statsVsOpposingPitcher['throws'] == 'R':
                    vsLHPorRHP2 = vsLHPorRHP2[11].text
                else:
                    vsLHPorRHP2 = soup.find_all('td',class_= "split-start")
                    vsLHPorRHP2 = vsLHPorRHP2[11].text            

                
                if int(last7DaysStats.get('ab', 0).get('text')) > 10:
                    if int(statsVsOpposingPitcher['ab']) > 4:
                        awaystats.append((float(last7DaysStats.get('obp', 0).get('text')) * .7 + float(statsVsOpposingPitcher['obp']) * .25 + float(vsLHPorRHP) * .05))
                        awaystats2.append((float(last7DaysStats.get('ops', 0).get('text')) * .7 + float(statsVsOpposingPitcher['ops']) * .25 + float(vsLHPorRHP2) * .05))
                    else:
                        awaystats.append((float(last7DaysStats.get('obp', 0).get('text')) * .7 + float(vsLHPorRHP) * .3))
                        awaystats2.append((float(last7DaysStats.get('ops', 0).get('text')) * .7 + float(vsLHPorRHP2) * .3))
                else:
                    if int(statsVsOpposingPitcher['ab']) > 4:
                        awaystats.append((float(statsForPlayer2024['obp']) * .7 + float(statsVsOpposingPitcher['obp']) * .25 + float(vsLHPorRHP) * .05))
                        awaystats2.append((float(statsForPlayer2024['ops']) * .7 + float(statsVsOpposingPitcher['ops']) * .25 + float(vsLHPorRHP2) * .05))
                    else:
                        awaystats.append((float(statsForPlayer2024['obp']) * .7 +  float(vsLHPorRHP) * .3))
                        awaystats2.append((float(statsForPlayer2024['ops']) * .7 +  float(vsLHPorRHP2) * .3))
            else:
                awaystats.append((float(statsForPlayer2024['obp']) * .875))
                awaystats2.append((float(statsForPlayer2024['ops']) * .875))


#        print(awaystats)

        homeTeam = str(split[-2][8:]).strip()
#        print(f"homeTeam: {homeTeam}")

        url = f"https://www.rotowire.com{links[index+11].get('href')}"
        print("D",url)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        homeWhip = soup.find_all(class_="p-card__stat-value")[2].text
        if homeWhip == '0.00':
            homeWhip = avgwhip
        homeWhip = float(homeWhip)
#        print(f"homeWhip: {homeWhip}")

        for i in range(index +12,index + 21,1):
            try:
                url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={links[i].get('href').split('-')[-1]}&stats=batting"
                print("E",url)
                response = json.loads(requests.get(url,headers=headers).text)
                #avg,obo,slg, and ops
                statsForPlayer2024 = response['basic']['batting']['body'][-1]
            except KeyError:
                print(f"Key error for: {url}")
                sys.exit()

            if statsForPlayer2024['season'] == '2024' and statsForPlayer2024['league_level'] == 'MAJ' :
                last7DaysStats = response['gamelog']['majors']['batting']['footer'][0]
                statsVsOpposingPitcher = response['matchup']['batting'][0]

                vsLHPorRHP = None
                url = f"https://www.rotowire.com{links[i].get('href')}"
                print("F",url)
                response = requests.get(url,headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')

                vsLHPorRHP = soup.find_all('td',class_= "hide-until-sm split-end")
                if statsVsOpposingPitcher['throws'] == 'R':
                    vsLHPorRHP = vsLHPorRHP[2].text
                else:
                    vsLHPorRHP = vsLHPorRHP[10].text            

                vsLHPorRHP2 = soup.find_all('td',class_= "split-end")
                if statsVsOpposingPitcher['throws'] == 'R':
                    vsLHPorRHP2 = vsLHPorRHP2[11].text
                else:
                    vsLHPorRHP2 = soup.find_all('td',class_= "split-start")
                    vsLHPorRHP2 = vsLHPorRHP2[11].text            

                
                if int(last7DaysStats.get('ab', 0).get('text')) > 10:
                    if int(statsVsOpposingPitcher['ab']) > 4:
                        homestats.append((float(last7DaysStats.get('obp', 0).get('text')) * .7 + float(statsVsOpposingPitcher['obp']) * .25 + float(vsLHPorRHP) * .05))
                        homestats2.append((float(last7DaysStats.get('ops', 0).get('text')) * .7 + float(statsVsOpposingPitcher['ops']) * .25 + float(vsLHPorRHP2) * .05))
                    else:
                        homestats.append((float(last7DaysStats.get('obp', 0).get('text')) * .7 + float(vsLHPorRHP) * .3))
                        homestats2.append((float(last7DaysStats.get('ops', 0).get('text')) * .7 + float(vsLHPorRHP2) * .3))
                else:
                    if int(statsVsOpposingPitcher['ab']) > 4:
                        homestats.append((float(statsForPlayer2024['obp']) * .7 + float(statsVsOpposingPitcher['obp']) * .25 + float(vsLHPorRHP) * .05))
                        homestats2.append((float(statsForPlayer2024['ops']) * .7 + float(statsVsOpposingPitcher['ops']) * .25 + float(vsLHPorRHP2) * .05))
                    else:
                        awaystats.append((float(statsForPlayer2024['obp']) * .7 +  float(vsLHPorRHP) * .3))
                        homestats2.append((float(statsForPlayer2024['ops']) * .7 +  float(vsLHPorRHP2) * .3))
            else:
                homestats.append((float(statsForPlayer2024['obp']) * .875))
                homestats2.append((float(statsForPlayer2024['ops']) * .875))


#        print(homestats)
        print(awayTeam,homeTeam)

        print(awayWhip)
        print(homeWhip)

        indexForOdds = [index for index,elem in enumerate(awayTeams) if elem.startswith(awayTeam.split()[-1])]
        awayTeam = team_abbreviations[awayTeam]
        homeTeam = team_abbreviations[homeTeam]
        
        awayScore = 0
        batterNum = 0
        numOuts = 0
        oddsAtBatHappens = 0
        
        print(awaystats)
        print(homestats)
#        print(awayTeam,homeTeam)

        while numOuts < 3 and batterNum < len(awaystats):
            curBatter = float(awaystats[batterNum])
            adjCurBatter = curBatter + (.1 * (homeWhip - avgwhip) / avgwhip)
            print(f"Batter Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{awayScore}")
            
            if batterNum < 3:
                oddsAtBatHappens += adjCurBatter * adjCurBatter/.330
                if adjCurBatter < 0.330:
                    numOuts += 1
                else:
                    awayScore += adjCurBatter
                if batterNum == 2:
                    oddsAtBatHappens = min(oddsAtBatHappens,1)
            else:
                awayScore += adjCurBatter * adjCurBatter/.350 * oddsAtBatHappens
                if adjCurBatter < 0.330:
#                    oddsAtBatHappens *= (1 - adjCurBatter)
                    oddsAtBatHappens -= adjCurBatter
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
            adjCurBatter = curBatter + (.1 * (awayWhip - avgwhip) / avgwhip)
            print(f"Batter Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{homeScore}")
            
            if batterNum < 3:
                oddsAtBatHappens += adjCurBatter * adjCurBatter/.330
                if adjCurBatter < 0.330:
                    numOuts += 1
                else:
                    homeScore += adjCurBatter
                if batterNum == 2:
                    oddsAtBatHappens = min(oddsAtBatHappens,1)
            else:
                homeScore += adjCurBatter * adjCurBatter/.350 * oddsAtBatHappens
                if adjCurBatter < 0.330:
#                    oddsAtBatHappens *= (1 - adjCurBatter)
                    oddsAtBatHappens -= adjCurBatter
                    numOuts += 1
                else:
                    oddsAtBatHappens *= min(adjCurBatter/.400,1)
                oddsAtBatHappens *= (batterNum / (batterNum+1))

            batterNum += 1        
        homeScore /= 2


        print(f"{awayTeam} predicted runs: {awayScore}")
        print(f"{homeTeam} predicted runs: {homeScore}")
        print(f"Predicted total runs: {homeScore + awayScore}")

        status_index = int(index//11.5)
        try:
            half_innings.append([f"{awayTeam}({game_times[index//23]})"] + [round(awayScore,3)] + [f"{confirmedOrExpected[status_index]}"])
            half_innings.append([f"{homeTeam}({game_times[index//23]})"] + [round(homeScore,3)] + [f"{confirmedOrExpected[status_index + 1]}"])        
        except:
            print(status_index,status_index+1)
            print(index//23)
            [print(elem) for elem in enumerate(confirmedOrExpected)]
            [print(elem) for elem in enumerate(game_times)]
            raise SystemError

        game_lineup_status = "C"
        if confirmedOrExpected[status_index] != game_lineup_status or confirmedOrExpected[status_index +1] != game_lineup_status:
            game_lineup_status = "E"
        
        if indexForOdds:
            indexForOdds = indexForOdds[0]
            if homeScore + awayScore < 1:
                NRFIs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})({odds[(2 * indexForOdds)+1]})"] + [round(homeScore + awayScore,3)] + [f"{game_lineup_status}"])
                if homeScore < .5 and awayScore < .5:
                    GameAgreeBothHalfs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})({odds[(2 * indexForOdds)+1]})"] + [round(homeScore + awayScore,2)] + [round(awayScore,2)] + [round(homeScore,2)] + [f"{game_lineup_status}"])
            else: 
                YRFIs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})({odds[2 * indexForOdds]})"] + [round(homeScore + awayScore,3)] + [f"{game_lineup_status}"])
                if homeScore >= .5 and awayScore >= .5:
                    GameAgreeBothHalfs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})({odds[2 * indexForOdds]})"] + [round(homeScore + awayScore,2)] + [round(awayScore,2)] + [round(homeScore,2)] + [f"{game_lineup_status}"])

        else:
            if homeScore + awayScore < 1:
                NRFIs.append([-1] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})"] + [round(homeScore + awayScore,3)] + [f"{game_lineup_status}"]) 
                if homeScore < .5 and awayScore < .5:
                    GameAgreeBothHalfs.append([-1] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})"] + [round(homeScore + awayScore,2)] + [round(awayScore,2)] + [round(homeScore,2)] + [f"{game_lineup_status}"])
            else:
                YRFIs.append([-1] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})"] + [round(homeScore + awayScore,3)] + [f"{game_lineup_status}"])
                if homeScore >= .5 and awayScore >= .5:
                    GameAgreeBothHalfs.append([-1] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})"] + [round(homeScore + awayScore,2)] + [round(awayScore,2)] + [round(homeScore,2)] + [f"{game_lineup_status}"])


print("\n\n")
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

print("YRFIs:")
for elem in YRFIs:
    print(elem[1:])
print()
print("NRFIs:")
for elem in NRFIs:
    print(elem[1:])
print()


half_innings = sorted(half_innings,key=lambda x: x[1],reverse=True)    
print("Half innings")
print("YRFIs: ")
firstNRFI = True
for elem in half_innings:
    if elem[1] < .5 and firstNRFI:
        print("\nNRFIs: ")
        firstNRFI = False
    print(elem)

GameAgreeBothHalfs = sorted(GameAgreeBothHalfs,key=lambda x: x[2],reverse=True)
print("Both half inning predictions match full game prediction")
for elem in GameAgreeBothHalfs:
    print(elem[1:])


YRFIs.extend(NRFIs)

YRFIs = [elem[1:] for elem in YRFIs]

df = pd.DataFrame(YRFIs, columns=['Game', 'numPoints','Status'])

print(df)

plt.figure(figsize=(55, 6)) 
plt.bar(df['Game'] + df['Status'], df['numPoints'], linestyle='-')

plt.axhline(y=1, color='r', linestyle='--')

plt.xlabel('Game')
plt.ylabel('Points in 1st inning')
plt.title('NRFI/YRFI Chart')

plt.savefig('gameNRFIYRFI.png')
plt.clf()

df = pd.DataFrame(half_innings, columns=['Team', 'numPoints','Status'])

print(df)

plt.figure(figsize=(55, 6)) 
plt.bar(df['Team'] + df['Status'], df['numPoints'], linestyle='-')

plt.axhline(y=.5, color='r', linestyle='--')

plt.xlabel('Team')
plt.ylabel('Points in 1st inning')
plt.title('Half inning NRFI/YRFI Chart')

plt.savefig('teamNRFIYRFI.png')
plt.clf()
