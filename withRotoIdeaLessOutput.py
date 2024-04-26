import requests
from bs4 import BeautifulSoup

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
    "Red": "BOS",
    "Cubs": "CHC",
    "White": "CHW",
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
    "Blue": "TOR",
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

output_lines = []

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a')
    matchuplocs = [index for index, link in enumerate(links) if 'lineup__matchup' in link.get('class', [])]
    links = links[matchuplocs[0]:-54]
    game_times = soup.find_all('div',class_="lineup__time")[:-2]
    game_times = [elem.text for elem in game_times]

    for i in range(len(matchuplocs)):
        index = matchuplocs[i]-matchuplocs[0]
    
        awaystats = []
        homestats = []
        
        awaystats2 = []
        homestats2 = []

        awayTeam = links[index].text.split()[0]


        url = f"https://www.rotowire.com{links[index+1].get('href')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        awayWhip = soup.find_all(class_="p-card__stat-value")[2].text
        if awayWhip == '0.00':
            awayWhip = avgwhip
        awayWhip = float(awayWhip)

        for i in range(index +2,index + 11,1):
            try:
                url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={links[i].get('href').split('-')[-1]}&stats=batting"
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

        homeTeam = links[index].text.split()[-2]

        url = f"https://www.rotowire.com{links[index+11].get('href')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        homeWhip = soup.find_all(class_="p-card__stat-value")[2].text
        if homeWhip == '0.00':
            homeWhip = avgwhip
        homeWhip = float(homeWhip)

        for i in range(index +12,index + 21,1):
            try:
                url = f"https://www.rotowire.com/baseball/ajax/player-page-data.php?id={links[i].get('href').split('-')[-1]}&stats=batting"
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

        indexForOdds = [index for index,elem in enumerate(awayTeams) if elem.startswith(awayTeam.split()[-1])]             

        awayTeam = team_abbreviations[awayTeam]
        homeTeam = team_abbreviations[homeTeam]

        awayScore = 0
        batterNum = 0
        numOuts = 0
        oddsAtBatHappens = 0
        
        while numOuts < 3 and batterNum < len(awaystats):
            curBatter = float(awaystats[batterNum])
            adjCurBatter = curBatter + (.1 * (homeWhip - avgwhip) / avgwhip)
            
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
            adjCurBatter = curBatter + (.1 * (awayWhip - avgwhip) / avgwhip)
            
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
                    oddsAtBatHappens *= (1 - adjCurBatter)
                    numOuts += 1
                else:
                    oddsAtBatHappens *= min(adjCurBatter/.400,1)
                oddsAtBatHappens *= (batterNum / (batterNum+1))

            batterNum += 1        
        homeScore /= 2

        half_innings.append([f"{awayTeam}({game_times[index//23]})"] + [awayScore])
        half_innings.append([f"{homeTeam}({game_times[index//23]})"] + [homeScore])        

        if indexForOdds:
            indexForOdds = indexForOdds[0]
            if homeScore + awayScore < 1:
                NRFIs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})({odds[(2 * indexForOdds)+1]})"] + [homeScore + awayScore])
                if homeScore < .5 and awayScore < .5:
                    GameAgreeBothHalfs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})({odds[(2 * indexForOdds)+1]})"] + [homeScore + awayScore] + [awayScore] + [homeScore])
            else: 
                YRFIs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})({odds[2 * indexForOdds]})"] + [homeScore + awayScore])
                if homeScore >= .5 and awayScore >= .5:
                    GameAgreeBothHalfs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})({odds[2 * indexForOdds]})"] + [homeScore + awayScore] + [awayScore] + [homeScore])

        else:
            if homeScore + awayScore < 1:
                NRFIs.append([-1] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})"] + [homeScore + awayScore]) 
                if homeScore < .5 and awayScore < .5:
                    GameAgreeBothHalfs.append([-1] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})"] + [homeScore + awayScore] + [awayScore] + [homeScore])
            else:
                YRFIs.append([-1] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})"] + [homeScore + awayScore])
                if homeScore >= .5 and awayScore >= .5:
                    GameAgreeBothHalfs.append([-1] + [f"{awayTeam} @ {homeTeam}({game_times[index//23]})"] + [homeScore + awayScore] + [awayScore] + [homeScore])

NRFIs = sorted(NRFIs,key=lambda x: x[2],reverse=True)
YRFIs = sorted(YRFIs,key=lambda x: x[2],reverse=True)


output_lines.append("YRFIs:")
for elem in YRFIs:
    output_lines.append(f"{elem[1:]}")
output_lines.append("\n")

output_lines.append("NRFIs:")
for elem in NRFIs:
    output_lines.append(f"{elem[1:]}")
output_lines.append("\n")


half_innings = sorted(half_innings,key=lambda x: x[1],reverse=True)    
output_lines.append("Half innings")
output_lines.append("YRFIs:")
firstNRFI = True
for elem in half_innings:
    if elem[1] < .5 and firstNRFI:
        output_lines.append("\nNRFIs:")
        firstNRFI = False
    output_lines.append(f"{elem}")

GameAgreeBothHalfs = sorted(GameAgreeBothHalfs,key=lambda x: x[2],reverse=True)
output_lines.append("Both half inning predictions match full game prediction:\n")
for elem in GameAgreeBothHalfs:
    output_lines.append(f"{elem[1:]}")
    
output = '\n'.join(output_lines)
for elem in output_lines:
    print(elem)
print(output)

print("Printing...")
print(output)
