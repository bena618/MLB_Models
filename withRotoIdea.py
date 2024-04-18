import requests
from bs4 import BeautifulSoup
import re
from selenium import webdriver

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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


url = 'https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow'
response = requests.get(url)

avgwhip = 1.313	

NRFIs = []
YRFIs = []
NRFIsUseAVG = []
YRFIsUseAVG = []
bothAgree = []

chrome_options = webdriver.ChromeOptions() 
#chrome_options.add_argument('--headless') 
#chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=chrome_options) 
url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=1st-inning&subcategory=1st-inning-total-runs"
driver.get(url)
html = driver.page_source    
soup = BeautifulSoup(html, "html.parser")
teamsAndLines = soup.find_all("div", class_="sportsbook-event-accordion__wrapper expanded")

awayTeams = [elem.text.split()[1] for elem in teamsAndLines]
awayTeams = [elem[:elem.index('at')] for elem in awayTeams]
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
    links = links[487:-54]
#    print(len(links))
    [print(elem) for elem in enumerate(links)]

    for index in range(0,len(links),23):
        awaystats = []
        homestats = []
        awayTeam = " ".join(links[index].text.split()[:-3])
        print(awayTeam)

        url = f"https://www.rotowire.com{links[index+1].get('href')}"
        print(url)
        
        response = requests.get(url,headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        awayWhip = soup.find_all(class_="p-card__stat-value")[2].text
        if awayWhip == 0:
            awayWhip = avgwhip
        print(awayWhip)

        for i in range(index +2,11,1):
            url = f"https://www.rotowire.com{links[index].get('href')}"
            driver.get(url)
            driver.implicitly_wait(100)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            theTable = soup.find_all('div',class_="light")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            print(theTable)
            raise SystemError
            #avg,obo,slg, and ops
#            awaystats.extend[] = re.findall(wOBApattern, elems)

        print(awaystats)
        raise SystemError
        awaystats2 = re.findall(avgPattern, elems)
        awaystats2 = [elem[:4] for elem in awaystats2]
#        print(elems)
#        [print(elem) for elem in enumerate(awaystats2)]
#        awaystats2 = awaystats2[::3]


        homeTeam = " ".join(names[index+11].text.split()[:-2])

        elems = ""
        elems += "".join([elem.text for elem in names[index+13:index+22]])

        homestats = re.findall(wOBApattern, elems)
        homestats2 = re.findall(avgPattern, elems)
        homestats2 = [elem[:4] for elem in homestats2]

#        homestats2 = homestats2[::3]

        print(homestats)
        print(awayTeam,homeTeam)
        #Way this is rn away team faces home whip 
        awayWhip = avgwhip
        #.replace('.', '') so st. louis doesnt throw stuff off
#        print(i,pitchers[i],"-".join(awayTeam.split()).lower(),pitchers[i+1])
        if pitchers[i].find("-".join(awayTeam.split()).lower().replace('.', '')) != -1: 
            try:
                url = f"https://www.fantasypros.com{pitchers[i+1]}"
                i+=2
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                theTable = soup.find_all('table')[3]

        #        awayWhip = (float)((theTable.find_all('td')[-1].text))
    #        try:    
                awayWhip = float(theTable.find_all('td')[-1].text) 
                if homeWhip == 0.0:
                    homeWhip = avgwhip

    #        except ValueError: 
            except:
#                i-=1
                print(f"Error so average used instead for awayWhip({awayTeam}), likely no listed pitcher or no stats so risky to follow")
        else:
            i+=1
    #            awayWhip = avgwhip

        homeWhip = avgwhip
        #.replace('.', '') so st. louis doesnt throw stuff off
#        print(i,pitchers[i],"-".join(homeTeam.split()).lower().replace('.', ''),pitchers[i+1])
        if pitchers[i].find("-".join(homeTeam.split()).lower().replace('.', '')) != -1:
            try:
                url = f"https://www.fantasypros.com{pitchers[i+1]}"
                i+=2
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                theTable = soup.find_all('table')[3]
    #        homeWhip = (float)((theTable.find_all('td')[-1].text))
    #        try:    
                homeWhip = float(theTable.find_all('td')[-1].text)
                #Dodgers had this one time and in general just very unlikely to be real because would require perfect game every appearence and would have to be a starter doing that
                if homeWhip == 0.0:
                    homeWhip = avgwhip
 
    #        except ValueError: 
            except:
                print(f"Error so average used instead for homeWhip({homeTeam}), likely no listed pitcher or no stats so risky to follow")
#                i-=1
        else:
            i+=1
#                homeWhip = avgwhip

        print(awayWhip)
        print(homeWhip)

        awayScore = 0
        batterNum = 0
        numOuts = 0
        oddsAtBatHappens = 0
        
        #Techincally shouldny limit # batters but if get to 9 then yrfi has to hapen
        while numOuts < 3 and batterNum < len(awaystats):
            curBatter = float(awaystats[batterNum])
#            adjCurBatter = curBatter + (homeWhip-avgwhip)/4.5
            adjCurBatter = curBatter + (.1 * (homeWhip - avgwhip) / avgwhip)
            print(f"Batter Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{awayScore}")
            
            if batterNum < 3:
                oddsAtBatHappens += adjCurBatter * adjCurBatter/.340
                if adjCurBatter < 0.340:
#                    oddsAtBatHappens += adjCurBatter * adjCurBatter/.340
                    numOuts += 1
                else:
#                    oddsAtBatHappens += adjCurBatter
                    awayScore += adjCurBatter
                if batterNum == 2:
                    oddsAtBatHappens = min(oddsAtBatHappens/1.2,1)
            else:
#                awayScore += adjCurBatter * oddsAtBatHappens
                awayScore += adjCurBatter * adjCurBatter/.400 * oddsAtBatHappens
                if adjCurBatter < 0.340:
                    oddsAtBatHappens *= (1 - adjCurBatter)
                    numOuts += 1
                else:
                    #Odds for consecutive batters at most will stay the same 
                    oddsAtBatHappens *= min(adjCurBatter/.400,1)
#                awayScore += adjCurBatter * oddsAtBatHappens
#                awayScore += adjCurBatter * adjCurBatter/.400 * oddsAtBatHappens
                oddsAtBatHappens *= (batterNum / (batterNum+1))


            '''
            if adjCurBatter < .340:
                numOuts +=1
#                oddsAtBatHappens -= .333
                oddsAtBatHappens += adjCurBatter * curBatter/.340 * .2

            else:
                oddsAtBatHappens += adjCurBatter * curBatter/.340

            if batterNum == 3:
                oddsAtBatHappens -= 2    
            if batterNum+1 >= 3:
                awayScore += adjCurBatter * adjCurBatter/.340 * oddsAtBatHappens
            else:
                awayScore += adjCurBatter * adjCurBatter/.340
            '''

#            print(f"Batter Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{awayScore}")
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
                oddsAtBatHappens += adjCurBatter * adjCurBatter/.340
                if adjCurBatter < 0.340:
                    numOuts += 1
                else:
                    homeScore += adjCurBatter
                if batterNum == 2:
                    oddsAtBatHappens = min(oddsAtBatHappens/1.2,1)
            else:
#                homeScore += adjCurBatter * oddsAtBatHappens
                homeScore += adjCurBatter * adjCurBatter/.400 * oddsAtBatHappens
                if adjCurBatter < 0.340:
                    oddsAtBatHappens *= (1 - adjCurBatter)
                    numOuts += 1
                else:
                    #Odds for consecutive batters at most will stay the same 
                    oddsAtBatHappens *= min(adjCurBatter/.400,1)
#                homeScore += adjCurBatter * oddsAtBatHappens
#                homeScore += adjCurBatter * adjCurBatter/.400 * oddsAtBatHappens


            '''
            curBatter = float(homestats[batterNum])
            adjCurBatter = curBatter + (awayWhip-avgwhip)/4.5

            if batterNum < 3:
               oddsAtBatHappens += adjCurBatter * curBatter/.340

            if adjCurBatter < .340:
                numOuts +=1
#                oddsAtBatHappens -= .15
                oddsAtBatHappens += adjCurBatter * curBatter/.340 * .2
            else:
                oddsAtBatHappens += adjCurBatter * curBatter/.340

            if batterNum == 3:
                oddsAtBatHappens -= 2    
            if batterNum+1 >= 3:
                homeScore += adjCurBatter * adjCurBatter/.340 * oddsAtBatHappens
            else:
                homeScore += adjCurBatter * adjCurBatter/.340
            '''
#            print(f"Batter Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{homeScore}")

            batterNum += 1        
        homeScore /= 2


        print(f"{awayTeam} predicted runs: {awayScore}")
        print(f"{homeTeam} predicted runs: {homeScore}")
        print(f"Predicted total runs: {homeScore + awayScore}")
#        print(f"Predicted odds or NRFI: ")
        indexForOdds = [index for index,elem in enumerate(awayTeams) if awayTeam.split()[-1].startswith(elem)]             
#        print(indexForOdds)

        awayTeam = team_abbreviations[awayTeam]
        homeTeam = team_abbreviations[homeTeam]

        if indexForOdds:
#            indexForOdds = indexForOdds[0]
            NRFIs.append([indexForOdds[0]] + [f"{awayTeam} @ {homeTeam}({odds[(2 * indexForOdds[0])+1]})"] + [homeScore + awayScore]) if homeScore + awayScore < 1 else YRFIs.append([indexForOdds[0]] + [f"{awayTeam} @ {homeTeam}({odds[2 * indexForOdds[0]]})"] + [homeScore + awayScore])
        else:
            NRFIs.append([-1] + [f"{awayTeam} @ {homeTeam}"] + [homeScore + awayScore]) if homeScore + awayScore < 1 else YRFIs.append([-1] + [f"{awayTeam} @ {homeTeam}"] + [homeScore + awayScore])


#        print(awayWhip)
#        print(homeWhip)

        awayScore = 0
        batterNum = 0
        numOuts = 0
        oddsAtBatHappens = 0
        
        print(awaystats2)
        print(homestats2)
        print(awayTeam,homeTeam)

        while numOuts < 3 and batterNum < len(awaystats2):
            curBatter = float(awaystats2[batterNum])
            adjCurBatter = curBatter + (.1 * (homeWhip - avgwhip) / avgwhip)
            print(f"Batter2 Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{awayScore}")
            
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

        while numOuts < 3 and batterNum < len(homestats2):
            curBatter = float(homestats2[batterNum])
            adjCurBatter = curBatter + (.1 * (homeWhip - avgwhip) / avgwhip)
            print(f"Batter2 Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{homeScore}")
            
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
#            print(f"Batter Num: {batterNum +1}, {adjCurBatter},{oddsAtBatHappens},{homeScore}")

            batterNum += 1        
        homeScore /= 2


        print(f"{awayTeam}2 predicted runs: {awayScore}")
        print(f"{homeTeam}2 predicted runs: {homeScore}")
        print(f"Predicted total runs: {homeScore + awayScore}")

        if indexForOdds:
            indexForOdds = indexForOdds[0]
            if homeScore + awayScore < 1:
                NRFIsUseAVG.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({odds[(2 * indexForOdds)+1]})"] + [homeScore + awayScore])
                #if was just added to nrfi then would be agreement
                tmp = NRFIs[len(NRFIs) -1]
                if tmp[1] == f'{awayTeam} @ {homeTeam}({odds[(2 * indexForOdds)+1]})':
                    tmp.extend([homeScore + awayScore])
                    bothAgree.append(tmp)
            else: 
                YRFIsUseAVG.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({odds[2 * indexForOdds]})"] + [homeScore + awayScore])
                tmp = YRFIs[len(YRFIs) -1]
                if tmp[1] == f'{awayTeam} @ {homeTeam}({odds[(2 * indexForOdds)]})':
                    tmp.extend([homeScore + awayScore])
                    bothAgree.append(tmp)


        else:
            if homeScore + awayScore < 1:
                NRFIsUseAVG.append([-1] + [f"{awayTeam} @ {homeTeam}"] + [homeScore + awayScore]) 
                tmp = NRFIs[len(NRFIs) -1]
                print(f"NRFI tmp: {tmp},{homeScore + awayScore},{awayTeam} @ {homeTeam},{tmp[1]},{tmp[1] == f'{awayTeam} @ {homeTeam}'}")
                if tmp[1] == f'{awayTeam} @ {homeTeam}':
                    tmp.extend([homeScore + awayScore])
                    bothAgree.append(tmp)
            else:
                YRFIsUseAVG.append([-1] + [f"{awayTeam} @ {homeTeam}"] + [homeScore + awayScore])
                tmp = YRFIs[len(YRFIs) -1]
                print(f"YRFI tmp: {tmp},{homeScore + awayScore},{awayTeam} @ {homeTeam},{tmp[1]},{tmp[1] == f'{awayTeam} @ {homeTeam}'}")
                if tmp[1] == f'{awayTeam} @ {homeTeam}':
                    tmp.extend([homeScore + awayScore])
                    bothAgree.append(tmp)
#        print(bothAgree)


        '''
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

        '''
        '''    
        awayScore = 0
        numHitsPredicted = 0
        batterNum = 0
        numBatswOBAgt330wOutPitcher = 0
        numBatsGoodVspitcer = 0
        numOuts = 0

        
        while numOuts < 3
            curBatter = awaystats[batterNum]
            adjCurBatter = curBatter * homeWhip/avgwhip
            if curBatter >= .330:
                numBatswOBAgt330wOutPitcher += 1
                if adjCurBatter >= .330:
                    numBatsGoodVspitcer += 1
            else:
                if curBatter <= .270 and adjCurBatter <= .270:
                    numOuts += 1
                    awayScore += adjCurBatter        

            awayScore += adjCurBatter

            batterNum += 1
            
        print(f"{awayTeam} runs: {awayScore}")
        '''
#        i += 2
    
NRFIs = sorted(NRFIs,key=lambda x: x[2],reverse=True)
YRFIs = sorted(YRFIs,key=lambda x: x[2],reverse=True)
NRFIsUseAVG = sorted(NRFIsUseAVG,key=lambda x: x[2],reverse=True)
YRFIsUseAVG = sorted(YRFIsUseAVG,key=lambda x: x[2],reverse=True)    
bothAgree = sorted(bothAgree,key=lambda x: (x[2] + x[3])/2,reverse=True)


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


print("|--------------------------------------------------|")
print("|                 YRFIs(AVG)                       |")
print(("|--------------------------------------------------|"))
for elem in YRFIsUseAVG:
    print(f"|{elem[1].center(50, '-')}|")
    print("|--------------------------------------------------|")
print("\n")

print("|--------------------------------------------------|")
print("|                 NRFIs(AVG)                       |")
print(("|--------------------------------------------------|"))
for elem in NRFIsUseAVG:
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
print("YRFIs(AVG)")
for elem in YRFIsUseAVG:
    print(elem)
print()
print("NRFIs(AVG)")
for elem in NRFIsUseAVG:
    print(elem)

YRFIs.extend(NRFIs)


YRFIs = [elem[1:3] for elem in YRFIs]

df = pd.DataFrame(YRFIs, columns=['Game', 'numPoints'])

print(df)

plt.figure(figsize=(30, 6)) 
plt.bar(df['Game'], df['numPoints'], linestyle='-')

plt.axhline(y=1, color='r', linestyle='--')

plt.xlabel('Game')
plt.ylabel('Points in 1st inning')
plt.title('NRFI/YRFI Chart(Original)')

plt.savefig('plot.png')
plt.clf()


YRFIsUseAVG.extend(NRFIsUseAVG)


YRFIsUseAVG = [elem[1:3] for elem in YRFIsUseAVG]

df = pd.DataFrame(YRFIsUseAVG, columns=['Game', 'numPoints'])

print(df)

plt.figure(figsize=(30, 6)) 
plt.bar(df['Game'], df['numPoints'], linestyle='-')

plt.axhline(y=1, color='r', linestyle='--')

plt.xlabel('Game')
plt.ylabel('Points in 1st inning')
plt.title('NRFI/YRFI Chart(Variation)')

plt.savefig('plot2.png')
plt.clf()

plt.figure(figsize=(30, 6)) 

X_axis = np.arange(len(bothAgree)) 

plt.bar(X_axis - 0.2, [elem[2] for elem in bothAgree], 0.4, label = 'Original') 
plt.bar(X_axis + 0.2, [elem[3] for elem in bothAgree], 0.4, label = 'Variation') 

plt.axhline(y=1, color='r', linestyle='--')

plt.xticks(X_axis, [elem[1] for elem in bothAgree]) 
plt.xlabel('Game')
plt.ylabel('Points in 1st inning')
plt.title('NRFI/YRFI Agreements')
plt.legend() 
plt.savefig('plot3.png')
