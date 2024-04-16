import requests
from bs4 import BeautifulSoup
import re
from selenium import webdriver


#For Github
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}

url = 'https://www.fantasypros.com/mlb/lineups/'
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
awayTeams = [elem[:elem.index('at')] for elem in awayTeams]

teamsAndLines = [elem.text for elem in teamsAndLines]
odds = []
#YRFI,NRFI pattern
[odds.extend(line.split("0.5")[1:3]) for line in teamsAndLines]
odds = [elem[:4] for elem in odds]
#print(odds)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    pitchers = soup.find_all('div', class_='game-info')
    pitchers = [elem.find_all("a") for elem in pitchers]
    pitchers = [elem.get('href') for list in pitchers for elem in list]
    pitchers = pitchers[1::2]

    names = soup.find_all("tr")

#    print(names)
    i = 0
    #[print("".join(elem.text)) for elem in names[0:22]]
    for index in range(0,len(names),22):
        awayTeam = " ".join(names[index].text.split()[:-2])
        wOBApattern = r'(1?\.[0-9]{3}){3}'

#        [print("".join(elem.text[1:])) for elem in names[index+2:index+11]]
#        [print("".join(elem.text[1:])) for elem in names[index+2:index+6]]

        elems = ""
        elems += "".join([elem.text for elem in names[index+2:index+11]])

        awaystats = re.findall(wOBApattern, elems)
        print(awaystats)

        homeTeam = " ".join(names[index+11].text.split()[:-2])
        elems = ""
        elems += "".join([elem.text for elem in names[index+13:index+22]])

        homestats = re.findall(wOBApattern, elems)
        print(homestats)
        print(awayTeam,homeTeam)

        try:
            url = f"https://www.fantasypros.com{pitchers[i]}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            theTable = soup.find_all('table')[3]
    #        awayWhip = (float)((theTable.find_all('td')[-1].text))
#        try:    
            awayWhip = float(theTable.find_all('td')[-1].text) 
#        except ValueError: 
        except:
            print("Error so average used instead for awayWhip, likely no listed pitcher or no stats so risky to follow")
            awayWhip = avgwhip

        try:
            url = f"https://www.fantasypros.com{pitchers[i+1]}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            theTable = soup.find_all('table')[3]
#        homeWhip = (float)((theTable.find_all('td')[-1].text))
#        try:    
            homeWhip = float(theTable.find_all('td')[-1].text) 
#        except ValueError: 
        except:
            print("Error so average used instead for homeWhip, likely no listed pitcher or no stats so risky to follow")
            homeWhip = avgwhip

        print(awayWhip)
        print(homeWhip)

        awayScore = 0
        batterNum = 0
        numOuts = 0
        oddsAtBatHappens = 0
        

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
        indexForOdds = [index for index,elem in enumerate(awayTeams) if awayTeam.split()[-1].startswith(elem)][0]             
#        print(indexForOdds)
        NRFIs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({odds[(2 * indexForOdds)+1]})"] + [homeScore + awayScore]) if homeScore + awayScore < 1 else YRFIs.append([indexForOdds] + [f"{awayTeam} @ {homeTeam}({odds[2 * indexForOdds]})"] + [homeScore + awayScore])
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
        i += 2
    
    NRFIs = sorted(NRFIs,key=lambda x: x[2],reverse=True)
    YRFIs = sorted(YRFIs,key=lambda x: x[2],reverse=True)

    print("YRFIs")
    for elem in YRFIs:
        print(elem)
    print("NRFIs")
    for elem in NRFIs:
        print(elem)
