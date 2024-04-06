import requests
from bs4 import BeautifulSoup
'''
url = 'https://fantasydata.com/mlb/daily-lineups'
response = requests.get(url)
print(response)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    tag_with_lineups = soup.find_all('script', string=lambda x: x and 'var app = angular.module("fantasydata")' in x)[3]
    
    tag_with_lineups_as_str = str(tag_with_lineups)
    tag_with_lineups_as_str = tag_with_lineups_as_str.split("Games")[1]
    tag_with_lineups_as_str = tag_with_lineups_as_str.split("var LineupsController")[0]
    teams = tag_with_lineups_as_str.split("FullName")[1:]
    teams = [elem[3:elem.find(",")][:-1] for elem in teams]
    pitchers = tag_with_lineups_as_str.split("ProbablePitcherDetails")[1:]
    pitchers = [elem[31:elem.find("FirstInitial")-3] for elem in pitchers]
    lineups = tag_with_lineups_as_str.split("BattingOrderConfirmed")[1:]
    ''''''
    lineups = [elem[45:elem.find("FirstInitial")-3] for elem in lineups]


    num = 1
    for elem in lineups:
        tname = elem
        print(tname)
        num += 1
        if num % 9 == 0:
            print()
    #print(tag_with_lineups_as_str)
'''
'''
url = 'https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow'
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
#    pitchers = soup.find_all(class_="lineup__player-highlight-name")
    #[-1] needed if factor in throwing hand of pitcher
#    pitchers = [" ".join(elem.text.split()[:-1]) for elem in pitchers]

    links = soup.find_all("a")[:-54]
    links = links[482:]
    # [x:x+20] = link with teams names, away pitcher,away lineup, home pitcher,home team
    print(len(links))


#    print(lineups[0])
    teams = []
    pitchers = []
    lineups = []
    for index in range(0,len(links),20):
        a = [elem.text for elem in links[index].find_all("div")]
        print(a)
#        blue_jays = team_names[0].strip().split()[0]
  #      yankees = team_names[1].strip().split()[0]

 #       print("Blue Jays:", blue_jays)
#        print("Yankees:", yankees)
        #teams.append()

#        print(elem)
#        print()
'''
'''
    tag_with_lineups_as_str = str(tag_with_lineups)
    tag_with_lineups_as_str = tag_with_lineups_as_str.split("Games")[1]
    tag_with_lineups_as_str = tag_with_lineups_as_str.split("var LineupsController")[0]
    teams = tag_with_lineups_as_str.split("FullName")[1:]
    teams = [elem[3:elem.find(",")][:-1] for elem in teams]
    lineups = tag_with_lineups_as_str.split("BattingOrderConfirmed")[1:]
'''

#For Github
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
}

url = 'https://www.rotowire.com/baseball/daily-lineups.php'
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    #    pitchers = soup.find_all(class_="lineup__player-highlight-name")
    #[-1] needed if factor in throwing hand of pitcher, if found with above line
    #    pitchers = [" ".join(elem.text.split()[:-1]) for elem in pitchers]

    links = soup.find_all("a")[:-54]
    links = links[477:]
    # [x:x+23] = link with teams names, away pitcher,away lineup, home pitcher,home team,2 links for tickets then alerts

    teams = []
    pitchers = []
    lineups = []
    print(len(links))
    for index in range(0,len(links),23):
        teamsInMatchup = ["".join(elem.text.strip().split()[:-1]).lower() for elem in links[index].find_all("div")]
        teams.extend(teamsInMatchup)
        #Get whip      
        url = f"https://www.rotowire.com/{links[index+1].get('href')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        print(links[index+1])
        print(soup.find_all(class_="p-card__stat-value"))
        pitchers.append([elem.text for elem in links[index+1]] + [soup.find_all(class_="p-card__stat-value")[2].text])      
        lineups.append([elem.get('title') for elem in links[index+2:index+11]])
        #Get whip      
        url = f"https://www.rotowire.com/{links[index+11].get('href')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        pitchers.append([elem.text for elem in links[index+11]] + [soup.find_all(class_="p-card__stat-value")[2].text])      
        lineups.append([elem.get('title') for elem in links[index+12:index+21]])

pitchersThenFIStats = []
teamsThenFIStats = []
url = 'https://sports.betmgm.com/en/blog/mlb/nrfi-yrfi-stats-records-no-runs-first-inning-yes-runs-first-inning-runs-mlb-teams-bm03/'
response = requests.get(url,headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    #Pitcher first inning stats
    theTable = soup.find_all('table')[3]
    rows = theTable.find_all('tr')
    for row in rows:
        curRow = [elem.text for elem in row.find_all('td')]
        #if curRow[0] in pitchers:
        pitchersThenFIStats.extend(curRow)
 
    #Team first inning scoring stats, [2] would be team runs allowed but I think just will use pitcher specific stats
    #[1] is teams 1st inning scoring stats for now I think will use in addition to batters stats since 1st inning batters generally same dudes each game
    #Team first inning stats
    theTable = soup.find_all('table')[1]
    rows = theTable.find_all('tr')
    for row in rows:
        curRow = [elem.text for elem in row.find_all('td')]
        #if curRow[0] in pitchers:
        teamsThenFIStats.extend(curRow)
    print(pitchers)
    print(teamsThenFIStats)
