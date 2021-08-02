import json
import requests

data = {
    "context": 'In 1954, the Yankees won over 100 games, but the Indians took the pennant with an AL record 111 wins; 1954 was famously referred to as "The Year the Yankees Lost the Pennant". In , the Dodgers finally beat the Yankees in the World Series, after five previous Series losses to them, but the Yankees came back strong the next year. On October 8, 1956, in Game Five of the 1956 World Series against the Dodgers, pitcher Don Larsen threw the only perfect game in World Series history, which remains the only perfect game in postseason play and was the only no-hitter of any kind to be pitched in postseason play until Roy Halladay pitched a no-hitter on October 6, 2010.',
    "question": "Who was the winning pitcher in the 1956 World Series?",
    "idx": "0"
}

res = requests.post("http://xx.xx.xx.xx:8088/mrc",
                    data=json.dumps(data),
                    headers={"Content-Type": "application/json"})

a = json.loads(res.text)
print(a)
