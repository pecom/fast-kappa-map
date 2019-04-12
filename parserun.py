import bettermap
import json

srad = []
midpoint = []
Nside = []
sigfactor = []

with open('config.json') as f:
    data = json.load(f)


srad = data['srad']
midpoint = data['midpoint']
Nside = data['Nside']
sigfactor = data['sigfactor']

if(data['range'] == 0):
    print("Running on")
    print(data)
    i = len(srad)
    for k in range(i):
        bettermap.Analysis(srad[k]*.1, midpoint[k], Nside[k], sigfactor[k])

if(data['range'] == 1):
    print("Running on range")
    print(data)
    for i in srad:
        for j in midpoint:
            for k in Nside:
                for l in sigfactor:
                    bettermap.Analysis(i*.1, j, k, l)
        
