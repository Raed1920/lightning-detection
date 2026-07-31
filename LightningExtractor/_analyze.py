import json
with open('output/VID_20250828_193327/report.json') as f:
    data = json.load(f)
confs = [d['confidence'] for d in data['detections']]
confs.sort()
print('Total:', len(confs))
print('Min:', round(confs[0],2), 'Max:', round(confs[-1],2))
for thresh in [1,2,3,5,10,15]:
    cnt = sum(1 for c in confs if c < thresh)
    print(f'  conf < {thresh}: {cnt}')
print()
print('Low confidence details (conf < 3):')
for d in data['detections']:
    if d['confidence'] < 3:
        print(f'  #{d["index"]:3d}  frame={d["frame"]:4d}  conf={d["confidence"]:.2f}  bright={d["brightness_score"]:.2f}  edge={d["edge_score"]:.2f}  ts={d["timestamp"]}')
