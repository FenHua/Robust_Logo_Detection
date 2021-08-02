import json
import numpy as np

# 完成网络预测结果的置信度拉伸
data=json.load(open('results.json'))   # 输入网络检测结果json格式
res=[]
scomin,scomax = 2,0
a,b = 0.5,1

'''
filter_results = []
threshold = np.load('thr.npy')
for i in data:
    if i['score']>=threshold[(i['category_id']-1)]:
        filter_results.append(i)
'''
filter_results = data

for i in filter_results:
    scomin=min(scomin,i['score'])
    scomax=max(scomax,i['score'])
k = 0.5/(scomax-scomin)

for i in filter_results:
    i['score']=a+k*(i['score']-scomin)
    if i['score']<=0.5:
        i['score']=0.5
    if i['score']>1:
        i['score']=1
    res.append(i)

# 将置信度修改后的json保存
with open('resultsB.json','w') as f:
    json.dump(res,f)