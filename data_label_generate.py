a = ['0','1']
b = ['0','1','2','3']
c = '''短袖上衣
长袖上衣
短袖衬衫
长袖衬衫
背心上衣
吊带上衣
无袖上衣
短外套
短马甲
长袖连衣裙 
短袖连衣裙 
无袖连衣裙
长马甲
长外套
连体衣
古风
短裙
中等半身裙
长半身裙
短裤
中裤
长裤
背带裤'''.split()
a_ = {'0':'纯商品展示','1':'试穿展示'}
b_ = {'0':'正面','1':'背面','2':'左侧','3':'右侧'}

labels = {}
for i in a:
    for j in b:
        for k in c:
            name = i+'-'+j+'-'+k
            if name not in labels.keys():
                labels[name]={len(labels.keys()):a_[i]+'-'+b_[j]+'-'+k}

import json
with open('labels.json', 'w') as f:
    json.dump(labels, f,ensure_ascii=False,indent=4)
json.load(open('labels.json'))


labels = {}
for i in a:
    for j in b:
        for k in c:
            name = i+'-'+j+'-'+k
            if name not in labels.keys():
                labels[len(labels.keys())]=a_[i]+'-'+b_[j]+'-'+k

import json
with open('labels_.json', 'w') as f:
    json.dump(labels, f,ensure_ascii=False,indent=4)
json.load(open('labels_.json'))


with open('label_display.json', 'w') as f:
    json.dump(a_, f,ensure_ascii=False,indent=4)

with open('label_viewpoint.json', 'w') as f:
    json.dump(b_, f,ensure_ascii=False,indent=4)


c_ = {}
for i,j in zip(range(23),c):
    c_[str(i)] = j
with open('label_label.json', 'w') as f:
    json.dump(c_, f,ensure_ascii=False,indent=4)