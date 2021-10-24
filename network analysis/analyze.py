import pandas as pd
import os
import json
from PIL import Image
import numpy as np
from wordcloud import WordCloud,ImageColorGenerator
from matplotlib import pyplot as plt
nodes = pd.read_csv(r'D:\pycharm_file\social-network\node_label_asyn.csv',dtype = object)

type_3 = []
type_1 = []
type_0 = []
for i in range(len(nodes)):
    if nodes['type'][i] == '0':
        type_0.append(nodes['id'][i])
    if nodes['type'][i] == '1':
        type_1.append(nodes['id'][i])
node = pd.read_csv(r'D:\pycharm_file\social-network\node.csv',low_memory=False,dtype = object)
bot1 = set()
bot2 = set()
bot3 = set()
human = set()
for i in range(len(node)):
    if node['type'][i] == 'human':
        human.add(node['id'][i])
    elif node['type'][i] == 'socialSpam':
        bot1.add(node['id'][i])
    elif node['type'][i] == 'traditionalSocialSpam':
        bot2.add(node['id'][i])
    elif node['type'][i] == 'fakeFollower':
        bot3.add(node['id'][i])


num = 0
for i in range(len(type_1)):
    if type_1[1] in human:
       num += 1


def clean():
    usrid = os.listdir(r'D:\tweet\users\users')
    for i in range(len(usrid)):
        with open(r'D:\tweet\users\users'+'\\'+usrid[i],'r',encoding='utf-8') as f:
            data = json.loads(f.read())
        tmp = []
        for item in data['timeline']:
            if item not in tmp:
                tmp.append(item)
        data['timeline'] = tmp
        with open(r'D:\tweet\users\users'+'\\'+usrid[i], 'w', encoding='utf-8') as f:
            json.dump(data,f,indent=1)


def text_gengerate(ids):
    tweets = ''

    for i in range(len(ids)):
        with open(r'D:\tweet\users\users'+'\\'+ids[i]+'.json','r',encoding='utf-8') as f:
            data = json.loads(f.read())
        for item in data['timeline']:
            tweets += ' '+ item['text']
    return tweets


text = text_gengerate(type_0)

wc = WordCloud(max_words=100,
               background_color="white",
               stopwords={'https','This','co','t'},
               margin=10,
               random_state=1).generate(text)
default_colors = wc.to_array()
plt.imshow(default_colors,interpolation="bilinear")
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud2.png')
plt.show()
