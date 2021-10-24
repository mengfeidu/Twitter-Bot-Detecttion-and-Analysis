import os
import pandas as pd
import json

def get_user():
    dirlist = os.listdir(r'D:\tweet\1101_1\1101_1')

    twitter_name = {'1101':[]}
    for item in dirlist:
        try:
            with open(r'D:\tweet\1101\1101'+'\\'+item,'r',encoding = 'utf-8') as f:
                data = json.loads(f.read())

        except:
            continue

        twitter_name['1101'].append(data['user']['id_str'])

    with open(r'D:\pycharm_file\social-network\dataset\twitter_id.json','wb',encoding='utf-8') as f:
        json.dump(twitter_name,f)

users = pd.read_csv(r'D:\pycharm_file\social-network\dataset\final_network.csv',dtype=object)
test = users[0:1000]
test = test.drop('Unnamed: 0',axis = 1)


with open('test1.csv', 'w', encoding='utf-8') as f:
    for i in range(1000):
        f.write(test['o'][i]+','+test['r'][i]+'\n')



#
# tdir = os.listdir(r'D:\tweet\tweets')
#
# dir_lists = []
# for item in tdir:
#     dir_lists.append(set(os.listdir(r'D:\tweet\tweets' + '\\' + item)))
#
# for id in users:
#
#     for item in tdir:
#         if id in item:
#             break
#     with open(r'D:\tweet\tweets' + '\\' + item + '\\' + str(id) + '.json', 'r', encoding='utf-8') as f:
#         data = json.loads(f.read())
#     user_profile = data['user']
#     with open(r'D:\tweet\users@10000'+'\\'+id, 'wb', encoding='utf-8') as f:
#         json.dump(user_profile, f)

# with open(r'D:\pycharm_file\social-network\dataset\final_id.json','r') as f:
#     data = json.loads(f.read())