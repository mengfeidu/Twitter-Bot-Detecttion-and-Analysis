import pandas as pd

node = pd.read_csv(r'D:\pycharm_file\social-network\node.csv',low_memory=False,dtype = object)
edge = pd.read_csv(r'D:\pycharm_file\social-network\edge.csv',low_memory=False,dtype=object)



communty = pd.read_csv(r'D:\pycharm_file\social-network\node_label_asyn.csv',low_memory=False,dtype=object)

social_set = set()
for i in range(len(node)):
    if node['type'][i] != 'human':
        social_set.add(node['id'][i])

red_set = set()
blue_set = set()

for i in range(len(communty)):
    if communty['type'][i] == '0':
        blue_set.add(communty['id'][i])
    if communty['type'][i] == '1':
        red_set.add(communty['id'][i])

bot_edge = []
connect_edge = []
bot_human = []
human_bot = []

for i in range(len(edge)):
    if edge['source'][i] in social_set and edge['target'][i] in social_set:
        bot_edge.append((edge['source'][i],edge['target'][i]))
    if (edge['source'][i] in red_set and edge['target'][i] in blue_set) or (edge['source'][i] in blue_set and edge['target'][i] in red_set):
        connect_edge.append((edge['source'][i],edge['target'][i]))
    if edge['source'][i] in social_set and edge['target'][i] not in social_set:
        bot_human.append((edge['source'][i],edge['target'][i]))
    if edge['source'][i] not in social_set and edge['target'][i] in social_set:
        human_bot.append((edge['source'][i],edge['target'][i]))

human_edge = []
for i in range(len(edge)):
    if edge['source'][i] not in social_set and edge['target'][i] not in social_set:
        human_edge.append((edge['source'][i],edge['target'][i]))

red  = 0
blue = 0
for item in connect_edge:
    if item[0] in red_set:
        red += 1
    else:
        blue +=1

setin = list(set(bot_human).intersection(set(connect_edge)))
human = set()
for item in setin:
    human.add(item[1])
info = 0
for item in human_edge:
    if item[0] in human:
        info += 1
