import os
import json
import numpy as np
import networkx as nx
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
def cdf(data):
    ecdf = sm.distributions.ECDF(data)
    x = np.linspace(min(data), max(data))
    y = ecdf(x)
    plt.step(x, y)
    plt.show()

    fig,ax0 = plt.subplots(nrows=1,figsize=(6,6))
    ax0.hist(data,10,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
usrid = os.listdir(r'D:\tweet\users\users')

# with open(r'D:\pycharm_file\social-network\pre\index.json','r',encoding='utf-8') as f:
#     index = json.loads(f.read())


G = nx.Graph()
nodes = pd.read_csv(r'D:\pycharm_file\social-network\node.csv',dtype = object)
node = list(nodes['id'])
type = list(nodes['type'])

edges = pd.read_csv(r'D:\pycharm_file\social-network\edge.csv',dtype = object)
source = edges['source']
target = edges['target']

edge = []
weight = []
consider_set = set()
for i in range(len(source)):
    tmp = (source[i],target[i])
    if tmp not in consider_set:
        edge.append(tmp)
        weight.append({'weight': 1})
        consider_set.add(tmp)
    else:
        index = edge.index(tmp)
        weight[index]['weight'] += 1
weighted_edge = []
for i in range(len(edge)):
    weighted_edge.append((edge[i][0],edge[i][1],weight[i]['weight']))

G.add_weighted_edges_from(weighted_edge)

import community

socialspam = []
fake = []
human = []
for i in range(len(nodes)):
    if nodes['type'][i] == 'socialSpam':
        socialspam.append(nodes['id'][i])
    if nodes['type'][i] == 'fakeFollower':
        fake.append(nodes['id'][i])
    if nodes['type'][i] == 'human':
        human.append(nodes['id'][i])

def betweenness():
    cen = nx.algorithms.centrality.betweenness_centrality(G)
    socialspam_cen = []
    fake_cen = []
    human_cen = []
    for i in socialspam:
        socialspam_cen.append(cen[i])
    for i in fake:
        fake_cen.append(cen[i])
    for i in human:
        human_cen.append(cen[i])
cen = nx.algorithms.centrality.betweenness_centrality(G)




#partition = community.best_partition(G, partition=None, weight='weight', resolution=1.0)
#partition = list(nx.algorithms.community.asyn_lpa_communities(G,weight='weight'))

# nodes['label'] = [0] * len(nodes['id'])

# for i in range(len(nodes)):
#     for j in range(len(partition)):
#         if nodes['id'][i] in partition[j]:
#             nodes['label'][i] = j
#             break
#
# with open(r'D:\pycharm_file\social-network\node_label_asyn.csv','w',encoding='utf-8') as f:
#     f.writelines('id,type\n')
#     for i in range(len(nodes['id'])-1):
#         str1 = nodes['id'][i]+','+str(nodes['label'][i])+'\n'
#         f.writelines(str1)
#     str1 = nodes['id'][len(nodes['id'])-1]+','+str(nodes['label'][len(nodes['id'])-1])
#     f.writelines(str1)


# import numpy as np
# A=np.array(nx.adjacency_matrix(G).todense())
#
# for i in range(len(A[0])):
#     A[i,i] = sum(A[i,:])
# D = np.diag(np.diagonal(A))
#
# L = D - A

