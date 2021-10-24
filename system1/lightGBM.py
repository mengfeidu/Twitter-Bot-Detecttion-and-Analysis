from os import replace
import numpy as np
import datetime
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import csv
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def del_attr(a,b,l):
    for row in l:
        del(a[row])
        del(b[row])

def grid_search_cv(estimator, params, X_train, y_train, scoring='f1', cat_feature_index=None,returnBestParam=False):
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import GridSearchCV

    # grid = GridSearchCV(estimator, params, cv=LeaveOneOut(), scoring=scoring, n_jobs=-1, iid=False)
    grid = GridSearchCV(estimator, params, cv=5, scoring=scoring)

    if cat_feature_index:
        grid.fit(X_train, y_train, cat_features=cat_feature_index)
    else:
        grid.fit(X_train, y_train)

    print("best parameters:", grid.best_params_, "best score:", grid.best_score_)
    print ("best estimator:", grid.best_estimator_)
    if returnBestParam:
        return grid.best_estimator_, grid.best_params_
    else:
        return grid.best_estimator_

def lightGBM(X_train, y_train):
    import lightgbm as lgb

    params = {
        'learning_rate': [0.001,0.01,0.1],
        'n_estimators': [400,500,600],
        'num_leaves': range(4,16,4),
        # 'colsample_bytree' : [i/10.0 for i in range(1,10)],
        # 'subsample' : [i/10.0 for i in range(1,10)],
        'reg_alpha' : [0,0.5,1,1.2],
        'reg_lambda' : [0,0.5,1,1.2,1.4],
        }
    estimator = lgb.LGBMClassifier(
        num_class = 1,
        learning_rate=0.05,
        n_estimators=222,
        num_leaves=55,
        max_depth=7,
        colsample_bytree=0.5,
        subsample=0.5,
        reg_alpha=0,
        reg_lambda=0,
        boosting_type='gbdt',
    )
    print ("------------ lightGBM ------------")
    clf,best_param1 = grid_search_cv(estimator, params,X_train, y_train, returnBestParam=True)
    feature, NF = feature_importance(feature_name, clf.feature_importances_)
    return clf, feature, NF

def train(func, X_train, y_train, save=False):
    import joblib
    if func == 'lightgbm':
        clf = lightGBM(X_train, y_train)
    elif func == 'RF':
        clf = RF(X_train, y_train)
    if save:
        joblib.dump(clf, "model_"+ func + ".m")

    return clf

def predict(clf, X_test):
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    y_pred_prob = [row[1] for row in y_pred_prob]
    return y_pred,y_pred_prob


def test(func,X_test, y_test, clf):
    y_pred,y_pred_prob = predict(clf, X_test)
    res =  evaluation(y_test, y_pred,y_pred_prob)
    return res

def evaluation(y_true, y_pred,y_pred_prob):
    from sklearn import metrics

    acc = metrics.accuracy_score(y_true, y_pred)
    print ("Accuracy:", acc)
    auc = metrics.roc_auc_score(y_true, y_pred_prob,average='micro')
    print ("AUC:", auc)
    print (metrics.classification_report(y_true, y_pred, digits=4))

    precision = metrics.precision_score(y_true, y_pred,pos_label= 1)
    recall = metrics.recall_score(y_true, y_pred,pos_label= 1)
    f1 = metrics.f1_score(y_true, y_pred,pos_label= 1)

    weighted_precision = metrics.precision_score(y_true, y_pred, average='weighted')
    weighted_recall = metrics.recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    return auc,f1,acc, auc, precision, recall, f1, weighted_precision, weighted_recall, weighted_f1

def feature_importance(feature_name, importance):
    feature = {}
    for i in range(len(feature_name)):
        feature[feature_name[i]] = importance[i]

    feature = sorted(feature.items(), key=lambda x: x[1], reverse=True)

    NF = 0
    for i in range(len(feature)):
        NF = max(feature[i][1], NF)

    for i in range(len(feature)):
        if feature[i][1] > 0:
            print (feature[i][0], '->', feature[i][1]/NF)
        else:
            break
    print("\n")

    return feature, NF

if __name__ == "__main__":
    social_spambots_1 = open(r'social_spambots_1.csv\users.csv', 'r',encoding="utf8")
    social_spambots_2 = open(r'social_spambots_2.csv\users.csv', 'r',encoding="utf8")
    social_spambots_3 = open(r'social_spambots_3.csv\users.csv', 'r',encoding="utf8")
    genuine_accounts = open(r'genuine_accounts.csv\users.csv','r',encoding='utf8')

    social_spambots_1_reader = csv.reader(social_spambots_1)
    social_spambots_2_reader = csv.reader(social_spambots_2)
    social_spambots_3_reader = csv.reader(social_spambots_3)
    genuine_accounts_reader = csv.reader(genuine_accounts)

    user_metadata = ["statuses_count", "followers_count", "friends_count", "favourites_count",
        "listed_count", "default_profile", "profile_use_background_image", "verified"]
    derived_features = ["tweet_freq", "followers_growth_rate", "friends_growth_rate", "favourites_growth_rate", "listed_growth_rate", "followers_friends_ratio",
        "screen_name_length", "num_digits_in_screen_name", "name_length",
        "num_digits_in_name", "description_length"]

    feature_name = []
    headers = genuine_accounts_reader.__next__()
    for feature in headers:
        if feature in user_metadata or feature in derived_features:
            feature_name.append(feature)
    feature_name += derived_features
    social_spambots_1_reader.__next__()
    social_spambots_2_reader.__next__()
    social_spambots_3_reader.__next__()
    genuine_accounts_reader.__next__()

    reader_list = [
        social_spambots_1_reader,social_spambots_2_reader,
        social_spambots_3_reader,genuine_accounts_reader
    ]
    # 建两个list，featureList装特征值，labelList装类别标签
    featureList = []
    labelList = []

    featureList_test = []
    labelList_test = []

    # 构建虚假用户和真实用户字典
    users_label = dict()
    # 遍历虚假用户文件的每一行
    i = 0
    for reader in reader_list:
        if i == 0:
            for row in reader:
                users_label[row[0]] = (1, 1)
        elif i != 3:
            for row in reader:
                users_label[row[0]] = (1, 0)
        else:
            for row in reader:
                users_label[row[0]] = (0, 1)
        i += 1

    social_spambots_1 = open(r'social_spambots_1.csv\users.csv', 'r',encoding="utf8")
    social_spambots_2 = open(r'social_spambots_2.csv\users.csv', 'r',encoding="utf8")
    social_spambots_3 = open(r'social_spambots_3.csv\users.csv', 'r',encoding="utf8")
    genuine_accounts = open(r'genuine_accounts.csv\users.csv','r',encoding='utf8')

    social_spambots_1_reader = csv.reader(social_spambots_1)
    social_spambots_2_reader = csv.reader(social_spambots_2)
    social_spambots_3_reader = csv.reader(social_spambots_3)
    genuine_accounts_reader = csv.reader(genuine_accounts)

    reader_list = [
        social_spambots_1_reader,social_spambots_2_reader,
        social_spambots_3_reader,genuine_accounts_reader
    ]

    # 数据集的读取
    for reader in reader_list:
        reader_head = reader.__next__()
        index_list = []
        for i in range(len(reader_head)):
            if reader_head[i] in feature_name:
                index_list.append(i)
            if reader_head[i] == 'profile_use_background_image':
                background_index = i
            if reader_head[i] == 'verified':
                verified_index = i
            if reader_head[i] == 'updated':
                updated_index = i
            if reader_head[i] == 'crawled_at':
                crawled_index = i
            if reader_head[i] == 'timestamp':
                timestamp_index = i
            if reader_head[i] == 'statuses_count':
                statuses_index = i
            if reader_head[i] == 'followers_count':
                followers_index = i
            if reader_head[i] == 'friends_count':
                friends_index = i
            if reader_head[i] == 'favourites_count':
                favourites_index = i
            if reader_head[i] == 'listed_count':
                listed_index = i
            if reader_head[i] == 'screen_name':
                screen_index = i
            if reader_head[i] == 'name':
                name_index = i
            if reader_head[i] == 'description':
                description_index = i

        for row in reader:
            # 将类别标签加入到labelList中
            # profile_use_background_image 属性的清洗
            if row[background_index] == 'NULL':
                continue
            elif row[background_index] == '':
                row[background_index] = '0'

            # verified 属性的清洗
            if row[verified_index] == '':
                row[verified_index] = '0'

            # user_age 的计算
            temp = row[crawled_index]
            crawled_at = datetime.datetime.strptime(temp, '%Y-%m-%d %H:%M:%S')
            timestamp = datetime.datetime.strptime(row[timestamp_index], '%Y-%m-%d %H:%M:%S')
            user_age = (crawled_at-timestamp).days+1

            # 其他属性的计算
            tweet_freq = eval(row[statuses_index])/user_age
            followers_growth_rate = eval(row[followers_index])/user_age
            friends_growth_rate = eval(row[friends_index])/user_age
            favourites_growth_rate = eval(row[favourites_index])/user_age
            listed_growth_rate = eval(row[listed_index])/user_age
            followers_friends_ratio = eval(row[followers_index])/eval(row[friends_index]) if eval(row[friends_index]) != 0 else 0
            screen_name_length = len(row[screen_index])
            num_digits_in_screen_name = len("".join(list(filter(str.isdigit, row[screen_index]))))
            name_length = len(row[name_index])
            num_digits_in_name = len("".join(list(filter(str.isdigit, row[name_index]))))
            description_length = len(row[description_index])

            if row[0] in users_label:
                k = users_label[row[0]]
                if k[1] == 1:
                    labelList_test.append(k[0])
                labelList.append(k[0])
            # 下面这几步的目的是为了让特征值转化成一种字典的形式，就可以调用sk-learn里面的DictVectorizer，直接将特征的类别值转化成0,1值
            rowDict = {}
            for i in index_list:
                if row[i] == '':
                    rowDict[reader_head[i]] = 0.0
                else:
                    rowDict[reader_head[i]] = float(row[i])
            rowDict['tweet_freq'] = float(tweet_freq)
            rowDict['followers_growth_rate'] = float(followers_growth_rate)
            rowDict['friends_growth_rate'] = float(friends_growth_rate)
            rowDict['favourites_growth_rate'] = float(favourites_growth_rate)
            rowDict['listed_growth_rate'] = float(listed_growth_rate)
            rowDict['followers_friends_ratio'] = float(followers_friends_ratio)
            rowDict['screen_name_length'] = float(screen_name_length)
            rowDict['num_digits_in_screen_name'] = float(num_digits_in_screen_name)
            rowDict['name_length'] = float(name_length)
            rowDict['num_digits_in_name'] = float(num_digits_in_name)
            rowDict['description_length'] = float(description_length)

            if row[0] in users_label:
                k = users_label[row[0]]
                if k[1] == 1:
                    featureList_test.append(rowDict)
                featureList.append(rowDict)

    # 测试集的构建
    vec = DictVectorizer()

    real_index = []
    bot_index = []
    for i in range(len(labelList_test)):
        if labelList_test[i] == 0:
            real_index.append(i)
        else: bot_index.append(i)
    bot_index = list(np.random.choice(np.array(bot_index), size=len(bot_index)//2, replace=False))
    real_index = list(np.random.choice(np.array(real_index), size=len(real_index)//2, replace=False))

    new_featureList_test, new_labelList_test = [], []
    for i in real_index+bot_index:
        new_featureList_test.append(featureList_test[i])
        new_labelList_test.append(labelList_test[i])

    dummyX_test = vec.fit_transform(new_featureList_test).toarray()
    dummyY_test = list(map(int,new_labelList_test))

    ggg,x_test,gggg,y_test= train_test_split(dummyX_test, dummyY_test, test_size = 0.99)

    # 训练集
    vec = DictVectorizer()
    for k in bot_index+real_index:
        featureList.pop(k)
        labelList.pop(k)
    dummyX = vec.fit_transform(featureList).toarray()
    dummyY = list(map(int,labelList))
    x_train,g,y_train,gg= train_test_split(dummyX, dummyY, test_size = 0.01)

    # 分类器
    # lightgbm
    clf, feature, NF = train("lightgbm", x_train,y_train, True)
    res = test("lightgbm",x_test, y_test, clf)

    # 特征重要度可视化
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    name = [feature[i][0] for i in range(len(feature))]
    importance = [feature[i][1]/NF for i in range(len(feature))]

    item = [(name[i], importance[i]) for i in range(len(feature))]
    item.sort(key=lambda x:x[1], reverse=False)

    name = [item[i][0] for i in range(len(item))]
    importance = [item[i][1] for i in range(len(item))]

    plt.barh(name, importance)
    plt.title('Lightgbm model feature importance')

    plt.savefig('特征重要度.png', bbox_inches='tight')
    plt.show()
