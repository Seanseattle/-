import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import preprocessing
def readFile():
    with open("matchDataTrain.csv") as csvFile:
        spamreader = pd.read_csv(csvFile, sep=',')
        min_max = preprocessing.MinMaxScaler()
        # data_list = spamreader.columns.tolist()
        # print(spamreader.dtypes)
        # print(data_list)
        # train_data = np.empty_like(spamreader)
        train_data = np.zeros(shape=(spamreader.__len__(),7))
        teamList = readTeamData("teamData.csv")
        i = 0
        while (i < spamreader.__len__()):
            string = spamreader['客场本场前战绩'][i]
            split_string = ''
            for char in string:
                if (char == '胜'):
                    train_data[i][0] = int(split_string)
                    split_string = ''
                elif (char == '负'):
                    train_data[i][1] = int(split_string)
                    split_string = ''
                else:
                    split_string += char
            string = spamreader['主场本场前战绩'][i]
            split_string = ''
            for char in string:
                if (char == '胜'):
                    train_data[i][2] = int(split_string)
                    split_string = ''
                elif (char == '负'):
                    train_data[i][3] = int(split_string)
                    split_string = ''
                else:
                    split_string += char
            string = spamreader['比分（客场:主场）'][i]
            split_string = ''
            for char in string:
                if (char == ":"):
                    a = int(split_string)
                    split_string = ''
                else:
                    split_string += char
            b = int(split_string)
            # train_data[i][8] = int(b/(a+b)*10)

            Ateam_Name = int(spamreader["客场队名"][i])
            Bteam_Name = int(spamreader["主场队名"][i])
            train_data[i][4] = teamList[Ateam_Name]
            train_data[i][5] = teamList[Bteam_Name]

            # train_data[i][6] = teamList[Bteam_Name][0]
            # train_data[i][7] = teamList[Bteam_Name][1]

            if a>b:
                train_data[i][6] = 1
            else:
                train_data[i][6] = 0
            i = i + 1
        print(train_data[6])
        train_data = min_max.fit_transform(train_data)

        start1 = 0
        end1 = 6300
        start2 = 0
        end2 = 6

        x, y = train_data[start1:end1,start2:end2].tolist(), train_data[start1:end1,end2].tolist()
        x_test, y_test = train_data[end1:,start2:end2].tolist(), train_data[end1:, end2].tolist()

        # gbn = GaussianNB()
        gbn = BernoulliNB()
        # gbn = MultinomialNB()
        # gbn = tree.DecisionTreeClassifier()
        y_pred = gbn.fit(x,y).predict(x_test)
        print("done")
        # print((y_pred != y_test).sum(),y_test.__len__)
        print(classification_report(y_test,y_pred))

        # print("start train")
        # print(type(y))
        # print(y)
        # knnModel = KNeighborsClassifier().fit(x, y)
        # print("start test")
        # result = knnModel.predict(x_test)
        # print("done")
        # print(classification_report(y_test, result))


def readTeamData(file):
    with open(file) as teamData:
        personData = pd.read_csv(teamData, sep=',')
        personList = np.array(personData).tolist()
        teamList = []
        ##计算球队的球员的per值，得到球队的总体实力
        temp = [0, 0]
        i = 0
        count = 0
        print(type(personData))
        # while(i<personData.__len__()):
        #     if(i == personData.__len__()-1 ):
        #         if (personData["投篮出手次数"][i] != 0.0 ):
        #             temp[0] += float(personData['投篮命中率'][i].strip("%")) / 100
        #             temp[0] = temp[0] / (count + 1)
        #             count = 0
        #             temp[1] += float(personData['投篮出手次数'][i])
        #             teamList.append([temp[0], temp[1]])
        #             break
        #         else:
        #             count = 0
        #     if (personData['队名'][i] == personData['队名'][i+1]):
        #         if(personData['投篮出手次数'][i] != 0.0):
        #             temp[0] += float(personData['投篮命中率'][i].strip("%"))/100
        #             temp[1] += float(personData['投篮出手次数'][i])
        #         count += 1
        #     else:
        #         if(personData["投篮出手次数"][i] != 0.0):
        #             temp[0] += float(personData['投篮命中率'][i].strip("%"))/100
        #             temp[0] = temp[0] / (count+1)
        #             count = 0
        #             temp[1] += float(personData['投篮出手次数'][i])
        #         else:
        #             count = 0
        #
        #         teamList.append([temp[0],temp[1]])
        #         temp[0] = 0
        #         temp[1] = 0
        #     i += 1
        teamNumber = 0
        per = 0
        for eachPerson in personList:
            if eachPerson[0] == teamNumber:
                per = ((eachPerson[22]+eachPerson[17]+eachPerson[14]+eachPerson[18]+eachPerson[19])-(eachPerson[7]-eachPerson[6])-(eachPerson[13]-eachPerson[12])-eachPerson[20])*eachPerson[4]/240
                count += per
            else:
                teamNumber += 1
                per = ((eachPerson[22]+eachPerson[17]+eachPerson[14]+eachPerson[18]+eachPerson[19])-(eachPerson[7]-eachPerson[6])-(eachPerson[13]-eachPerson[12])-eachPerson[20])*eachPerson[4]/240
                teamList.append(count+per)
                count = 0
        teamList.append(count+per)
        return teamList

# def dimensionReduction(filepath):
#     raw_data = pd.read_csv(filepath, header=0) #header 肯定是0啊。。
#
#     mx = raw_data.as_matrix()
#
#     for i in [5,8,11]:
#         percent = mx[:,i]
#         for j in range(0,len(percent)):
#             s=percent[j]
#             if isinstance(s, str):
#                 s = s[0:len(s)-2]
#                 s = round(float(s)*0.01, 3)
#             percent[j] = s
#
#         mx[:,i]=percent
#
#
#     #mx=np.nan_to_num(mx)
#     imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#     imp.fit(mx)
#     mx = imp.transform(mx)
#
#     #取出每个队伍的信息
#     teamNumber = 0
#     teamList = []
#     thisTeam = []
#     for eachPerson in mx:
#         if eachPerson[0] == teamNumber:
#             thisTeam.append(eachPerson)
#         else:
#             teamNumber += 1
#             teamList.append(thisTeam)
#             thisTeam = []
#             thisTeam.append(eachPerson)
#     teamList.append(thisTeam)
#
#     for eachTeam in teamList:
#         eachTeam.sort(key = lambda x:x[7], reverse=True) #以首发次数为标准排序3 上场时间4
#
#     featureList = []
#     for eachTeam in teamList:
#         sum = 0.0
#         for i in range(0,10):
#             sum += eachTeam[i][5] #-20效果不错 准确率提高1%
#         sum = sum / 5.0
#         featureList.append(sum)
#     return featureList


def train_model(train_data):
    start1 = 0
    end1 = 6000
    start2 = 0
    end2 = 7
    x,y = train_data[start1:end1][start2:end2],train_data[start1:end1][8]
    x_test,y_test= train_data[end1:][start2:end2], train_data[start1:][8]
    print("start train")
    knnModel = KNeighborsClassifier().fit(x,y)
    print("start test")
    result = knnModel.predict(x_test)
    print("done")
    print(classification_report(y_test, result))




if __name__ == '__main__':
    readFile()
    # readTeamData("teamData.csv")