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

def readFile():
    with open("matchDataTrain.csv") as csvFile:
        spamreader = pd.read_csv(csvFile, sep=',')
        data_list = spamreader.columns.tolist()
        print(spamreader.dtypes)
        print(data_list)
        train_data = np.empty_like(spamreader)
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
            string = spamreader['比分'][i]
            split_string = ''
            for char in string:
                if (char == ":"):
                    a = int(split_string)
                    split_string = ''
                else:
                    split_string += char
            b = int(split_string)
            # train_data[i][4] = int(b/(a+b)*10)
            if a>b:
                train_data[i][4] = 1
            else:
                train_data[i][4] = 0
            i = i + 1

        x, y = train_data[0:6000, 0:4].tolist(), train_data[0:6000,4].tolist()
        x_test, y_test = train_data[6000:, 0:4].tolist(), train_data[6000:, 4].tolist()

        # gbn = GaussianNB()
        # gbn = BernoulliNB()
        gbn = MultinomialNB()
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
        personList = personData.columns.tolist()
        print(personList)
        teamList = []
        templist = [0,0]
        i = 0
        count = 0
        print(personList)
        while(i<personData.__len__()-1):
            if (personData['队名'][i] != personData['队名'][i+1]):
                templist[0] += personData['投篮命中率'][i]
                templist[0] = templist[0]/count
                templist[1] += personData['投篮出手次数'][i]
                templist[1] = templist[1] / count
                teamList.append(templist)
                templist[0] = 0
                teamList[1] = 0
            else:
                templist[0] += personData['投篮命中率'][i]
                templist[1] += personData['投篮出手次数'][i]
            i += 1
        print(teamList)




def train_model(train_data):
    x,y = train_data[0:6000][0:3],train_data[0:6000][4]
    x_test,y_test= train_data[6000:][0:3], train_data[6000:][4]
    print("start train")
    knnModel = KNeighborsClassifier().fit(x,y)
    print("start test")
    result = knnModel.predict(x_test)
    print("done")
    print(classification_report(y_test, result))




if __name__ == '__main__':
    readFile()
    # readTeamData("teamData.csv")