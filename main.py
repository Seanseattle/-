import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

def loadData(featuresPath):
    feature = np.ndarray(shape=(0,2))
    label = np.ndarray(shape=(0,1))

    df = pd.read_table(featuresPath, delimiter=',', na_values='', header=0)
    #将读取到的DataFrame转化成数组进行处理
    matchList = np.array(df).tolist()
    matchInfoList = [] #处理后的feature数组
    labelList = [] #标签数组 暂时只考虑输赢
    for eachMatch in matchList:
        guestWinTime = int(eachMatch[2].split('胜')[0])
        guestLoseTime = eachMatch[2].split('胜')[1]
        guestLoseTime = int(guestLoseTime.split('负')[0])

        homeWinTime = int(eachMatch[3].split('胜')[0])
        homeLoseTime = eachMatch[3].split('胜')[1]
        homeLoseTime = int(homeLoseTime.split('负')[0])

        guestScore = int(eachMatch[4].split(':')[0])
        homeScore = int(eachMatch[4].split(':')[1])

        array = []
        #array.append(float(guestWinTime - guestLoseTime))
        array.append(guestWinTime)
        array.append(guestLoseTime)
        #array.append(float(homeWinTime - homeLoseTime))
        array.append(homeWinTime)
        array.append(homeLoseTime)

        matchInfoList.append(array)

        if(guestScore > homeScore):
            labelList.append(1)
        else:
            labelList.append(0)
    #处理结束
    matchInfoList = pd.DataFrame(matchInfoList)
    labelList = pd.DataFrame(labelList)
    labelList = np.ravel(labelList) #规整为一维向量
    #feature = np.concatenate(feature, matchInfoList)
    #label = np.concatenate(label, labelList)
    return matchInfoList, labelList





if __name__ == '__main__':
    feature, label = loadData('matchDataTrain.csv')
    '''
    winx = []
    winy = []
    losex = []
    losey = []
    i = 0

    for each in label:
        if(each == 1):
            winx.append(feature[0][i])
            winy.append(feature[1][i])
        else :
            losex.append(feature[0][i])
            losey.append(feature[1][i])
        i += 1
    plt.scatter(winx,winy,c='black')
    plt.scatter(losex,losey,c='red')
    plt.show()
    '''

    x_train, y_train = feature[:5000], label[:5000]
    x_test, y_test = feature[5000:], label[5000:]
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)
    print('start trainning')

    knnModel = KNeighborsClassifier().fit(x_train, y_train)
    print('start test')
    answerKnn = knnModel.predict(x_test)
    print('done')
    print('The classification report for knn:')
    print(classification_report(y_test, answerKnn))
