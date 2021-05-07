import pandas as pd
import numpy as np
from sklearn import datasets
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
#
# Import Keras modules
#
from keras import models
from keras import layers
from keras.utils import to_categorical

def encode_workarea():
    jobWorkAreaTable = pd.read_csv("FinalJobDetails.csv", index_col=False, header=0, usecols=['Work Area'])

    jobWorkAreaTable["Work Area"] = jobWorkAreaTable["Work Area"].astype('category')

    jobWorkAreaTable["Work Area Category"] = jobWorkAreaTable["Work Area"].cat.codes

    dictWorkArea = dict( enumerate(jobWorkAreaTable['Work Area'].cat.categories ) )
    jobWorkAreaArray = jobWorkAreaTable.to_numpy()

    y = jobWorkAreaArray[:, 1]

    for i in range(len(y)):
        if y[i] == 9 or y[i] == 16 or y[i] == 18 \
            or y[i] == 1 or y[i] == 3 or y[i] == 5 \
            or y[i] == 10 or y[i] == 14 or y[i] == 21 \
            or y[i] == 8 or y[i] == 11 or y[i] == 15 \
            or y[i] == 20 or y[i] == 22 or y[i] == 23:
            y[i] = -1
            jobWorkAreaArray[i][0] = "SAP"

    yw = jobWorkAreaArray[:, 0]
    for i in range(len(yw)):
        if isinstance(yw[i], float):
            yw[i] = "SAP"
    classes, y_indices = np.unique(yw, return_inverse=True)
    n_classes = classes.shape[0]

    class_counts = np.bincount(y_indices)
    #print (classes)
    #print (n_classes)
    #print (class_counts)
   
    jobWorkAreaTable["Work Area"] = y
    jobWorkAreaTable["Work Area"] = jobWorkAreaTable["Work Area"].astype('category')
    #print (jobWorkAreaTable.dtypes)
    jobWorkAreaTable["Work Area Category"] = jobWorkAreaTable["Work Area"].cat.codes
    jobWorkAreaArray = jobWorkAreaTable.to_numpy()

    jobWorkAreaArray = jobWorkAreaArray[:, 1]
    y = jobWorkAreaArray
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]

    class_counts = np.bincount(y_indices)
    #print("y values")
    #print (classes)
    #print (n_classes)
    #print (class_counts)

    classes, y_indices = np.unique(yw, return_inverse=True)
    n_classes = classes.shape[0]

    class_counts = np.bincount(y_indices)
    #print("yw values")
    #print (classes)
    #print (n_classes)
    #print (class_counts)

    return jobWorkAreaArray, dictWorkArea, classes
    #return y 


def encode_jobsegment():
    jobSegmentTable = pd.read_csv("FinalJobDetails.csv", index_col=False, header=0, usecols=['Job Segment'])

    jSegmentList = []
    jobSegDict = {}
    for index in range(len(jobSegmentTable)):
        jobSegString = jobSegmentTable["Job Segment"][index]
        if isinstance(jobSegString, float):
            jobSegString = "SAP"
        jobCategoriesList = jobSegString.split(",")
        jobSegDict[index] = jobCategoriesList
        #print (len(jobCategoriesList))

    dataF = pd.DataFrame.from_dict(
            { 'categories': jobSegDict })

    from collections import Counter
    dataFnew = dataF['categories'].apply(Counter)
    jobSegList = pd.DataFrame.from_records(dataFnew).fillna(value=0)

    jobSegArray = jobSegList.to_numpy()

    return jobSegArray

if __name__=="__main__":
    inputData = encode_jobsegment()
    outputData, dictWorkArea, classesWorkArea = encode_workarea()

    print ("length of input: {} and output: {}".format(len(inputData), len(outputData)))

    X = inputData
    y = outputData
    from sklearn.utils import shuffle

    X, y = shuffle(X, y)

    #print (X.shape, y.shape)
    #print (X[0:2])
    #print (y)
    tempy = y
    #
    # Create training and test split
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    from sklearn.decomposition import PCA
    # Make an instance of the Model
    pca = PCA(.95)

    pca.fit(X_train)
    print ("Number of PCA comps: ",pca.n_components_)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    #
    # Create categorical labels
    #
    train_labels = to_categorical(y_train)
    test_labels = to_categorical(y_test)
    #print (len(train_labels[0]))
    #print (test_labels)

    from sklearn.model_selection import StratifiedKFold
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    X = pca.transform(X)
    #y = pca.transform(outputData)

    #y = outputData
    y = to_categorical(y)
    # define 10-fold cross validation test harness
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, y):
        # Create the network
        network = models.Sequential()
        #network.add(layers.Dense(512, activation='relu', input_shape=(278,)))
        network.add(layers.Dense(512, activation='relu', input_shape=(pca.n_components_,)))
        network.add(Dropout(0.3))
        network.add(layers.Dense(256, activation='relu'))
        network.add(Dropout(0.3))
        network.add(layers.Dense(10, activation='softmax'))
        #
        # Compile the network
        #
        #network.compile(optimizer='rmsprop',
        #sgd = SGD(lr=0.01, momentum=0.9)
        network.compile(optimizer='adam',
        #network.compile(optimizer=sgd,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        #
        # Fit the neural network
        #
        #network.fit(X_train, train_labels, epochs=50, batch_size=40)
        network.fit(X[train], y[train], epochs=30, batch_size=32)

        #
        # Get the accuracy of test data set
        #
        test_loss, test_acc = network.evaluate(X[test], y[test])

        #
        # Print the test accuracy
        #
        print('Test Accuracy: ', test_acc, '\nTest Loss: ', test_loss)
        cvscores.append(test_acc * 100)

        from sklearn.metrics import classification_report

        y_pred = network.predict(X[test], batch_size=32, verbose=1)
        y_pred_bool = np.argmax(y_pred, axis=1)

        from sklearn.metrics import precision_recall_fscore_support as score

        precision, recall, fscore, support = score(tempy[test], y_pred_bool)

        pdf = pd.DataFrame(precision)
        pdf.columns = ["Precision"]
        rdf = pd.DataFrame(recall)
        rdf.columns = ["Recall"]
        fdf = pd.DataFrame(fscore)
        fdf.columns = ["F1 Score"]
        sdf = pd.DataFrame(support)
        sdf.columns = ["Support"]
        wdf = pd.DataFrame(classesWorkArea)
        wdf.columns = ["Labels"]
        classificationReport = pd.concat([wdf, pdf, rdf, fdf, sdf], axis=1)

        classificationReport.to_csv("ClassificationReport.csv", header=True)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
