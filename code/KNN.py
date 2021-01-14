import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# This class will pick the remaining 30% Documents and predict the class and then check prediction is correct or wrong
class TestingData:
    def __init__(self,n):
        self.labels = {"athletics": 101, "cricket": 124, "football": 265, "rugby": 147, "tennis": 100}
        self.Ratio = n
        self.TestVec = {}
        self.DatasetDic = {}
        self.Datasetpath = "../bbcsport/"
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.vocabulary = {}
        file=open("Vocabulary.txt","r")
        file=file.readlines()
        for x in file:
            x=x.split()
            self.vocabulary[x[0]] = x[1]
        self.features = list(self.vocabulary.keys())
        self.stop_words = stopwords.words('english')
        self.newFeature  = {}
        self.TotalDoc = 0



    def ReadTestingData(self):
        '''
        This function read the data from foldersby specfifed ratio of 70 /30 and do cleaning , extracting features
        :return:
        '''
        porter = PorterStemmer()

        for key, value in self.labels.items():
            dirPath = self.Datasetpath + key + "/"
            self.newFeature[key] = []
            NoOfDocRead = value * (self.Ratio/100)
            NoOfDocRead = round(NoOfDocRead)
            self.DatasetDic[key] = []
            print("READING DATA FROM FOLDER {0}...".format(key))
            self.TotalDoc = self.TotalDoc + (value - NoOfDocRead )
            for i in range(NoOfDocRead + 1,value+1):
                j = str(i)
                j = j.zfill(3)
                path = dirPath + j + ".txt"
                text = ""
                with open(path) as f:
                    file = f.read()

                for x in file:
                    if x in self.punctuations:
                        file = file.replace(x, " ")
                text = file.lower()
                words = text.split()
                data = ""
                lis = []
                for x in words:
                    stem = porter.stem(x)
                    data = data + stem + " "
                    if any(map(str.isdigit, stem)) == False and stem not in self.stop_words:
                        if stem not in self.features:
                            lis.append(stem)
                self.newFeature[key].append(lis)
                self.DatasetDic[key].append(data)


    def VectorRepresentation(self):
        '''
              This function perform 2 tasks
              1 > Create vectors of training data
              2 > find cosine similarity to each trained data , pick the top 3 that are most related and classify document accordingly
              :return:
        '''
        print("CREATING VECTORS OF TEST DATA...")
        newidf = math.log2(self.TotalDoc/1)
        for key, value in self.DatasetDic.items():

            self.TestVec[key] = []
            counter = 0
            for x in value:
                lis = []
                for word,idf in self.vocabulary.items():   #From vocabulary file pick dword and its respective calculated IDF
                    tf = x.count(word)
                    if tf > 0:
                        tf = 1 + math.log2(tf)
                    tf_idf = float(idf) * tf
                    lis.append(tf_idf)


                for word in self.newFeature[key][counter]:          #New features of testing data that are not exist in train data . add smooting 1 to DF
                     tf = x.count(word)
                     if tf >0:
                         tf = 1 + math.log2(tf)

                     tf_idf = tf * newidf
                     lis.append(tf_idf)

                counter = counter + 1
                self.TestVec[key].append(lis) #Vectors of test data created

    def CosineSimilarity(self,value):
        classes = ['athletics','cricket','football','rugby','tennis']
        vec1 = np.array(value)
        normVec1 = np.linalg.norm(vec1)
        max1 = 0
        max2 = 0
        max3 = 0
        max1Class =''
        max2Class =''
        max3Class= ''
        for i in range(5):
            file =open(classes[i]+".txt","r")
            file=file.readlines()
            for x in file:
                x1=list(map(float,x.split()))
                Zeros = len(value) - len(x1)
                for j in range(Zeros):
                    x1.append(0)
                trainedvec = np.array(x1)
                normTvec = np.linalg.norm(trainedvec)
                dot = np.dot(vec1,trainedvec)

                cos = dot / (normVec1 * normTvec)
                if cos > max1:
                    max3=max2
                    max3Class = max2Class
                    max2=max1
                    max2Class=max1Class
                    max1=cos
                    max1Class=classes[i]

                elif cos > max2:
                    max3 = max2
                    max3Class = max2Class
                    max2 = cos
                    max2Class = classes[i]

                elif cos > max3:
                    max3 = cos
                    max3class = classes[i]

        li=[]
        li.append(max1Class)
        li.append(max2Class)
        li.append(max3Class)
        return max(set(li), key=li.count)


    def FindingResults(self):
        '''
        This function perform 3 tasks
        1 > Take testing doc vector one by one and pass it to cosine similarity function to find its cosine similarity with trained data vectors
        2 > Check wehter the prediction is correct or wrong
        3 > Find accuracy
        :return:
        '''
        correct = 0
        for key, lis in self.TestVec.items():
            doc = self.labels.get(key)
            doc = round(doc * (self.Ratio/100))
            for value in lis:
                doc = doc + 1
                result=self.CosineSimilarity(value)
                print("DOCUMENT : {0} {1} | ".format(key, doc), end='')
                print("PREDICTION : {0}".format(result))
                if result == key:
                    print("****CORRECT****")
                    correct = correct + 1
                else:
                    print("WRONG !!!!!!!!!!!!")


        Accuracy = (correct / self.TotalDoc ) * 100
        print("\n\n-------------------------------------------")
        print("ACCURACY ACHIEVED  {0}".format(Accuracy))
        print("-------------------------------------------")

obj = TestingData(70)
obj.ReadTestingData()
obj.VectorRepresentation()
obj.FindingResults()

