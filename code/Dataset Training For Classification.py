import string
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# This class will train the data and stored the vectors of each document in specific .txt file
class TrainingDataset:

    def __init__(self,ratio):
        self.labels = {"athletics":101,"cricket":124,"football":265,"rugby":147,"tennis":100}  #Specifying no of documents have to read from each folder
        self.DatasetDic = {}
        self.splitRatio = ratio
        self.NoDoc = 0
        self.Datasetpath = "../bbcsport/"   #Datset Folder Path
        self.features = []
        self.stopwords = stopwords.words('english')
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        print("TRANING DATA RATIO IS SET TO {0}% ".format(ratio))

    def FileReading(self):
        '''
        This function read the data from foldersby specfifed ratio of 70 /30 and do cleaning , extracting features
        :return:
        '''
        DatsetSplit = open("DatasetSeparation.txt","w+") #This file store the number of document from each class assigned for training and testing
        porter = PorterStemmer()

        for key, value in self.labels.items():
            DatsetSplit.write(key+"\n"+"Training Data : ")
            dirPath = self.Datasetpath + key + "/"
            NoOfDocRead = value * (self.splitRatio / 100)
            NoOfDocRead = round(NoOfDocRead)
            self.NoDoc = self.NoDoc + NoOfDocRead
            NoOfDocRead = round(NoOfDocRead)
            self.DatasetDic[key] = []
            print("READING DATA FROM FOLDER {0}...".format(key))
            for i in range(1,NoOfDocRead+1):
                j = str(i)
                j = j.zfill(3)
                DatsetSplit.write(j+" ")
                path = dirPath + j + ".txt"
                text =""
                with open(path) as f:
                    file = f.read()

                for x in file:
                    if x in self.punctuations:
                        file = file.replace(x, " ")
                text= file.lower()
                words = text.split()                                #spliting dataof each file into words
                data = ""
                for x in words:
                    stem = porter.stem(x)                            # stem each word
                    data = data + stem + " "
                    if any(map(str.isdigit, stem)) == False :        #Check if the word is not in stop words, not already exist in features and not in digit then include it in features
                        if stem not in self.stopwords:
                            if stem not in self.features:
                                self.features.append(stem)


                self.DatasetDic[key].append(data)                   # Appending Data of each file to dictionary so the tf will calculated easily

            DatsetSplit.write("\n")
            DatsetSplit.write("Testing Data  : ")
            for i in range(NoOfDocRead+1,value+1):
                j= str(i)
                j=j.zfill(3)
                DatsetSplit.write(j+" ")
            DatsetSplit.write("\n")

        DatsetSplit.close()

    def VectorRepresentation(self):
        '''
        This function perform 2 tasks
        1 > store the features and each feature's IDF in vocabulary.txt
        2 > Represent Each document as vector by calculating TF_IDF of each feature
        :return:
        '''
        voc = open("Vocabulary.txt", "w+")
        IDf = []
        for word in  self.features:
            Df = 0
            for value in self.DatasetDic.values():
                for x in value:
                    if x.count(word) > 0:
                        Df= Df + 1                             #Finding Document frequency of each term

            idf = math.log2(self.NoDoc/ Df)                    #Finding Inverse Document frequency of each term
            IDf.append(idf)
            voc.write(word+"\t"+str(idf))
            voc.write("\n")

        voc.close()
        print("CREATING VECTORS OF EACH DOCUMENT")
        print("TOTAL DOC {0}".format(self.NoDoc))
        print("TOTAL NO OF FEATURES {0}".format(len(self.features)))
        for key, value in self.DatasetDic.items():
            file = open(key + ".txt", "w+")                    #Create a file by name of each class

            for x in value:
                lis = []
                for i in range(0, len(self.features)):
                    tf = x.count(self.features[i])
                    if tf > 0:
                        tf = 1 + math.log2(tf)

                    tf_idf = IDf[i] * tf
                    tf_idf = str(tf_idf)[:7]
                    tf_idf = tf_idf + " "
                    lis.append(tf_idf)

                str1=''.join(lis)
                file.write(str1)                              #Store each class document's vector in .txt file
                file.write("\n")
        print("DATA TRAINED AND SAVE SUCCESSFULLY ")


obj = TrainingDataset(70)
obj.FileReading()
obj.VectorRepresentation()

