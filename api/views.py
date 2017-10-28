from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import status
from rest_framework.views import APIView

from rest_framework.response import Response
from api.serializers import UserSerializer, GroupSerializer
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
  
class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer

class linearRegressionPredict(APIView):
    def get(self, request, *args, **kw):
        result = {"status" : 200, "burin" : "get"}
        response = Response(result, status.HTTP_200_OK)
        return response
    
    def post(self, request, *args, **kw):
        f = pd.DataFrame.from_dict(request.data, orient='index')
        data = f.copy().T
        data["Experience (years)"] = int(data["Experience (years)"])
        skills = data.iloc[0].Qualifications.split(", ")
        LEN = len(skills)

        for skill in skills:
            skill = skill.lower()
            data.insert(3, skill, 1, allow_duplicates=False)

        del data["Qualifications"]
        data.insert(0, "job_title_" + data.iloc[0].title, 1, allow_duplicates=False)
        del data["title"]

        data2 = data.copy()
        Candidate = pd.read_csv('all-label-final3FM.csv')
        Train = pd.read_csv('all-label-final3.csv')
        # เก่า
        # Candidate = pd.read_csv('fm.csv')
        # del Candidate["Skills"]
        # Train = pd.read_csv('label_onehot4.csv')
        
        
        Train = Train.reindex_axis(sorted(Train.columns), axis=1)  
        # Candidate
        def insert_row(idx, df, df_insert):
            return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop = True)
        for i in range(1, 100):
            data2 = insert_row(1, data2, data2.iloc[0])
        LEN = len(data2.keys())
        for key in Candidate.keys():
            if key in data2:
                data2.insert(LEN, key+".1", Candidate[key], allow_duplicates=False)
                for i in range(len(data2)):
                    temp = data2.iloc[i][key]
                    data2.iloc[i] = data2.iloc[i].set_value(key, data2.iloc[i][key+".1"])
                    data2.iloc[i] = data2.iloc[i].set_value(key+".1", temp)
            else:
                data2.insert(LEN, key, Candidate[key], allow_duplicates=False)
        LEN = len(data2.keys())
        for key in Train.keys():
            if key not in data2:
                data2.insert(LEN, key, 0, allow_duplicates=False)
        Tdata = Train.copy()
        y = Tdata["label"]
        NewData = Tdata.copy()
        del NewData["label"]
        X = NewData
        del data2["label"]
        # print(y)
        print(y)
        data2 = data2.reindex_axis(sorted(data2.columns), axis=1)     
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                            random_state=None)
        
        # model = ExtraTreesClassifier()
        
        normalized_X = preprocessing.normalize(X_train)
        standardized_X = preprocessing.scale(normalized_X)

        # model.fit(standardized_X, y_train)
        # meanny = np.mean(model.feature_importances_)
    
        # KEY = data2.keys()
        # count_fea = 280
        # for i in range(len(model.feature_importances_)):
        #     if model.feature_importances_[i] < meanny:
        #         del data2[KEY[i]]
        #         del X_test[KEY[i]]
        #         del X_train[KEY[i]]
        #         count_fea -= 1
        # print(count_fea)
        # print(X_train.shape)
        normalized_X = preprocessing.normalize(X_train)
        standardized_X = preprocessing.scale(normalized_X)
        normalized_X_test = preprocessing.normalize(X_test)
        standardized_X_test = preprocessing.scale(normalized_X_test)
        
        
        # logmodel = MLPClassifier(solver='lbfgs',activation='relu', alpha=1,hidden_layer_sizes=(280,30,1), random_state=None, max_iter=200)
        logmodel = LogisticRegression()
        logmodel.fit(standardized_X,y_train)
        normalized_X_new = preprocessing.normalize(data2)
        standardized_X_new = preprocessing.scale(normalized_X_new)
        
        predictions_new = logmodel.predict(standardized_X_new)
        idx = []
        # print(predictions_new[0])
        for i in range(len(predictions_new)):
            if predictions_new[i] == 1:
                idx.append(i)
        print(idx)
        # print(predictions_new)
        predictions_new = logmodel.predict(standardized_X)
        print(classification_report(y_train,predictions_new))
        print(metrics.accuracy_score(y_train, predictions_new))


        predictions_new = logmodel.predict(standardized_X_test)
        print(classification_report(y_test,predictions_new))
        print(metrics.accuracy_score(y_test, predictions_new))
  

        result = {"status" : 200, "burin" : "post", "prediction" : "OK"}
        response = Response(result, status.HTTP_200_OK)
        return response
