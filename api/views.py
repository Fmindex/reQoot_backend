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
from sklearn.linear_model import LinearRegression

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
        
        USAhousing = pd.read_csv('USA_Housing.csv')
        X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
        y = USAhousing['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        # print(request.data)
        F = pd.Series(request.data,index=request.data.keys())
        G = []
        for key in X_test.keys():
            G.append(float(F[key]))
        Res = np.array([G])
        print(Res)
        predictions = lm.predict(Res)
        
        
        result = {"status" : 200, "burin" : "post", "prediction" : predictions}
        response = Response(result, status.HTTP_200_OK)
        return response
