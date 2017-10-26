from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import status
from rest_framework.views import APIView

from rest_framework.response import Response
from api.serializers import UserSerializer, GroupSerializer
import json

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
        print(json.loads(request.body))
        result = {"status" : 200, "burin" : "post"}
        response = Response(result, status.HTTP_200_OK)
        return response
