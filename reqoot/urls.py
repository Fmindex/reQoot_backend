"""winerama URL Configuration

The urlpatterns list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import include, url
from django.contrib import admin
from rest_framework import routers
from api import views as apiviews

router = routers.DefaultRouter()
router.register(r'users', apiviews.UserViewSet)
router.register(r'groups', apiviews.GroupViewSet)
# router.register(r'lr_predict', apiviews.linearRegressionPredict)

urlpatterns = [
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^api/prediction', apiviews.linearRegressionPredict.as_view(), name='my_rest_view'),
    # url(r'^', include('reviews.urls', namespace="reviews")),
    # url(r'^reviews/', include('reviews.urls', namespace="reviews")),
    # url(r'^admin/', include(admin.site.urls)),
    # url(r'^accounts/', include('registration.backends.simple.urls')),
    # url(r'^accounts/', include('django.contrib.auth.urls', namespace="auth")),
    
]