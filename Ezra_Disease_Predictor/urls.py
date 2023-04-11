
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name = 'home'),
    # path('newResult/', views.newResult, name = 'result'),
    path('result/', views.result, name = 'result'),
    path('drug-recommendation/', views.drugRecommendation, name =
    'drug-recommendation')
]
