from django.urls import path
# from .views import (
#     PredictListAPIView
# )
from . import views

# urlpatterns = [
#     path('',views.Hi),
#     path('name', views.get),
#     # path('repo', views.get_name, name="name"),
#     # path('name', views.get_details, name="details")
# ]

urlpatterns = [
    path('',views.apiOverView),
    path('repo', views.get_probability),
    path('repo/commit', views.get_features),
]
