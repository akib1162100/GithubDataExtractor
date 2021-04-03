from django.urls import path
from . import views

urlpatterns = [
    path('', views.get),
    path('repo', views.get_name, name="name"),
    path('name', views.get_details, name="details")
]
