from django.shortcuts import render
from django.http import HttpResponse
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework import permissions
# from commitPredictor.models import GitRepo
from .form import RepoNameForm
from .preprocess import Preprocess
from .predictor.predict_commit import Predict


# Create your views here.

def index(request):
    return HttpResponse("hello from commit predictor")

def get(request):
    return HttpResponse("from get endpoint")

def get_name(request):
    return render(request, "reponame/repoName.html")

def get_details(request):
    features = []
    repo_name = request.GET['repo-name']
    process = Preprocess(repo_name)
    process.clone()
    process.process()
    df = process.get_features()
    print(df)
    df
    pre = Predict(df)
    prob = pre.return_predict()
    com = list(df.index)
    for i in range(len(prob)):
        detail ={'commit': com[i], 'prob': prob[i]}
        features.append(detail)

    # print(prob)
    return render(request, 'reponame/result.html', {"features": features})
