# from rest_framework.generics import ListAPIView
from django.http import HttpResponse, JsonResponse
from rest_framework import views
from rest_framework.response import Response
from rest_framework.decorators import api_view
from commitPredictor.preprocess import Preprocess
from commitPredictor.predictor.predict_commit import Predict
from .serializer import CommitPrediction, CommitDetail
import pandas as pd
import os
import glob


@api_view(['GET'])
def apiOverView(request):
    api_urls = {
        'Detail View': '/repo/<str:pk>/'
    }
    return Response(api_urls)



def Hi(request):
    return HttpResponse("hello from commit predictor api")

@api_view(['GET'])
def get_probability(request):
    repo_name = request.GET['name']
    features = []
    process = Preprocess(repo_name)
    # err = process.clone()
    # print(err)
    # if(err):
    #     return JsonResponse(err)
    try:
        process.clone()
        process.process()
    except Exception as e:
        print('exception',e)
        return HttpResponse(status=400)

    df = process.get_features()
    pre = Predict(df)
    prob = pre.return_predict()
    com = list(df.index)
    file_present =glob.glob('preprocessed_data.csv')
    if(file_present):
        os.remove('preprocessed_data.csv')
        df.to_csv(r'preprocessed_data.csv', header=True)

    df.to_csv(r'preprocessed_data.csv', header=True)
    for i in range(len(prob)):
        detail = {'commit': com[i], 'prob': prob[i][0]}
        features.append(detail)
    # print(features)
    # reuslts = CommitPrediction(features, many=True).data
    # print("reuslts",reuslts)
    return Response(features)

@api_view(['GET'])
def get_features(request):
    id = int(request.GET['id'])
    df = pd.read_csv('../../preprocessed_data.csv')
    # com = list(df.index)
    df0 = df.iloc[id]
    print(df0)
    details = {'commit': df0['commit'], 'nd': df0['ns'],'ns': df0['nm'],'nf': df0['nf'],'entropy': df0['entropy'],'la': df0['la'],'ld': df0['ld'],'lt': df0['lt'],'fix': df0['fix'],'ndev': df0['ndev'],'nuc': df0['pd'],'age': df0['npt'],'exp': df0['exp'],'rexp': df0['rexp'],'sexp': df0['sexp']}
    reuslts = CommitDetail(details, many=False).data
    return Response(reuslts)