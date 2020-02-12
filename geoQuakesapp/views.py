from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from django.template import RequestContext
from datetime import datetime
from django.core.serializers import serialize
from geoQuakesapp.models import Quake, Quake_Predictions
from django.template.context import Context
import pandas as pd

# Create your views here.
def quake_dataset(request):
    quakes = serialize('json', Quake.objects.order_by("ID")[:1000])
    return HttpResponse(quakes, content_type='json')

# Create endpoint for prediction dataset
def quake_dataset_pred(request):
    quakes_pred = serialize('json', Quake_Predictions.objects.all()[:1000])
    return HttpResponse(quakes_pred, content_type='json')

# Create endpoint for high risk quakes
def quake_dataset_pred_risk(request):
    quake_risk = serialize('json', Quake_Predictions.objects.filter(Magnitude__gt=6.5))
    return HttpResponse(quake_risk, content_type='json')

def pred_score():
    score = Quake_Predictions.objects.all()[0]
    ret_score = str(round(score.Score, 2))
    return ret_score

def home(request):
    return render(
        request,
        'app/index.html',
        {
            'title': 'Home Page',
            'pred_score': pred_score()
        }
    )