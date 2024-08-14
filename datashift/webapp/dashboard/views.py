import io

import pandas as pd
import requests
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render


# Create your views here.
def home(request: HttpRequest) -> HttpResponse:
    """

    :param request:
    :return:
    """
    return render(
        request,
        'home.html'
    )


def plot_results(request: HttpRequest) -> HttpResponse:
    """

    :param request:
    :return:
    """

    if request.method == 'GET':
        demo = request.POST.get('demo', None)

        if demo:
            pass

    elif request.method == 'POST':
        request.FILES.get('data', None)

    # TODO: Think how to upload and manage data
    # Compute all the stuff

    # store the dataset temporally?

    return render(
        request,
        'plot_results.html',
        context={

        }
    )
