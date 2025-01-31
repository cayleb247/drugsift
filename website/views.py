from flask import Blueprint, render_template, request, session, url_for, redirect
from main import getLitData
from . import db
from .models import queryData
import json

from website.models import queryData
from website.models import featureScoringData
from website.models import compoundScoringData
from website.models import associatedDiseases


views = Blueprint('views', __name__)

@views.route("/", methods=["POST", "GET"])
@views.route("/home", methods=["POST", "GET"])
def home():
    with open('Data/disease_table.json', 'r') as file:
        data = json.load(file)
    return render_template("index.html", disease_data=data)
    
@views.route("/loading", methods=["POST"])
def loading():
    if request.method == "POST":

        # get all user input
        session["disease-name"] = request.form.get("disease-name")
        session["disease-id"] = request.form.get("disease-id")
        session["num-processes"] = request.form.get("num-processes")
        session["email"] = request.form.get("email")
        session["api-key"] = request.form.get("api-key")
        session["remove-terms"] = request.form.get("remove-terms")
        return render_template("loading.html")

@views.route("/data")
def data():
    
    getLitData(session["disease-name"], int(session["num-processes"]), session["email"], session["remove-terms"])

    return render_template("data.html", 
                           compounds=compoundScoringData.query.all(), 
                           features=featureScoringData.query.all(), 
                           diseases=associatedDiseases.query.all(),
                           abstracts=queryData.query.first())

@views.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")