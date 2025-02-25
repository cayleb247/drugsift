from flask import Blueprint, render_template, request, session, url_for, redirect
from main import getLitData
from main import resetDatabase
from main import runWord2Vec
from main import protein_extactor
from . import db
from .models import queryData
import json
import subprocess
import os

from website.models import queryData
from website.models import featureScoringData
from website.models import compoundScoringData
from website.models import associatedDiseases
from website.models import cosineSimilarity

current_dir = os.path.dirname(os.path.abspath(__file__))

views = Blueprint('views', __name__)

@views.route("/", methods=["POST", "GET"])
@views.route("/home", methods=["POST", "GET"])
def home():
    with open(os.path.join(current_dir, '..', 'Data', 'disease_table.json'), 'r') as file:
        data = json.load(file)
    return render_template("index.html", disease_data=data)
    
@views.route("/loading", methods=["POST"])
def loading():
    if request.method == "POST":

        # identify what form was submitted
        form_id = request.form.get('form_id')

        if form_id == "form1":

            session.clear()

            # start with new database on every run
            resetDatabase()

            # get all user input
            session["disease-name"] = request.form.get("disease-name")
            session["disease-id"] = request.form.get("disease-id")
            session["num-processes"] = request.form.get("num-processes")
            session["email"] = request.form.get("email")
            session["api-key"] = request.form.get("api-key")
            session["remove-terms"] = request.form.get("remove-terms")

            # add the user's API-Key for the session
            if "api-key" in session:
                subprocess.run(f'export NCBI_API_KEY={session["api-key"]}', shell=True)

            # get a list of protein's amino acids from the inputted disease-id
            # if "disease-id" in session:      
            #     session["aa_seqs"] = protein_extactor(session["disease-id"])

            return render_template("loading.html", load_type = "disease", disease_name = session["disease-name"])
        
        elif form_id == "form2":
            session["associated-disease"] = request.form.get("associated-disease")

            return render_template("loading.html", load_type = "disease", disease_name = session["associated-disease"])
        
        elif form_id == "w2v":
            session["word2vec_term"] = request.form.get("w2v-disease")

            return render_template("loading.html", load_type = "w2v", prediction_term = session["word2vec_term"])
        
        elif form_id == "graph-model":
            
            return render_template("loading.html", load_type = "graph-model", disease_id = session["disease-id"])


            

@views.route("/data", methods=["POST", "GET"])
def data():

    if "word2vec_term" in session:

        runWord2Vec(session["word2vec_term"])

        return render_template("data.html", data_type = "w2v",
                                prediction_term = session["word2vec_term"],
                                w2v_terms = cosineSimilarity.query.all())

    elif "associated-disease" in session:

        getLitData(session["associated-disease"], int(session["num-processes"]), session["email"], session["remove-terms"])

        return render_template("data.html", 
                        data_type="associated-disease",
                        compounds=compoundScoringData.query.filter_by(search=session["associated-disease"]), 
                        features=featureScoringData.query.filter_by(search=session["associated-disease"]), 
                        diseases=associatedDiseases.query.filter_by(search=session["associated-disease"]),
                        abstracts=queryData.query.filter_by(search=session["associated-disease"]).first())
    
    elif "disease-name" in session:

        getLitData(session["disease-name"], int(session["num-processes"]), session["email"], session["remove-terms"])

        return render_template("data.html",
                            data_type="original-disease",
                            compounds=compoundScoringData.query.filter_by(search=session["disease-name"]), 
                            features=featureScoringData.query.filter_by(search=session["disease-name"]), 
                            diseases=associatedDiseases.query.filter_by(search=session["disease-name"]),
                            abstracts=queryData.query.filter_by(search=session["disease-name"]).first())
        
    else:
        error_message = "No disease name defined, please return to home page."
        return render_template("error.html", message = error_message)
         

@views.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")