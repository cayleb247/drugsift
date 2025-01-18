from flask import Blueprint, render_template, request, session, url_for, redirect
from . import db
from .models import queryData
import json

views = Blueprint('views', __name__)

@views.route("/", methods=["POST", "GET"])
@views.route("/home", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        search = request.form["abstract-search"]
        text = request.form["abstract-text"]

        new_abstract = queryData(search=search, abstract=text)
        db.session.add(new_abstract)
        db.session.commit()

        return redirect(url_for("views.data"))
    else:
        with open('Data/disease_table.json', 'r') as file:
            data = json.load(file)
        return render_template("index.html", disease_data=data)

@views.route("/data")
def data():

    abstracts = queryData.query.all()

    print(abstracts)

    return render_template("data.html", abstracts=abstracts)

@views.route("/tutorial")
def tutorial():
    return render_template("tutorial.html")