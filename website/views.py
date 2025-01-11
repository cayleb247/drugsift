from flask import Blueprint, render_template, request, session, url_for, redirect
from . import db
from .models import queryData

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
        return render_template("index.html")

@views.route("/data")
def data():

    abstracts = queryData.query.all()

    print(abstracts)

    return render_template("data.html", abstracts=abstracts)