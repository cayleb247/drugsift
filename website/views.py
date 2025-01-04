from flask import Blueprint, render_template, request, session, url_for, redirect

views = Blueprint('views', __name__)

@views.route("/", methods=["POST", "GET"])
@views.route("/home", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        query = request.form["user-query"]
        session["query"] = query
        return redirect(url_for("data"))
    else: 
        return render_template("index.html")

@views.route("/data")
def data():
    return render_template("data.html", query=session["query"], lemma=session["lemma"])