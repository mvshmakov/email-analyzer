import os

from celery import Celery
from flask import Flask, request, redirect, render_template, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask.ext.login import login_required

from seq2seq.execute import give_suggestion
from models import Email

# The number of emails per page
EMAILS_PER_PAGE = 10

app = Flask(__name__)

app.config["MAIL_USE_TLS"] = False
app.config["MAIL_USE_SSL"] = True
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
mail = Mail(app)

celery = Celery(
    "main_app",
    broker=os.environ["CELERY_BROKER"],
    backend=os.environ["CELERY_BACKEND"],
)
celery.conf.update(app.config)


@celery.task
def send_email_async(recipients, subject, body):
    email = Email(recipients=recipients, subject=subject, body=body)
    return Mail.send(email)


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/login")
def sign_in():
    form = request.form.to_dict()
    if form["MAIL_SERVER"]:
        app.config["MAIL_SERVER"] = form["MAIL_SERVER"]
        app.config["MAIL_PORT"] = form["MAIL_SERVER"]
        app.config["MAIL_USERNAME"] = form["MAIL_SERVER"]
        app.config["MAIL_PASSWORD"] = form["MAIL_SERVER"]

    return render_template("index/index.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    to_predict = request.form.to_dict()
    prediction = give_suggestion(to_predict["intial_text"])
    # jsonify(request.json) request.get_json(force=True)
    return jsonify({"suggestion": prediction})


@app.route("/get_emails", methods=["GET"])
@login_required
def get_emails():
    # Get paginated emails
    page = request.args.get("page", 1, type=int)
    emails = Email.query.paginate(page=page, per_page=EMAILS_PER_PAGE)

    return render_template("emails/index.html", emails=emails)


@app.route("/send_email", methods=["POST"])
@login_required
def send_email():
    recipients = request.form.get("recipients")
    subject = request.form.get("subject")
    body = request.form.get("body")

    email_data = {"subject": subject, "recipients": recipients, "body": body}

    send_email_async.delay(email_data)
    return redirect(url_for("index"))


@app.route("/search_email", methods=["POST"])
@login_required
def search_email():
    query = request.form.get("query")
    matched_emails = Email.model.query.filter_by(query).all()
    return render_template("emails/index.html", emails=matched_emails)


if __name__ == "__main__":
    app.run(debug=True)
