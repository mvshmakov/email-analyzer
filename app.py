import os

from celery import Celery
from flask import Flask, request, redirect, render_template, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask.ext.login import login_required

from seq2seq.execute import give_suggestion

SETTINGS = {}
ROWS_PER_PAGE = 10


class Email:
    pass


app = Flask(__name__)

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USERNAME"] = "yourId@gmail.com"
app.config["MAIL_PASSWORD"] = "*****"
app.config["MAIL_USE_TLS"] = False
app.config["MAIL_USE_SSL"] = True

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

postgre = SQLAlchemy(app)
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


@app.route("/predict", methods=["POST"])
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
    emails = Email.query.paginate(page=page, per_page=ROWS_PER_PAGE)

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


if __name__ == "__main__":
    app.run(debug=True)

# TODO: search emails
