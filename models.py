from app import db
from sqlalchemy.dialects.postgresql import JSON


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # IMAP-related
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    server = db.Column(db.String(1000))
    port = db.Column(db.String(1000))

    # POP3-related
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    server = db.Column(db.String(1000))
    port = db.Column(db.String(1000))


class Email(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipients = db.Column(JSON)
    subject = db.Column(db.String(1000))
    body = db.Column(JSON)
