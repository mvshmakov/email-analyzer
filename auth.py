from flask import Blueprint
from flask import redirect, url_for


auth = Blueprint('auth', __name__)


@auth.route('/login')
def login():
    return redirect(url_for('app.login'))
