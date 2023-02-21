import flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import click
from markupsafe import escape
from pathlib import Path

app = flask.Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////' + str(Path(app.root_path).parent.parent / 'data/flask.db')
app.config['SECRET_KEY'] = 'dev'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20))
    username = db.Column(db.String(20))
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def validate_password(self, password):
        return check_password_hash(self.password_hash, password)


class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(60))
    year = db.Column(db.String(4))


@login_manager.user_loader
def load_user(user_id):
    user = User.query.get(int(user_id))
    return user


@app.cli.command()
@click.option('--drop', is_flag=True, help='create after drop')
def init_db(drop):
    """init db"""
    if drop:
        db.drop_all()
    db.create_all()
    click.echo('initialized database')


@app.cli.command()
def forge():
    """generate fake data"""
    db.create_all()

    name = 'Jeffery'
    movies = [{
        'title': 'My Neighbor Totoro',
        'year': '1988'
    }, {
        'title': 'Dead Poets Society',
        'year': '1989'
    }, {
        'title': 'A Perfect World',
        'year': '1993'
    }, {
        'title': 'Leon',
        'year': '1994'
    }, {
        'title': 'Mahjong',
        'year': '1996'
    }, {
        'title': 'Swallowtail Butterfly',
        'year': '1996'
    }, {
        'title': 'King of Comedy',
        'year': '1999'
    }, {
        'title': 'Devils on the Doorstep',
        'year': '1999'
    }, {
        'title': 'WALL-E',
        'year': '2008'
    }, {
        'title': 'The Pork of Music',
        'year': '2012'
    }]

    user = User(name=name)
    db.session.add(user)

    for m in movies:
        movie = Movie(title=m['title'], year=m['year'])
        db.session.add(movie)

    db.session.commit()
    click.echo('generate fake data')


@app.cli.command()
@click.option('--username', prompt=True, help='the username used to login')
@click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True, help='the password used to login')
# @click.password_option('--password', help='the password used to login')
def admin(username, password):
    # create user
    db.create_all()

    user: User = User.query.first()
    if user is not None:
        click.echo('updating user...')
        user.set_password(password)
    else:
        click.echo('create user...')
        user = User(username=username, name='Admin')
        user.set_password(password)
        db.session.add(user)

    db.session.commit()
    click.echo('done!')


@app.context_processor
def inject_user():
    user = User.query.first()
    return dict(user=user)


@app.errorhandler(404)
def page_not_found(e):
    return flask.render_template('404.html'), 404


@app.route('/', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'POST':
        if not current_user.is_authenticated:
            return flask.flash(flask.flash('index'))

        title = flask.request.form.get('title')
        year = flask.request.form.get('year')
        if not title or not year or len(year) > 4 or len(title) > 60:
            flask.flash('Invalid input')
            return flask.redirect(flask.url_for('index'))
        movie = Movie(title=title, year=year)
        db.session.add(movie)
        db.session.commit()
        flask.flash('item created')
        return flask.redirect(flask.url_for('index'))

    movies = Movie.query.all()
    return flask.render_template('index.html', movies=movies)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'POST':
        username = flask.request.form['username']
        password = flask.request.form['password']
        if not username or not password:
            flask.flash('Invalid input', 'error')
            return flask.redirect(flask.url_for('login'))
        user: User = User.query.first()

        if username == user.username and user.validate_password(password):
            login_user(user)
            flask.flash('Login success')
            return flask.redirect(flask.url_for('index'))

        flask.flash('Invalid username and password')
        return flask.redirect(flask.url_for('login'))

    return flask.render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flask.flash('Goodbye')
    return flask.redirect(flask.url_for('index'))


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if flask.request.method == 'POST':
        name = flask.request.form['name']
        if not name or len(name) > 20:
            flask.flash('Invalid input')
            flask.redirect(flask.url_for('settings'))

        current_user.name = name
        db.session.commit()
        flask.flash('Settings updated.')
        flask.redirect(flask.url_for('index'))

    return flask.render_template('settings.html')


@app.route('/movie/edit/<int:movie_id>', methods=['GET', 'POST'])
@login_required
def edit(movie_id):
    movie = Movie.query.get_or_404(movie_id)

    if flask.request.method == 'POST':
        title = flask.request.form['title']
        year = flask.request.form['year']
        if not title or not year or len(year) != 4 or len(title) > 60:
            flask.flash('Invalid input')
            return flask.redirect(flask.url_for('edit', movie_id=movie_id))
        movie.title = title
        movie.year = year
        db.session.commit()
        flask.flash('item updated')
        return flask.redirect(flask.url_for('index'))

    return flask.render_template('edit.html', movie=movie)


@app.route('/movie/delete/<int:movie_id>')
@login_required
def delete(movie_id):
    movie = Movie.query.get_or_404(movie_id)
    db.session.delete(movie)
    db.session.commit()
    flask.flash('Item deleted')
    return flask.redirect(flask.url_for('index'))


@app.route('/user/<name>')
def user_page(name):
    return flask.jsonify(user=escape(name))


@app.route('/test')
def test():
    print(flask.url_for('index'))
    print(flask.url_for('user_page', name='jeffery'))
    print(flask.url_for('test'))
    print(flask.url_for('test', num=2))

    return 'Test Page'


if __name__ == '__main__':
    app.run(debug=True)
