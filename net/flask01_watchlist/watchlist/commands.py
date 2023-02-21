import click
from watchlist import app, db
from watchlist.models import User, Movie


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
