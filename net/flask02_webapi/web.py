from flask import Flask
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

tasks = [{
    'id': 1,
    'title': u'Buy groceries',
    'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
    'done': False
}, {
    'id': 2,
    'title': u'Learn Python',
    'description': u'Need to find a good Python tutorial on the web',
    'done': False
}]


class HelloWorld(Resource):

    def get(self):
        return {'msg': 'Hello World!'}


class TaskListApi(Resource):

    def __init__(self) -> None:
        self.reqparser = reqparse.RequestParser()
        self.reqparser.add_argument('title', type=str, required=True, help='no task title provided', location='json')
        self.reqparser.add_argument('description', type=str, default='', location='json')

        super(TaskListApi, self).__init__()

    def get(self):
        return {'tasks': []}


class TaskApi(Resource):

    def __init__(self) -> None:
        self.reqparser.add_argument('title', type=str, location='json')
        self.reqparser.add_argument('description', type=str, location='json')
        self.reqparser.add_argument('done', type=bool, location='json')
        super(TaskApi, self).__init__()


# add api
api.add_resource(HelloWorld, '/', endpoint='hi')
api.add_resource(TaskListApi, '/todo/api/v1.0/tasks', endpoint='task')
api.add_resource(TaskApi, '/todo/api/v1.0/tasks/<int:id>', endpoint='task')

if __name__ == '__main__':
    app.run(debug=True)