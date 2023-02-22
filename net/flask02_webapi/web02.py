from flask import Flask, abort
from flask_restful import Api, Resource, reqparse, fields, marshal

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

task_fields = {
    'title': fields.String,
    'description': fields.String,
    'done': fields.Boolean,
    'uri': fields.Url('task', absolute=True)
}


class HelloWorld(Resource):

    def get(self):
        return {'msg': 'Hello World!'}


class TaskListApi(Resource):

    def __init__(self) -> None:
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('title', type=str, required=True, help='no task title provided', location='json')
        self.parser.add_argument('description', type=str, default='', location='json')
        super(TaskListApi, self).__init__()

    def get(self):
        return {'tasks': [marshal(v, task_fields) for v in tasks]}

    def post(self):
        args = self.parser.parse_args()
        task = {
            'id': tasks[-1]['id'] + 1 if len(tasks) > 0 else 1,
            'title': args['title'],
            'description': args['description'],
            'done': False
        }
        tasks.append(task)
        return {'task': marshal(task, task_fields)}, 201


class TaskApi(Resource):

    def __init__(self) -> None:
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('title', type=str, location='json')
        self.parser.add_argument('description', type=str, location='json')
        self.parser.add_argument('done', type=bool, location='json')
        super(TaskApi, self).__init__()

    def get(self, id):
        task = [v for v in tasks if v['id'] == id]
        if len(task) == 0:
            abort(404)
        return {'task': marshal(task[0], task_fields)}

    def put(self, id):
        task = [v for v in tasks if v['id'] == id]
        if len(task) == 0:
            abort(404)
        task = task[0]
        args = self.parser.parse_args()
        for k, v in args.items():
            task[k] = v
        return {'task': marshal(task, task_fields)}

    def delete(self, id):
        task = [v for v in tasks if v['id'] == id]
        if len(task) == 0:
            abort(404)
        tasks.remove(task[0])
        return {'result': True}


# add api
api.add_resource(HelloWorld, '/', endpoint='hi')
api.add_resource(TaskListApi, '/todo/api/v1.0/tasks', endpoint='tasks')
api.add_resource(TaskApi, '/todo/api/v1.0/tasks/<int:id>', endpoint='task')

if __name__ == '__main__':
    app.run(debug=True)