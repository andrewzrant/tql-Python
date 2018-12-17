    from flask import Flask, request
    from flask_restful import Resource, Api, reqparse

    todos = {}
    class SklearnRestful(Resource):
        def get(self, todo_id):
            # json.dumps(str(_), default=lambda x: x.__dict__, indent=4)
            # _ = model.predict(text)
            return {todo_id: todos[todo_id], 'len': len(todos)}

        def put(self, todo_id):
            todos[todo_id] = request.form['data']
            return {todo_id: todos[todo_id]}

    print(todos)

    app = Flask(__name__)
    api = Api(app)
    api.add_resource(SklearnRestful, '/<string:todo_id>')
    app.run('0.0.0.0', port=5000, debug=True)
