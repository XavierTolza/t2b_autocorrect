from PIL.Image import Image
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)


class Endpoint(Resource):
    def get(self):
        raise NotImplementedError("todo")

    def post(self):
        args = request.data
        img = Image.frombytes(args[3:])
        return {"res": args.decode("utf-8")}


api = Api(app)
api.add_resource(Endpoint, '/')
