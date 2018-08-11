from flask import Flask, request

from flask_cors import CORS
from flask_restful import Resource, Api
from flask_apidoc import ApiDoc

import uuid

__author__ = "Salvo Rinzivillo"
__email__ = "rinzivillo@isti.cnr.it"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
api = Api(app)
doc = ApiDoc(app=app)
CORS(app)

# Status code
SUCCESS = 200
CREATED = 201
BAD_REQUEST = 400
UNAUTHORIZED = 401
FORBIDDEN = 403
NOT_FOUND = 404
UNAVAILABLE = 451
NOT_IMPLEMENTED = 501

class ClusteringExperiment(Resource):

    def get(self):
        """
            @api {get} /api/ClusteringExperiment Create
        """
        token = str(uuid.uuid4())

        return {'token':token, 'type': 'general'}, SUCCESS


api.add_resource(ClusteringExperiment, '/api/ClusteringExperiment')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
