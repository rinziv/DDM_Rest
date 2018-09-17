import ddm
import random

from flask import Flask, request

from flask_cors import CORS
from flask_restful import Resource, Api
from flask_apidoc import ApiDoc

import json

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


config = json.load(open("resources/configuration.json"))


class KmeansExperiment(Resource):

    def __init__(self):
        self.kmeans_config = config['algorithms']['kmeans']['parameters']

    def __run_experiment(self, token, params):

        res = json.load(open("resources/kmeans.json"))
        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/KmeansExperiment Create
        """
        token = str(uuid.uuid4())

        c1 = request.args['c1'] if 'c1' in request.args else self.kmeans_config['c1']
        c2 = request.args['c2'] if 'c2' in request.args else self.kmeans_config['c2']
        distance = request.args['distance'] if 'distance' in request.args else self.kmeans_config['distance']

        params = {
            'c1': c1,
            'c2': c2,
            'distance': distance,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/KmeansExperiment Create
        """
        token = str(uuid.uuid4())

        c1 = request.form['c1'] if 'c1' in request.form else self.kmeans_config['c1']
        c2 = request.form['c2'] if 'c2' in request.form else self.kmeans_config['c2']
        distance = request.form['distance'] if 'distance' in request.form else self.kmeans_config['distance']

        params = {
            'c1': c1,
            'c2': c2,
            'distance': distance,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS


class DbscanExperiment(Resource):

    def __run_experiment(self, token, params):

        res = json.load(open("resources/dbscan.json"))
        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/DbscanExperiment Create
        """
        token = str(uuid.uuid4())

        eps = request.args['eps'] if 'eps' in request.args else 1.8
        min_pts = request.args['min_pts'] if 'min_pts' in request.args else 3

        params = {
            'eps': eps,
            'min_pts': min_pts,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/DbscanExperiment Create
        """
        token = str(uuid.uuid4())

        eps = request.form['eps'] if 'eps' in request.form else 1.8
        min_pts = request.form['min_pts'] if 'min_pts' in request.form else 3

        params = {
            'eps': eps,
            'min_pts': min_pts,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS


class HierarchicalExperiment(Resource):

    def __run_experiment(self, token, params):

        res = json.load(open("resources/hierarchical.json"))
        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/HierarchicalExperiment Create
        """
        token = str(uuid.uuid4())

        link_criteria = request.args['link_criteria'] if 'link_criteria' in request.args else 'min'
        distance = request.args['distance'] if 'distance' in request.args else 'euclidean'
        matrix_type = request.args['matrix_type'] if 'matrix_type' in request.args else 'distance'

        params = {
            'link_criteria': link_criteria,
            'distance': distance,
            'matrix_type': matrix_type,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/HierarchicalExperiment Create
        """
        token = str(uuid.uuid4())

        link_criteria = request.form['link_criteria'] if 'link_criteria' in request.form else 'min'
        distance = request.form['distance'] if 'distance' in request.form else 'euclidean'
        matrix_type = request.form['matrix_type'] if 'matrix_type' in request.form else 'distance'

        params = {
            'link_criteria': link_criteria,
            'distance': distance,
            'matrix_type': matrix_type,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS


class AprioriExperiment(Resource):

    def __run_experiment(self, token, params):

        res = json.load(open("resources/apriori.json"))
        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/AprioriExperiment Create
        """
        token = str(uuid.uuid4())

        min_sup = request.args['min_sup'] if 'min_sup' in request.args else 0.3
        sup_type = 'r'
        min_conf = request.args['min_conf'] if 'min_conf' in request.args else 0.7

        params = {
            'min_sup': min_sup,
            'sup_type': sup_type,
            'min_conf': min_conf,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/AprioriExperiment Create
        """
        token = str(uuid.uuid4())

        min_sup = request.form['min_sup'] if 'min_sup' in request.form else 0.3
        sup_type = 'r'
        min_conf = request.form['min_conf'] if 'min_conf' in request.form else 0.7

        params = {
            'min_sup': min_sup,
            'sup_type': sup_type,
            'min_conf': min_conf,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS


class DecisionTreeExperiment(Resource):

    def __run_experiment(self, token, params):

        res = json.load(open("resources/tree.json"))
        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/DecisionTreeExperiment Create
        """
        token = str(uuid.uuid4())

        split_function = request.args['split_function'] if 'split_function' in request.args else 'me'
        min_samples_split = request.args['min_samples_split'] if 'min_samples_split' in request.args else 2
        min_samples_leaf = request.args['min_samples_leaf'] if 'min_samples_leaf' in request.args else 1

        params = {
            'split_function': split_function,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/DecisionTreeExperiment Create
        """
        token = str(uuid.uuid4())

        split_function = request.form['split_function'] if 'split_function' in request.form else 'me'
        min_samples_split = request.form['min_samples_split'] if 'min_samples_split' in request.form else 2
        min_samples_leaf = request.form['min_samples_leaf'] if 'min_samples_leaf' in request.form else 1

        params = {
            'split_function': split_function,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
        }

        res = self.__run_experiment(token, params)

        return res, SUCCESS


api.add_resource(KmeansExperiment, '/api/KmeansExperiment')
api.add_resource(DbscanExperiment, '/api/DbscanExperiment')
api.add_resource(HierarchicalExperiment, '/api/HierarchicalExperiment')
api.add_resource(AprioriExperiment, '/api/AprioriExperiment')
api.add_resource(DecisionTreeExperiment, '/api/DecisionTreeExperiment')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
