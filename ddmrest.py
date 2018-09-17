import json
import uuid

import pandas as pd

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
from flask_apidoc import ApiDoc

import ddm.didactic_kmeans as didactic_kmeans
import ddm.didactic_dbscan as didactic_dbscan
import ddm.didactic_hierarchical as didactic_hier
import ddm.didactic_apriori as didactic_apriori


__author__ = "Salvo Rinzivillo, Riccardo Guidotti"
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


# git submodule update --init --recursive --remote


class KmeansExperiment(Resource):

    def __init__(self):
        kmeans_config = config['algorithms']['kmeans']['parameters']
        param2index = {e['key']: i for i, e in enumerate(kmeans_config)}

        self.dataset_id = None
        self.dataset = None

        self.c1 = kmeans_config[param2index['c1']]['value']
        self.c2 = kmeans_config[param2index['c2']]['value']
        self.distance = kmeans_config[param2index['distance']]['value']

    def __run_experiment(self, token):

        # res_static = json.load(open("resources/kmeans.json"))
        kmeans = didactic_kmeans.DidatticKMeans(K=2, centroid_indexs=(self.c1, self.c2), dist=self.distance)
        kmeans.fit(self.dataset, step_by_step=False, plot_figures=False)
        res = kmeans.get_jdata()

        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/KmeansExperiment Create
        """
        token = str(uuid.uuid4())

        dataset_config = config['algorithms']['kmeans']['dataset']
        dataset2index = {e['key']: i for i, e in enumerate(dataset_config)}

        self.dataset_id = request.args['dataset']
        self.dataset = pd.read_csv(dataset_config[dataset2index[self.dataset_id]]['path'], header=None).values

        self.c1 = int(request.args['c1']) if 'c1' in request.args else self.c1
        self.c2 = int(request.args['c2']) if 'c2' in request.args else self.c2
        self.distance = request.args['distance'] if 'distance' in request.args else self.distance

        res = self.__run_experiment(token)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/KmeansExperiment Create
        """
        token = str(uuid.uuid4())

        dataset_config = config['algorithms']['kmeans']['dataset']
        dataset2index = {e['key']: i for i, e in enumerate(dataset_config)}

        self.dataset_id = request.form['dataset']
        self.dataset = pd.read_csv(dataset_config[dataset2index[self.dataset_id]]['path'], header=None).values

        self.c1 = int(request.form['c1']) if 'c1' in request.form else self.c1
        self.c2 = int(request.form['c2']) if 'c2' in request.form else self.c2
        self.distance = request.form['distance'] if 'distance' in request.form else self.distance

        res = self.__run_experiment(token)

        return res, SUCCESS


class DbscanExperiment(Resource):

    def __init__(self):
        dbscan_config = config['algorithms']['dbscan']['parameters']
        param2index = {e['key']: i for i, e in enumerate(dbscan_config)}

        self.dataset_id = None
        self.dataset = None

        self.eps = dbscan_config[param2index['eps']]['value']
        self.min_pts = dbscan_config[param2index['min_pts']]['value']

    def __run_experiment(self, token):

        # res = json.load(open("resources/dbscan.json"))
        dbscan = didactic_dbscan.DidatticDbscan(eps=self.eps, min_pts=self.min_pts)
        dbscan.fit(self.dataset, step_by_step=False, plot_figures=False)
        res = dbscan.get_jdata()

        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/DbscanExperiment Create
        """
        token = str(uuid.uuid4())

        dataset_config = config['algorithms']['dbscan']['dataset']
        dataset2index = {e['key']: i for i, e in enumerate(dataset_config)}

        self.dataset_id = request.args['dataset']
        self.dataset = pd.read_csv(dataset_config[dataset2index[self.dataset_id]]['path'], header=None).values

        self.eps = float(request.args['eps']) if 'eps' in request.args else self.eps
        self.min_pts = int(request.args['min_pts']) if 'min_pts' in request.args else self.min_pts

        res = self.__run_experiment(token)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/DbscanExperiment Create
        """
        token = str(uuid.uuid4())

        dataset_config = config['algorithms']['dbscan']['dataset']
        dataset2index = {e['key']: i for i, e in enumerate(dataset_config)}

        self.dataset_id = request.form['dataset']
        self.dataset = pd.read_csv(dataset_config[dataset2index[self.dataset_id]]['path'], header=None).values

        self.eps = float(request.form['eps']) if 'eps' in request.form else self.eps
        self.min_pts = int(request.form['min_pts']) if 'min_pts' in request.form else self.min_pts

        res = self.__run_experiment(token)

        return res, SUCCESS


class HierarchicalExperiment(Resource):

    def __init__(self):
        hierarchical_config = config['algorithms']['hierarchical']['parameters']
        param2index = {e['key']: i for i, e in enumerate(hierarchical_config)}

        self.dataset_id = None
        self.dataset = None

        self.link_criteria = hierarchical_config[param2index['link_criteria']]['value']
        self.distance = hierarchical_config[param2index['distance']]['value']
        self.matrix_type = hierarchical_config[param2index['matrix_type']]['value']

    def __run_experiment(self, token):

        # res = json.load(open("resources/hierarchical.json"))

        hier = didactic_hier.DidatticHierarchical()
        use_distances = self.matrix_type == 'distance'
        hier.fit(self.dataset, link_criteria=self.link_criteria, use_distances=use_distances, step_by_step=False,
                 distance_type=self.distance, plot_figures=False)
        res = hier.get_jdata()
        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/HierarchicalExperiment Create
        """
        token = str(uuid.uuid4())

        dataset_config = config['algorithms']['dbscan']['dataset']
        dataset2index = {e['key']: i for i, e in enumerate(dataset_config)}

        self.dataset_id = request.args['dataset']
        self.dataset = pd.read_csv(dataset_config[dataset2index[self.dataset_id]]['path'], header=None).values

        self.link_criteria = request.args['link_criteria'] if 'link_criteria' in request.args else self.link_criteria
        self.distance = request.args['distance'] if 'distance' in request.args else self.distance
        self.matrix_type = request.args['matrix_type'] if 'matrix_type' in request.args else self.matrix_type

        res = self.__run_experiment(token)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/HierarchicalExperiment Create
        """
        token = str(uuid.uuid4())

        dataset_config = config['algorithms']['dbscan']['dataset']
        dataset2index = {e['key']: i for i, e in enumerate(dataset_config)}

        self.dataset_id = request.form['dataset']
        self.dataset = pd.read_csv(dataset_config[dataset2index[self.dataset_id]]['path'], header=None).values

        self.link_criteria = request.form['link_criteria'] if 'link_criteria' in request.form else self.link_criteria
        self.distance = request.form['distance'] if 'distance' in request.form else self.distance
        self.matrix_type = request.form['matrix_type'] if 'matrix_type' in request.form else self.matrix_type

        res = self.__run_experiment(token)

        return res, SUCCESS


def read_transactional_dataset(filename):
    data = open(filename, 'r')
    dataset = list()
    for row in data:
        dataset.append(row.strip().split(','))
    return dataset


class AprioriExperiment(Resource):

    def __init__(self):
        apriori_config = config['algorithms']['apriori']['parameters']
        param2index = {e['key']: i for i, e in enumerate(apriori_config)}

        self.dataset_id = None
        self.dataset = None

        self.min_sup = apriori_config[param2index['min_sup']]['value']
        self.min_conf = apriori_config[param2index['min_conf']]['value']

    def __run_experiment(self, token):

        # res = json.load(open("resources/apriori.json"))
        apriori = didactic_apriori.DidatticApriori(min_sup=self.min_sup, sup_type='r')
        apriori.fit(self.dataset, step_by_step=False)
        apriori.extract_rules(min_conf=self.min_conf)
        res = apriori.get_jdata()

        res['token'] = token
        res['type'] = 'general'

        return res

    def get(self):
        """
            @api {get} /api/AprioriExperiment Create
        """
        token = str(uuid.uuid4())

        dataset_config = config['algorithms']['apriori']['dataset']
        dataset2index = {e['key']: i for i, e in enumerate(dataset_config)}

        self.dataset_id = request.args['dataset']
        self.dataset = read_transactional_dataset(dataset_config[dataset2index[self.dataset_id]]['path'])

        self.min_sup = float(request.args['min_sup']) if 'min_sup' in request.args else self.min_sup
        self.min_conf = float(request.args['min_conf']) if 'min_conf' in request.args else self.min_conf

        res = self.__run_experiment(token)

        return res, SUCCESS

    def post(self):
        """
            @api {post} /api/AprioriExperiment Create
        """
        token = str(uuid.uuid4())

        dataset_config = config['algorithms']['apriori']['dataset']
        dataset2index = {e['key']: i for i, e in enumerate(dataset_config)}

        self.dataset_id = request.form['dataset']
        self.dataset = read_transactional_dataset(dataset_config[dataset2index[self.dataset_id]]['path'])

        self.min_sup = float(request.form['min_sup']) if 'min_sup' in request.form else self.min_sup
        self.min_conf = float(request.form['min_conf']) if 'min_conf' in request.form else self.min_conf

        res = self.__run_experiment(token)

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
