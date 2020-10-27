from flask import Flask, request, jsonify
from flask_cors import CORS
import recommendation
from recommender.modeling.spark_prediction import results
from flask_restplus import Api, Resource, fields
import os

flask_app = Flask(__name__)
app = Api(app=flask_app,
          version="1.0",
          title="Recommendation API Demo",
          description="Recommend Movies")
# CORS(app)

name_space = app.namespace('movie', description='Recommend Movies')

model = app.model('Name Model',
                  {'name': fields.String(required=True,
                                         description="Name of the movie",
                                         help="Name cannot be blank.")})

list_of_names = {}

@name_space.route("/<string:movie_name>")
class MainClass(Resource):

    @app.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'},
             params={'movie_name': 'Specify the movie name you are interested in'})
    def get(self, movie_name):
        try:
            res = recommendation.results(movie_name)
            return jsonify(res)
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not find information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not find information", statusCode="400")

    # def post(self, name):
    #     return {
    #         "status": "Posted new data"
    #     }

    # @app.expect(model)
    # def post(self, name):
    #     try:
    #         res = recommendation.results(request.json['name'])
    #         return jsonify(res)
    #         # list_of_names[id] = request.json['name']
    #         # return {
    #         #     "status": "New person added",
    #         #     "name": list_of_names[id]
    #         # }
    #     except KeyError as e:
    #         name_space.abort(500, e.__doc__, status="Could not save information", statusCode="500")
    #     except Exception as e:
    #         name_space.abort(400, e.__doc__, status="Could not save information", statusCode="400")


# @app.route('/movie', methods=['GET'])
#     def recommend_movies():
#             res = recommendation.results(request.args.get('title'))
#             return jsonify(res)
#
# def recommend_movies():
#     res = results(request.args.get('title'))
#     return jsonify(res)


if __name__ == '__main__':
    port = int(os.environ.get("PORT" or 5000))
    app.run(host="0.0.0.0", port=port)
