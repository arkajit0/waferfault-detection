from wsgiref import simple_server
from dash import Dash
import dash_daq as daq
import dash_core_components as dcc
from dash.dependencies import Input, Output, MATCH
import plotly.graph_objs as go
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import dash_html_components as html
from flask import Flask, request, render_template
from flask import Response, redirect
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
import pandas as pd
import pymongo
import json
import threading
from dashboard import Pages


values = Pages()

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

server = Flask(__name__)
dash_app = Dash(__name__, server = server, url_base_pathname='/dashboard/')
dash_app.layout = html.Div([
                                dcc.Tabs(id="tabs-styled-with-props", value='tab-1', children=[
                                    dcc.Tab(label='Prediction Report', value='tab-1'),
                                    dcc.Tab(label='Training Report', value='tab-2'),
                                ], colors={
                                    "border": "white",
                                    "primary": "gold",
                                    "background": "cornsilk"
                                }),
                                html.Div(id='tabs-content-props')
                            ])
dashboard.bind(server)
CORS(server)

#prediction func
def train_processing():
    path = 'training-batch-files'
    #
    train_valObj = train_validation(path)  # object initialization

    train_valObj.train_validation()  # calling the training_validation function

    trainModelObj = trainModel()  # object initialization
    trainModelObj.trainingModel()  # training the model for the files in the table


@server.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@server.route('/dashboard/')
def render_dashboard():
    return redirect('/dash1')

@server.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predictRouteClient():
    path = 'prediction-batch-files'

    # try:
    # if request.json is not None:
    #     path = request.json['filepath']
    #
    #     pred_val = pred_validation(path) #object initialization
    #
    #     pred_val.prediction_validation() #calling the prediction_validation function
    #
    #     pred = prediction(path) #object initialization
    #
    #     # predicting for dataset present in database
    #     path,json_predictions = pred.predictionFromModel()
    #     return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
    # elif request.form is not None:
    #     path = request.form['filepath']
    #
    #     pred_val = pred_validation(path) #object initialization
    #
    #     pred_val.prediction_validation() #calling the prediction_validation function
    #
    #     pred = prediction(path) #object initialization
    #
    #     # predicting for dataset present in database
    #     path,json_predictions = pred.predictionFromModel()
    #     return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
    # else:
    path = path

    pred_val = pred_validation(path)  # object initialization

    pred_val.prediction_validation()  # calling the prediction_validation function

    pred = prediction(path)  # object initialization

    # predicting for dataset present in database
    json_predictions = pred.predictionFromModel()
    return Response("Prediction File created at !!!" + str(path) + 'and few of the predictions are ' + str(
        json.loads(json_predictions)))
        # print('Nothing Matched')
    # except ValueError:
    #     return Response("Error Occurred! %s" %ValueError)
    # except KeyError:
    #     return Response("Error Occurred! %s" %KeyError)
    # except Exception as e:
    #     return Response("Error Occurred! %s" %e)



@server.route("/train", methods=['GET'])
@cross_origin()
def trainRouteClient():

    try:
        # if request.json['folderPath'] is not None:
            # path = request.json['folderPath']
        train_thread = threading.Thread(target=train_processing)
        train_thread.start()


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("You will receive an mail when Training is complete!!!")

@server.route("/track", methods=['POST'])
def tracker():
    print(request)
    if request.method == 'POST':
        print(request.json)

@dash_app.callback(Output('tabs-content-props', 'children'),
              Input('tabs-styled-with-props', 'value'))
def render_content(tab):
    if tab == 'tab-1':

        return values.prediction_page()

    elif tab == 'tab-2':
        return values.training_page()

#training
@dash_app.callback(
    Output({'type': 'graph-train-b'}, 'figure'),
    [Input(component_id={'type': 'drop-down-train-b'}, component_property='value')]
)
def update_graph(s_valuece):
    # print(s_valuece)
    dic = {"namevalidation": [values.train_nameerror, "Wrong File Name"], "columnvalidation": [values.train_columnerror, "Wrong Column Length"],
           "columnvaluemissing": [values.train_missingvalues, "Missing values in column"]}
    labels = [dic[s_value][1] for s_value in s_valuece]
    value = [dic[s_value][0] for s_value in s_valuece]
    fig = go.Figure(data=[go.Pie(labels=labels,
                                 values=value)], layout={"showlegend": False, "height": 340, "width": 460})
    #     go.Figure(data=[go.Pie(labels=['Good Data','Bad Data'],
    #                              values=[good_csv, bad_csv])])

    return fig

@dash_app.callback(
    Output({'type': 'graph-train-a'}, 'figure'),
    [Input(component_id={'type': 'drop-down-train-a'}, component_property='value')]
)
def update_graph(s_valuece):
    # print(s_valuece)
    if s_valuece == "data":
        fig = go.Figure(data=[go.Pie(labels=['Good Data', 'Bad Data'],
                                     values=[values.good_data_train, values.bad_data_train])])
    else:
        fig = go.Figure(data=[go.Pie(labels=['Good Files', 'Bad Files'],
                                     values=[values.good_files_train, values.bad_files_train])])

    return fig


# predictions
@dash_app.callback(
    Output({'type': 'graph-pred-b'}, 'figure'),
    [Input(component_id={'type': 'drop-down-pred-b'}, component_property='value')]
)
def update_graph(s_valuece):
    # print(s_valuece)
    dic = {"namevalidation": [values.pred_nameerror, "Wrong File Name"], "columnvalidation": [values.pred_columnerror, "Wrong Column Length"],
           "columnvaluemissing": [values.pred_missingvalues, "Missing values in column"]}
    labels = [dic[s_value][1] for s_value in s_valuece]
    value = [dic[s_value][0] for s_value in s_valuece]
    fig = go.Figure(data=[go.Pie(labels=labels,
                                 values=value)], layout={"showlegend": False, "height": 340, "width": 460})
    #     go.Figure(data=[go.Pie(labels=['Good Data','Bad Data'],
    #                              values=[good_csv, bad_csv])])

    return fig


@dash_app.callback(
    Output({'type': 'graph-pred-a'}, 'figure'),
    [Input(component_id={'type': 'drop-down-pred-a'}, component_property='value')]
)
def update_graph(s_valuece):
    # print(s_valuece)
    if s_valuece == "data":
        fig = go.Figure(data=[go.Pie(labels=['Good Data', 'Bad Data'],
                                     values=[values.good_data_pred, values.bad_data_pred])])
    else:
        fig = go.Figure(data=[go.Pie(labels=['Good Files', 'Bad Files'],
                                     values=[values.good_files_pred, values.bad_files_pred])])

    return fig


app = DispatcherMiddleware(server, {
    '/dash1': dash_app.server
})
# port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    # host = '0.0.0.0'
    # port = 5000
    # httpd = simple_server.make_server(host, port, app)
    # # print("Serving on %s %d" % (host, port))
    # httpd.serve_forever()
    run_simple(hostname='0.0.0.0', port=5000, application=app, use_debugger=True)
