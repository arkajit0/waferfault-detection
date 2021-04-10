from wsgiref import simple_server
from dash import Dash
from datetime import date
import dash_daq as daq
import dash_core_components as dcc
from dash.dependencies import Input, Output, MATCH
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import dash_html_components as html
from flask import Flask, request, render_template, jsonify
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
import pickle
import threading
import dash_table
from dashboard import Pages
import config as cg


values = Pages()

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

server = Flask(__name__)
dash_app = Dash(__name__, server=server, url_base_pathname='/dashboard/')
analyzer_app = Dash(__name__, server=server, url_base_pathname='/analyze/')
Filepage_app = Dash(__name__, server=server, url_base_pathname='/filedetails/')
dash_app.layout = html.Div(
                            [
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

analyzer_app.layout = values.Model_Analyzer()

Filepage_app.layout = html.Div([
                                    dcc.RadioItems(
                                        id = "file-type",
                                        options=[{'label': "Training Files", 'value':'Training'},
                                                 {'label': "Prediction Files", 'value':'Prediction'},
                                                 {'label': 'New Incoming Files', 'value': 'newfile'}],
                                        value="Training"
                                    ),
                                    dcc.Dropdown(
                                    id = "data-collection",
                                    options=[{'label': "Wrong File Name", 'value': "namevalidation"},
                                                             {'label': "Wrong Column Length", 'value': "columnvalidation"},
                                                             {'label': "Missing values in column", 'value': "columnvaluemissing"}],
                                #                 multi=False,
                                    value="namevalidation",
                                    ),
                                    dash_table.DataTable(
                                    id='table',
                                    columns=[{'name': 'date', 'id': 'date'},
                                             {'name': 'current_time', 'id': 'current_time'},
                                             {'name': 'filename', 'id': 'filename'}],
                                    data=[],

                                )
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

@server.route('/tracker', methods=["POST"])
def watch_dog():
    if request.method == "POST":
        print(request)
        data = request.json
        client = pymongo.MongoClient(cg.Mongodb)
        db = client["tracking_db"]
        conn = db["track_data"]
        conn.insert_one(data)
    return jsonify(status="Done")

@server.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val = pred_validation(path)  # object initialization

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path)  # object initialization

            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % path)
        elif request.form is not None:
            path = request.form['filepath']

            pred_val = pred_validation(path)  # object initialization

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(path)  # object initialization

            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % path)

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


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

@server.route("/track", methods=['POST', 'GET'])
def tracker():
    print(request)
    return jsonify(status1="ok")

@dash_app.callback(Output('tabs-content-props', 'children'),
              Input('tabs-styled-with-props', 'value'))
def render_content(tab):
    if tab == 'tab-1':

        return values.prediction_page()

    elif tab == 'tab-2':
        return values.training_page()


#training

@dash_app.callback(Output('datewise-train', 'figure'),
                    [Input('train-picker-range', 'start_date'),
                     Input('train-picker-range', 'end_date')])
def update_output(start_date, end_date):
    print(start_date, end_date)
    if start_date is not None:
        df = values.datewise_train[(values.datewise_train['date'] >= start_date)]
        date_list = list(df['date'].value_counts().to_dict().keys())
        total_files = list(df['date'].value_counts().to_dict().values())
        fig = {'data': [{'x': date_list, 'y': total_files, 'type': 'bar'}],
               'layout': {'title': 'Training Files by date', "width": 500}}
        return fig
    if end_date is not None:
        df = values.datewise_train[(values.datewise_train['date'] <= end_date)]
        date_list = list(df['date'].value_counts().to_dict().keys())
        total_files = list(df['date'].value_counts().to_dict().values())
        fig = {'data': [{'x': date_list, 'y': total_files, 'type': 'bar'}],
               'layout': {'title': 'Training Files by date', "width": 500}}
        return fig
    else:
        df = values.datewise_train[(values.datewise_train['date'] >= start_date) & (values.datewise_train['date'] <= end_date)]
        date_list = list(df['date'].value_counts().to_dict().keys())
        total_files = list(df['date'].value_counts().to_dict().values())
        fig = {'data': [{'x': date_list, 'y': total_files, 'type': 'bar'}],
               'layout': {'title': 'Training by date', "width": 500}}
        return fig



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
                                 values=value)], layout={"showlegend": False, "height": 340, "width": 460,
                                                         'title': 'Training Bad File Details'})
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
                                     values=[values.good_data_train, values.bad_data_train])],
                        layout={"showlegend": False, "height": 340, "width": 460,
                                'title': 'Training Data Details'}
                        )
    else:
        fig = go.Figure(data=[go.Pie(labels=['Good Files', 'Bad Files'],
                                     values=[values.good_files_train, values.bad_files_train])],
                        layout={"showlegend": False, "height": 340, "width": 460,
                                'title': 'Training File Details'}
                        )

    return fig


# predictions
@dash_app.callback(Output('datewise-pred', 'figure'),
                    [Input('pred-picker-range', 'start_date'),
                     Input('pred-picker-range', 'end_date')])
def update_output(start_date, end_date):
    print(start_date, end_date)
    if start_date is not None:
        df = values.datewise_pred[(values.datewise_pred['date'] >= start_date)]
        date_list = list(df['date'].value_counts().to_dict().keys())
        total_files = list(df['date'].value_counts().to_dict().values())
        fig = {'data': [{'x': date_list, 'y': total_files, 'type': 'bar'}],
               'layout': {'title': 'Prediction Files by date', "width": 500}}
        return fig
    if end_date is not None:
        df = values.datewise_pred[(values.datewise_pred['date'] <= end_date)]
        date_list = list(df['date'].value_counts().to_dict().keys())
        total_files = list(df['date'].value_counts().to_dict().values())
        fig = {'data': [{'x': date_list, 'y': total_files, 'type': 'bar'}],
               'layout': {'title': 'Prediction Files by date', "width": 500}}
        return fig
    else:
        df = values.datewise_pred[(values.datewise_pred['date'] >= start_date) & (values.datewise_pred['date'] <= end_date)]
        date_list = list(df['date'].value_counts().to_dict().keys())
        total_files = list(df['date'].value_counts().to_dict().values())
        fig = {'data': [{'x': date_list, 'y': total_files, 'type': 'bar'}],
               'layout': {'title': 'Training by date', "width": 500}}
        return fig

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
                                 values=value)], layout={"showlegend": False, "height": 340, "width": 460,
                                                         'title': 'Prediction Bad File Details'})
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
                                     values=[values.good_data_pred, values.bad_data_pred])],
                        layout={"showlegend": False, "height": 340, "width": 460,
                                'title': 'Prediction Data Details'}
                        )
    else:
        fig = go.Figure(data=[go.Pie(labels=['Good Files', 'Bad Files'],
                                     values=[values.good_files_pred, values.bad_files_pred])],
                        layout={"showlegend": False, "height": 340, "width": 460,
                                'title': 'Prediction File Details'}
                        )

    return fig


@analyzer_app.callback(
    Output('radio-items2', 'options'),
    Output('radio-items2', 'value'),
    Output('graph2-id', 'figure'),
    [Input('radio-items1', component_property='value')]
)
def update_graph(s_value):
    # print(s_value)
    option = [{'label': i, 'value': i + "_" + str(s_value)} for i in values.model_list_cluster[s_value]]
    value = option[0]['value']
    model_status = pd.DataFrame([file["model_status"] for file in values.model_logs.find({"cluster_no": str(s_value)})][0])
    fig = {'data': [{'x': model_status["models"].to_list(), 'y': model_status["scores"].to_list(), 'type': 'bar'}],
           'layout': {'title': 'Model Status', "width": 500}}
    return option, value, fig


@analyzer_app.callback(
    Output('record-id', 'value'),
    Output('train-id', 'value'),
    Output('test-id', 'value'),
    Output('precision-id', 'value'),
    Output('recall-id', 'value'),
    Output('acc-id', 'value'),
    Output('f1-id', 'value'),
    Output('graph-id', 'figure'),
    [Input('radio-items2', component_property='value')]
)
def update_graph(s_value):
    # print(s_value)
    model_name = s_value.split('_')[0]
    cluster = s_value.split('_')[1]
    fetch_data = values.model_log_data.loc[(values.model_log_data["model_name"] == model_name) & (values.model_log_data["cluster"] == cluster)]
    pickle_data = fetch_data['model_analysis']
    con = pickle.loads(list(pickle_data)[0])
    precision = con["precision"]
    recall = con["recall"]
    f1 = con["f1"]
    try:
        accuracy = int(float(fetch_data["Accuracy_score"]) * 100)
    except:
        accuracy = int(float(fetch_data["AUC_score"]) * 100)
    train_set = int(values.model_log_data.loc[values.model_log_data["cluster_label"] == cluster]["train_set"].to_string(index=False))
    test_set = int(values.model_log_data.loc[values.model_log_data["cluster_label"] == cluster]["test_set"].to_string(index=False))
    records = int(train_set) + int(test_set)

    fig = ff.create_annotated_heatmap(con['conf'], colorscale='Viridis')

    return records, train_set, test_set, precision, recall, accuracy, f1, fig

@Filepage_app.callback(
    Output('table', 'data'),
    [Input('data-collection', component_property='value'),
     Input('file-type', component_property='value')]
)
def update_graph(value, file_type):
    if file_type == "Training":
        df = values.train_files[value]
        data = df.to_dict('records')
        return data
    elif file_type == "Prediction":
        df = values.pred_files[value]
        data=df.to_dict('records')
        return data
    else:
        data = values.track_files.to_dict('records')
        return data

app = DispatcherMiddleware(server, {
    '/dash1': dash_app.server,
    '/dash2': analyzer_app.server,
    '/dash3': Filepage_app.server
})
# port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    # host = '0.0.0.0'
    # port = 5000
    # httpd = simple_server.make_server(host, port, app)
    # # print("Serving on %s %d" % (host, port))
    # httpd.serve_forever()
    run_simple(hostname='0.0.0.0', port=5000, application=app, use_debugger=True, use_reloader=True)
