import config as cfg
import pymongo
import dash_daq as daq
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, MATCH
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import numpy as np
import pandas as pd



class Pages:
    def __init__(self):
        self.client = pymongo.MongoClient(cfg.Mongodb)
        #training db
        self.db_train = self.client["Data_Collection"]
        self.train_logs = self.client["Training_Logs"]
        try:
            self.train_nameerror = \
            pd.DataFrame(self.train_logs["nameValidationLog"].find())["status"].value_counts().to_dict()["failure"]
        except:
            self.train_nameerror = 0
        try:
            self.train_missingvalues = \
            pd.DataFrame(self.train_logs["missingValuesInColumn"].find())["status"].value_counts().to_dict()["failure"]
        except:
            self.train_missingvalues = 0
        try:
            self.train_columnerror = \
            pd.DataFrame(self.train_logs["columnValidationLog"].find())["status"].value_counts().to_dict()["failure"]
        except:
            self.train_columnerror = 0
        self.bad_data_train_collection = self.db_train["Bad_Data"]
        self.good_data_train_collection = self.db_train["Good_Data"]
        self.bad_files_train = self.bad_data_train_collection.count()
        self.good_files_train = self.good_data_train_collection.count()
        self.bad_data_train = sum([len(i['csv_data']) for i in self.bad_data_train_collection.find()])
        self.good_data_train = sum([len(i['csv_data']) for i in self.good_data_train_collection.find()])

        #prediction db
        self.db_pred = self.client["Data_Collection_Prediction"]
        self.pred_logs = self.client["Prediction_Logs"]
        try:
            self.pred_nameerror = pd.DataFrame(self.pred_logs["nameValidationLog"].find())["status"].value_counts().to_dict()["failure"]
        except:
            self.pred_nameerror = 0
        try:
            self.pred_missingvalues = pd.DataFrame(self.pred_logs["missingValuesInColumn"].find())["status"].value_counts().to_dict()["failure"]
        except:
            self.pred_missingvalues = 0
        try:
            self.pred_columnerror = pd.DataFrame(self.pred_logs["columnValidationLog"].find())["status"].value_counts().to_dict()["failure"]
        except:
            self.pred_columnerror = 0
        self.bad_data_pred_collection = self.db_pred["Bad_Data"]
        self.good_data_pred_collection = self.db_pred["Good_Data"]
        self.bad_files_pred = self.bad_data_pred_collection.count()
        self.good_files_pred = self.good_data_pred_collection.count()
        self.bad_data_pred = sum([len(i['csv_data']) for i in self.bad_data_pred_collection.find()])
        self.good_data_pred = sum([len(i['csv_data']) for i in self.good_data_pred_collection.find()])
        self.model_db = self.client["Models"]
        self.avg_score = np.mean(
            [self.model_db[coll].find_one({}, {"_id": 0, "score": 1})['score'] for coll in self.model_db.list_collection_names() if
             'score' in self.model_db[coll].find_one({}, {"_id": 0, "score": 1})])
        self.pred_report = pd.DataFrame(self.client["Prediction_Files"]["Prediction_Result"].find())
        self.predicted_report = self.pred_report["Prediction"].value_counts().to_dict()

    def training_page(self):
        date = ["24-03-2021", "25-03-2021", "26-03-2021", "27-03-2021"]
        data = [234, 567, 568, 899]

        html_page_upper = html.Div([
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [{'x': date, 'y': data, 'type': 'bar'}],
                        'layout': {'title': 'Prediction by date'}
                    }
                )
            ], style={
                "height": "340px",
                "width": "50%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            }),
            html.Div([
                dcc.Dropdown(
                    id={
                        'type': 'drop-down-train-a',
                    },
                    options=[{'label': "DATA", 'value': "data"},
                             {'label': "FILES", 'value': "file"},
                             ],
                    multi=False,
                    value="data",
                    style={
                        "height": "40px",
                        "width": "100px",
                        "top": "0",
                        "right": "100%",
                        "verticalAlign": "middle"

                    }
                ),
                dcc.Graph(
                    id={
                        'type': 'graph-train-a',

                    },
                    figure={}
                )

            ], style={
                "height": "340px",
                "width": "50%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            })
        ], style={"height": "360px",
                  "width": "100%",
                  'border': '3px solid green',
                  'display': 'flex',
                  'padding': '10px'

                  })

        html_page_lower = html.Div([
            html.Div([
                dcc.Graph(
                    figure=go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=int(self.avg_score * 100),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [None, 100]}},
                        title={'text': "Avg. Prediction Score"}),
                        layout={"width": 400}),

                )

            ], style={
                "height": "340px",
                "width": "30%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            }),
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [{'x': ["Defected Wafer", "Undefected Wafer"],
                                  'y': [self.predicted_report[-1], self.predicted_report[1]], 'type': 'bar'}],
                        'layout': {'title': 'Predictions',
                                   'width': "400"},
                    }
                )
            ], style={
                "height": "340px",
                "width": "30%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            }),
            html.Div([
                dcc.Dropdown(
                    id={
                        'type': 'drop-down-train-b',

                    },
                    options=[{'label': "Wrong File Name", 'value': "namevalidation"},
                             {'label': "Wrong Column Length", 'value': "columnvalidation"},
                             {'label': "Missing values in column", 'value': "columnvaluemissing"}],
                    multi=True,
                    value=["namevalidation", "columnvalidation"],
                    style={

                        'width': "50%",

                    }
                ),
                dcc.Graph(
                    id={
                        'type': 'graph-train-b',

                    },
                    figure={}
                )

            ], style={
                "height": "340px",
                "width": "40%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            })
        ], style={"height": "360px",
                  "width": "100%",
                  'border': '3px solid green',
                  'display': 'flex',
                  'padding': '10px'

                  })

        return [html_page_upper, html_page_lower]


    def prediction_page(self):
        date = ["24-03-2021", "25-03-2021", "26-03-2021", "27-03-2021"]
        data = [234, 567, 568, 899]


        html_page_upper = html.Div([
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [{'x': date, 'y': data, 'type': 'bar'}],
                        'layout': {'title': 'Prediction by date'}
                            }
                        )
                    ], style={
                                "height": "340px",
                                "width": "50%",
                                'border': '3px solid green',
                                'display': 'flex',
                                'padding': '10px'
                            }),
            html.Div([
                dcc.Dropdown(
                    id={
                        'type': 'drop-down-pred-a',
                    },
                    options=[{'label': "DATA", 'value': "data"},
                             {'label': "FILES", 'value': "file"},
                             ],
                    multi=False,
                    value="data",
                    style={
                        "height": "40px",
                        "width": "100px",
                        "top": "0",
                        "right": "100%",
                        "verticalAlign": "middle"

                    }
                ),
                dcc.Graph(
                    id={
                        'type': 'graph-pred-a',

                    },
                    figure={}
                )

            ], style={
                "height": "340px",
                "width": "50%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            })
        ], style={"height": "360px",
                  "width": "100%",
                  'border': '3px solid green',
                  'display': 'flex',
                  'padding': '10px'

                  })

        html_page_lower = html.Div([
            html.Div([
                dcc.Graph(
                    figure=go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=int(self.avg_score * 100),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [None, 100]}},
                        title={'text': "Avg. Prediction Score"}),
                        layout={"width": 400}),

                )

            ], style={
                "height": "340px",
                "width": "30%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            }),
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [{'x': ["Defected Wafer", "Undefected Wafer"],
                                  'y': [self.predicted_report[-1], self.predicted_report[1]], 'type': 'bar'}],
                        'layout': {'title': 'Predictions',
                                   'width': "400"},
                    }
                )
            ], style={
                "height": "340px",
                "width": "30%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            }),
            html.Div([
                #             dcc.Graph(
                #                 id={
                #                     'type': 'graph-pred-b',

                #                 },
                #                 figure={}
                #             ),
                dcc.Dropdown(
                    id={
                        'type': 'drop-down-pred-b',

                    },
                    options=[{'label': "Wrong File Name", 'value': "namevalidation"},
                             {'label': "Wrong Column Length", 'value': "columnvalidation"},
                             {'label': "Missing values in column", 'value': "columnvaluemissing"}],
                    multi=True,
                    value=["namevalidation", "columnvalidation"],
                    style={

                        'width': "50%",

                    }
                ),
                dcc.Graph(
                    id={
                        'type': 'graph-pred-b',

                    },
                    figure={}
                )

            ], style={
                "height": "340px",
                "width": "40%",
                'border': '3px solid green',
                'display': 'flex',
                'padding': '10px'
            })
        ], style={"height": "360px",
                  "width": "100%",
                  'border': '3px solid green',
                  'display': 'flex',
                  'padding': '10px'

                  })

        return [html_page_upper, html_page_lower]
