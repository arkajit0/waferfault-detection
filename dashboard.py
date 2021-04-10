import config as cfg
from datetime import date
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
    """
                        This class shall be used for analyzing logs in database


                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                        """

    def __init__(self):
        self.client = pymongo.MongoClient(cfg.Mongodb)
        #training db
        self.db_train = self.client["Data_Collection"]
        self.train_logs = self.client["Training_Logs"]
        self.datewise_train = pd.DataFrame(self.train_logs["nameValidationLog"].find())
        try:
            self.train_nameerror = self.datewise_train["status"].value_counts().to_dict()["failure"]
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

        self.nameval_train = pd.DataFrame(self.train_logs["nameValidationLog"].find({'status': 'failure'},
                                                                    {'_id': 0, 'date': 1, 'current_time': 1,
                                                                     'filename': 1}))
        self.missingsval_train = pd.DataFrame(self.train_logs["missingValuesInColumn"].find({'status': 'failure'},
                                                                        {'_id': 0, 'date': 1, 'current_time': 1,
                                                                         'filename': 1}))
        self.colval_train = pd.DataFrame(self.train_logs["columnValidationLog"].find({'status': 'failure'},
                                                                     {'_id': 0, 'date': 1, 'current_time': 1,
                                                                      'filename': 1}))
        self.train_files = {"namevalidation": self.nameval_train, "columnvalidation": self.colval_train,
                            "columnvaluemissing": self.missingsval_train}


        #prediction db
        self.db_pred = self.client["Data_Collection_Prediction"]
        self.pred_logs = self.client["Prediction_Logs"]
        self.datewise_pred = pd.DataFrame(self.pred_logs["nameValidationLog"].find())
        try:
            self.pred_nameerror = self.datewise_pred["status"].value_counts().to_dict()["failure"]
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

        self.nameval_pred = pd.DataFrame(self.pred_logs["nameValidationLog"].find({'status': 'failure'},
                                                                                    {'_id': 0, 'date': 1,
                                                                                     'current_time': 1,
                                                                                     'filename': 1}))
        self.missingsval_pred = pd.DataFrame(self.pred_logs["missingValuesInColumn"].find({'status': 'failure'},
                                                                                            {'_id': 0, 'date': 1,
                                                                                             'current_time': 1,
                                                                                             'filename': 1}))
        self.colval_pred = pd.DataFrame(self.pred_logs["columnValidationLog"].find({'status': 'failure'},
                                                                                     {'_id': 0, 'date': 1,
                                                                                      'current_time': 1,
                                                                                      'filename': 1}))
        self.pred_files = {"namevalidation": self.nameval_pred, "columnvalidation": self.colval_pred,
                            "columnvaluemissing": self.missingsval_pred}
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

        #model db
        self.model_logs = self.train_logs["ModelTrainingLog"]
        self.clusters = int(pd.DataFrame(self.model_logs.find({"cluster_name": "KMeans"}))["clusters_total"])
        self.model_log_data = pd.DataFrame(self.model_logs.find())
        self.model_list_cluster = [list(self.model_log_data[self.model_log_data["cluster"] == str(i)]["model_name"]) for
                                   i in range(0, self.clusters)]
        self.total_train_set = self.model_log_data[self.model_log_data["train_set"].notna()]["train_set"].astype('int').sum()
        self.total_test_set = self.model_log_data[self.model_log_data["test_set"].notna()]["test_set"].astype('int').sum()

        #FileTracker
        self.tracker_db = self.client["tracking_db"]
        self.track_col = self.tracker_db["track_data"]
        self.track_files = pd.DataFrame(self.track_col.find({}, {"_id": 0, "timestamp": 1, "filename": 1}))
        self.track_files["timestamp"] = pd.to_datetime(self.track_files["timestamp"])
        self.track_files["date"] = [data.date() for data in self.track_files["timestamp"]]
        self.track_files["current_time"] = [data.time() for data in self.track_files["timestamp"]]

    def training_page(self):
        """
                            Method Name: training_page
                            Description: Analysis of training data


                            Written By: iNeuron Intelligence
                            Version: 1.0
                            Revisions: None
                """

        html_page_upper = html.Div([
            html.Div([
                dcc.DatePickerRange(
                    id='train-picker-range',
                    min_date_allowed=date(2021, 3, 13),
                    max_date_allowed=date(2021, 10, 10),
                    initial_visible_month=date(2021, 1, 13),
                    start_date = date(2021, 1, 13),
                    end_date = date(2021, 7, 13),
                    style = {"height": "200px"}
                ),
                dcc.Graph(
                    id = 'datewise-train',
                    figure={}
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
                        "width": "50%",

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
                        title={'text': "Avg. Training Score"}),
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
                html.H1("Total Records", style={"text-align": "center"}),
                html.Br(),
                daq.LEDDisplay(

                    label="Training Data",
                    value=self.total_train_set,
                    color="#FF5E5E",
                    style={"display": "inline-block", "margin-left": 80}

                ),
                daq.LEDDisplay(

                    label="Testing Data",
                    value=self.total_test_set,
                    color="#FF5E5E",
                    style={"display": "inline-block"}

                )
            ], style={
                "height": "340px",
                "width": "30%",
                'border': '3px solid green',

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
        """
                                    Method Name: prediction_page
                                    Description: Analysis of prediction data


                                    Written By: iNeuron Intelligence
                                    Version: 1.0
                                    Revisions: None
                        """

        html_page_upper = html.Div([
            html.Div([
                dcc.DatePickerRange(
                    id='pred-picker-range',
                    min_date_allowed=date(2021, 3, 13),
                    max_date_allowed=date(2021, 10, 10),
                    initial_visible_month=date(2021, 1, 13),
                    start_date=date(2021, 1, 13),
                    end_date=date(2021, 7, 13),
                    style={"height": "200px"}
                ),
                dcc.Graph(
                    id='datewise-pred',
                    figure={}
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
                        'width': "70%",

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

                dcc.Link(
                    html.Button('Model Analyze'),
                    id="navigate-to-analyze-page", href='/analyze',
                    style={"display": "inline-block"},
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

    def Model_Analyzer(self):
        """
                                            Method Name: Model_Analyzer
                                            Description: Analysis of Models


                                            Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None
                                """

        analyzer_page = html.Div([
                        html.H1("Model Performance Analyzer", style={'text-align': 'center'}),
                        html.H4("Different Clusters"),
                        dcc.RadioItems(
                    id='radio-items1',
                    options=[{'label': i, 'value': i} for i in range(0, self.clusters)],
                    value=0,
                    labelStyle={'display': 'inline-block'}
                ),
                    html.H4("Models According to Different Clusters"),
                        dcc.RadioItems(
                    id='radio-items2',
                    options=[],
                    value={},
                    labelStyle={'display': 'inline-block'}
                        ),

                       html.Div([
                           #1st Div
                           html.Div([
                               daq.LEDDisplay(
                                       id = "record-id",
                                        label="Record",
                                        value={},
                                        color="#FF5E5E"
                                    ),
                               daq.LEDDisplay(
                                   id = "train-id",
                                        label="Train",
                                        value={},
                                        color="#FF5E5E",
                                       style={"display": "inline-block"}


                                    ),
                               daq.LEDDisplay(
                                           id = 'test-id',
                                        label="Train",
                                        value={},
                                        color="#FF5E5E",
                                           style={"display": "inline-block"}
                                    ),
                               daq.LEDDisplay(
                                           id = 'acc-id',
                                        label="Accuracy",
                                        value={},
                                        color="#FF5E5E",

                                    ),
                               daq.LEDDisplay(
                                           id = 'precision-id',
                                        label="Precision Score",
                                        value={},
                                        color="#FF5E5E",
                                           style={"display": "inline-block"}
                                    ),
                               daq.LEDDisplay(
                                           id = 'recall-id',
                                        label="Recall Score",
                                        value={},
                                        color="#FF5E5E",
                                           style={"display": "inline-block"}
                                    ),
                               daq.LEDDisplay(
                                           id = 'f1-id',
                                        label="F1 Score",
                                        value={},
                                        color="#FF5E5E",
                                           style={"display": "inline-block"}
                                    ),

                           ], style = {"height": "100%", "width": "30%", 'border': '3px solid green', 'padding': '10px',
                                             "margin-left": "2px",
                                        "margin-right": "2px", "margin-top": "2px", }),
                           #2nd div
                            html.Div([
                                html.H4("Confusion Matrix", style={"text-align": "center"}),

                                dcc.Graph(
                            id='graph-id',
                            figure={},
                            config={
                                    'displayModeBar': False
                                },
                                    style = {
                                        "margin-top": "50px"
                                    }



                        )


                                ],


                                style = {"height": "100%", "width": "30%", 'border': '3px solid green', 'padding': '10px',"margin-left": "2px",
                                        "margin-right": "2px", "margin-top": "2px", }),

                           #3rd div
                            html.Div([

                                dcc.Graph(
                                    id='graph2-id',
                                    figure={}
                                )],
                                    style = {"height": "100%", "width": "30%", 'border': '3px solid green', 'padding': '10px',"margin-left": "2px",
                                        "margin-right": "2px", "margin-top": "2px", }),


                       ],style = {"height": "720px",
                                    "width": "100%",
                                    'border': '3px solid green',
                                         'display': 'flex',
                                         'padding': '10px'



                                }

                    )
                ])
        return analyzer_page
