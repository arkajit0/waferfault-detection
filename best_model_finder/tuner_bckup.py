from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,accuracy_score, roc_curve, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import pandas as pd


class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object, cluster_no):
        self.file_object = file_object
        self.logger_object = logger_object
        self.cluster_no = cluster_no
        self.clf_rfc = RandomForestClassifier()
        self.clf_dtc = DecisionTreeClassifier()
        self.clf_svc = SVC()
        self.knn = KNeighborsClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_params_for_KNN(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_KNN
                                                Description: get the parameters for KNN Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.db_log(self.file_object,
                                  'Entered the get_best_params_for_Ensembled_KNN method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_knn = {
                'algorithm': ['ball_tree'],
                'leaf_size': [10, 17, 24],
                # 'n_neighbors':[4,5,8],
                # 'p':[1,2]
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.knn, self.param_grid_knn, verbose=3,
                                     cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.algorithm = self.grid.best_params_['algorithm']
            self.leaf_size = self.grid.best_params_['leaf_size']
            # self.n_neighbors = self.grid.best_params_['n_neighbors']
            # self.p  = self.grid.best_params_['p']

            # creating a new model with the best parameters
            self.knn = KNeighborsClassifier(algorithm=self.algorithm, leaf_size=self.leaf_size, n_jobs=-1)
            # training the mew model
            self.knn.fit(train_x, train_y)
            self.logger_object.db_log(self.file_object,
                                      'KNN best params: ' + str(
                                          self.grid.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return self.knn
        except Exception as e:
            self.logger_object.db_log(self.file_object,
                                      'Exception occured in knn method of the Model_Finder class. Exception message:  ' + str(
                                          e))
            self.logger_object.db_log(self.file_object,
                                      'knn Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_SVC(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_SVC
                                Description: get the parameters for SVC Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.db_log(self.file_object,
                                  {"Get SVC best model parameter": "Entered the get_best_params_for_SVC method of the Model_Finder class"})
        try:
            # initializing with different combination of parameters
            self.param_grid = {'C': [0.1, 1, 10, 100, 1000],
                              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                              'kernel': ['rbf']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf_svc, param_grid=self.param_grid, cv=5, refit = True,
                                     n_jobs=-1, verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #creating a new model with the best parameters
            self.clf = SVC(**self.grid.best_params_)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.db_log(self.file_object,
                                      "SVC best params: "+str(self.grid.best_params_)+". Exited the get_best_params_for_SVC method of the Model_Finder class")

            return self.clf
        except Exception as e:
            self.logger_object.db_log(self.file_object, "Exception occured in get_best_params_for_SVC method of the Model_Finder class. Exception message:  " + str(
                                       e))
            self.logger_object.db_log(self.file_object,
                                      "SVC Parameter tuning  failed. Exited the get_best_params_for_SVC method of the Model_Finder class")
            raise Exception()


    def get_best_params_for_Decision_Tree(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_Decision_Tree
                                Description: get the parameters for Decision Tree Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.db_log(self.file_object,
                                  {"Get Random Forest best model parameter": "Entered the get_best_params_for_random_forest method of the Model_Finder class"})
        try:
            # initializing with different combination of parameters
            self.param_grid = {'criterion': ['gini', 'entropy'],
                                'splitter': ['best', 'random'],
                                'max_depth': [5, 10, 15, 20],
                                'min_samples_split': range(2, 5, 1),
                                'min_samples_leaf': range(1, 5, 1)}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf_dtc, param_grid=self.param_grid, cv=5, n_jobs=-1, verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            # #extracting the best parameters
            # self.criterion = self.grid.best_params_['criterion']
            # self.max_depth = self.grid.best_params_['max_depth']
            # self.max_features = self.grid.best_params_['max_features']
            # self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = DecisionTreeClassifier(**self.grid.best_params_)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.db_log(self.file_object,
                                      "Decision Tree best params: "+str(self.grid.best_params_)+". Exited the get_best_params_for_Decision_Tree method of the Model_Finder class")

            return self.clf
        except Exception as e:
            self.logger_object.db_log(self.file_object, "Exception occured in get_best_params_for_Decision_Tree method of the Model_Finder class. Exception message:  " + str(
                                       e))
            self.logger_object.db_log(self.file_object,
                                      "Decision Tree Parameter tuning  failed. Exited the get_best_params_for_Decision_Tree method of the Model_Finder class")
            raise Exception()

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.db_log(self.file_object,
                                  {"Get Random Forest best model parameter": "Entered the get_best_params_for_random_forest method of the Model_Finder class"})
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf_rfc, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.db_log(self.file_object,
                                      "Random Forest best params: "+str(self.grid.best_params_)+". Exited the get_best_params_for_random_forest method of the Model_Finder class")

            return self.clf
        except Exception as e:
            self.logger_object.db_log(self.file_object,
                                      "Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  " + str(
                                       e))
            self.logger_object.db_log(self.file_object,
                                      "Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class")
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.db_log(self.file_object,
                                  {"Get Xgboost Model best Model parameter":
                                              "Entered the get_best_params_for_xgboost method of the Model_Finder class"})
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.db_log(self.file_object, "XGBoost best params: " + str(
                                       self.grid.best_params_) + ". Exited the get_best_params_for_xgboost method of the Model_Finder class")
            return self.xgb
        except Exception as e:
            self.logger_object.db_log(self.file_object, "Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  " + str(
                                       e))
            self.logger_object.db_log(self.file_object,
                                                  "XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class")
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.db_log(self.file_object,
                                              "Entered the get_best_model method of the Model_Finder class")

        try:
            # create best model for  KNN
            self.knn = self.get_best_params_for_KNN(train_x, train_y)
            self.prediction_knn = self.knn.predict(test_x)  # Predictions using the knn Model

            if len(
                    test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.knn_score = accuracy_score(test_y, self.prediction_knn)
                precision, recall, f1 = precision_score(test_y, self.prediction_knn), \
                                        recall_score(test_y, self.prediction_knn), f1_score(test_y, self.prediction_knn)
                conf_matrix = confusion_matrix(test_y, self.prediction_knn)
                model_analysis_dict = {"precision": precision, "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"Accuracy_score": str(self.knn_score), "model_name": "KNN",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})
            else:
                self.knn_score = roc_auc_score(test_y, self.prediction_knn)  # AUC for KNN
                fpr, tpr, thresholds = roc_curve(test_y, self.prediction_knn)
                precision, recall, f1 = precision_score(test_y, self.prediction_knn), \
                                        recall_score(test_y, self.prediction_knn), f1_score(test_y, self.prediction_knn)
                conf_matrix = confusion_matrix(test_y, self.prediction_knn)
                model_analysis_dict = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "precision": precision,
                                       "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"AUC_score": str(self.knn_score), "model_name": "KNN",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})


            # create best model for  SVC
            self.svc = self.get_best_params_for_SVC(train_x, train_y)
            self.prediction_svc = self.svc.predict(test_x)  # Predictions using the SVC Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.svc_score = accuracy_score(test_y, self.prediction_svc)
                precision, recall, f1 = precision_score(test_y, self.prediction_svc), \
                                        recall_score(test_y, self.prediction_svc), f1_score(test_y, self.prediction_svc)
                conf_matrix = confusion_matrix(test_y, self.prediction_svc)
                model_analysis_dict = {"precision": precision, "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"Accuracy_score": str(self.svc_score), "model_name": "SVC",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})
            else:
                self.svc_score = roc_auc_score(test_y, self.prediction_svc)  # AUC for SVC
                fpr, tpr, thresholds = roc_curve(test_y, self.prediction_svc)
                precision, recall, f1 = precision_score(test_y, self.prediction_svc), \
                                        recall_score(test_y, self.prediction_svc), f1_score(test_y, self.prediction_svc)
                conf_matrix = confusion_matrix(test_y, self.prediction_svc)
                model_analysis_dict = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "precision": precision,
                                       "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"AUC_score": str(self.svc_score), "model_name": "SVC",
                                                             "cluster": str(self.cluster_no), "model_analysis": pickled_model})



            # create best model for Decision tree
            self.decisiontree = self.get_best_params_for_Decision_Tree(train_x, train_y)
            self.prediction_decisiontree = self.decisiontree.predict(test_x)  # Predictions using the Decision tree Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.decisiontree_score = accuracy_score(test_y, self.prediction_decisiontree)
                precision, recall, f1 = precision_score(test_y, self.prediction_decisiontree), \
                                        recall_score(test_y, self.prediction_decisiontree), f1_score(test_y, self.prediction_decisiontree)
                conf_matrix = confusion_matrix(test_y, self.prediction_decisiontree)
                model_analysis_dict = {"precision": precision, "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"Accuracy_score": str(self.decisiontree_score),
                                                             "model_name": "DecisionTree",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})
            else:
                self.decisiontree_score = roc_auc_score(test_y, self.prediction_decisiontree)  # AUC for Decision tree
                fpr, tpr, thresholds = roc_curve(test_y, self.prediction_decisiontree)
                precision, recall, f1 = precision_score(test_y, self.prediction_decisiontree), \
                                        recall_score(test_y, self.prediction_decisiontree), f1_score(test_y, self.prediction_decisiontree)
                conf_matrix = confusion_matrix(test_y, self.prediction_decisiontree)
                model_analysis_dict = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "precision": precision,
                                       "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"AUC_score": str(self.decisiontree_score),
                                                             "model_name": "DecisionTree",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})

            # create best model for XGBoost
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                precision, recall, f1 = precision_score(test_y, self.prediction_xgboost), \
                                        recall_score(test_y, self.prediction_xgboost), f1_score(test_y,
                                                                                                self.prediction_xgboost)
                conf_matrix = confusion_matrix(test_y, self.prediction_xgboost)
                model_analysis_dict = {"precision": precision, "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"Accuracy_score": str(self.xgboost_score), "model_name": "XGBoost",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                fpr, tpr, thresholds = roc_curve(test_y, self.prediction_xgboost)
                precision, recall, f1 = precision_score(test_y, self.prediction_xgboost), \
                                        recall_score(test_y, self.prediction_xgboost), f1_score(test_y,
                                                                                                self.prediction_xgboost)
                conf_matrix = confusion_matrix(test_y, self.prediction_xgboost)
                model_analysis_dict = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "precision": precision,
                                       "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"AUC_score": str(self.xgboost_score), "model_name": "XGBoost",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})


            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                precision, recall, f1 = precision_score(test_y, self.prediction_random_forest), \
                                        recall_score(test_y, self.prediction_random_forest), f1_score(test_y,
                                                                                                self.prediction_random_forest)
                conf_matrix = confusion_matrix(test_y, self.prediction_random_forest)
                model_analysis_dict = {"precision": precision, "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"Accuracy_score": str(self.random_forest_score), "model_name": "RandomForest",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest) # AUC for Random Forest
                fpr, tpr, thresholds = roc_curve(test_y, self.prediction_random_forest)
                precision, recall, f1 = precision_score(test_y, self.prediction_random_forest), \
                                        recall_score(test_y, self.prediction_random_forest), f1_score(test_y,
                                                                                                self.prediction_random_forest)
                conf_matrix = confusion_matrix(test_y, self.prediction_random_forest)
                model_analysis_dict = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "precision": precision,
                                       "recall": recall, "f1": f1, "conf": conf_matrix}
                pickled_model = pickle.dumps(model_analysis_dict)
                self.logger_object.db_log(self.file_object, {"AUC_score": str(self.random_forest_score), "model_name": "RandomForest",
                                                             "cluster": str(self.cluster_no),
                                                             "model_analysis": pickled_model})

                # comparison of models
                model_name = [("XGBoost", self.xgboost, self.xgboost_score),
                              ("RandomForest", self.random_forest, self.random_forest_score),
                              ("DecisionTree", self.decisiontree, self.decisiontree_score),
                              ("SVC", self.svc, self.svc_score), ("KNN", self.knn, self.knn_score)]

                model_scores = [self.xgboost_score, self.random_forest_score, self.decisiontree_score, self.svc_score,
                                self.knn_score]

                best_model = model_name[model_scores.index(max(model_scores))]

                model_name_store = ["XGBoost", "RandomForest", "DecisionTree", "SVC", "KNN"]
                model_scores_store = [int(self.xgboost_score * 100), int(self.random_forest_score * 100),
                                      int(self.decisiontree_score * 100),
                                      int(self.svc_score * 100), int(self.knn_score * 100)]
                self.logger_object.db_log(self.file_object,
                                          {"model_status": pd.DataFrame(
                                              {"models": model_name_store, "scores": model_scores_store}).to_dict(
                                              'records'),
                                           "cluster_no": str(self.cluster_no), "Best_model": best_model[0]})

                return best_model[0], best_model[1], best_model[2]


        except Exception as e:
            self.logger_object.db_log(self.file_object, "Exception occured in get_best_model method of the Model_Finder class. Exception message:  " + str(
                                       e))
            self.logger_object.db_log(self.file_object, "Model Selection Failed. Exited the get_best_model method of the Model_Finder class")
            raise Exception()

