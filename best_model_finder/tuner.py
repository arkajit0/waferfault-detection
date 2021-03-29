from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,accuracy_score


class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf_rfc = RandomForestClassifier()
        self.clf_dtc = DecisionTreeClassifier()
        self.clf_svc = SVC()
        self.clf_logistic = LogisticRegression()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_params_for_Logistic_Regression(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_Logistic_Regression
                                Description: get the parameters for Logistic Regression Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.db_log(self.file_object,
                                       "Entered the get_best_params_for_Logistic_Regression method of the Model_Finder class")
        try:
            # initializing with different combination of parameters
            # self.param_grid = {'C': [0.001,0.01,0.1,1,10,100]}
            #
            # #Creating an object of the Grid Search class
            # self.grid = GridSearchCV(estimator=self.clf_logistic, param_grid=self.param_grid, cv=5, refit=True,
            #                          n_jobs=-1, verbose=3)
            # #finding the best parameters
            # self.grid.fit(train_x, train_y)

            #creating a new model with the best parameters
            self.clf = LogisticRegression()
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.db_log(self.file_object,
                                      "Logistic Regression best params: "+str(self.grid.best_params_)+". Exited the get_best_params_for_Logistic_Regression method of the Model_Finder class")

            return self.clf
        except Exception as e:
            self.logger_object.db_log(self.file_object, "Exception occured in get_best_params_for_Logistic_Regression method of the Model_Finder class. Exception message:  " + str(
                                       e))
            self.logger_object.db_log(self.file_object,
                                      "Logistic Regression Parameter tuning  failed. Exited the get_best_params_for_Logistic_Regression method of the Model_Finder class")
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
            # # create best model for  Logistic_Regression
            # self.lr = self.get_best_params_for_Logistic_Regression(train_x, train_y)
            # self.prediction_lr = self.lr.predict(test_x)  # Predictions using the Logistic_Regression
            #
            # if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
            #     self.lr_score = accuracy_score(test_y, self.prediction_lr)
            #     self.logger_object.db_log(self.file_object, "Accuracy for lr:" + str(self.lr_score))  # Log AUC
            # else:
            #     self.lr_score = roc_auc_score(test_y, self.prediction_lr)  # AUC for Logistic_Regression
            #     self.logger_object.db_log(self.file_object,"AUC for Logistic_Regression:" + str(
            #                                   self.svc_score))  # Log AUC

            # create best model for  SVC
            self.svc = self.get_best_params_for_SVC(train_x, train_y)
            self.prediction_svc = self.svc.predict(test_x)  # Predictions using the SVC Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.svc_score = accuracy_score(test_y, self.prediction_svc)
                self.logger_object.db_log(self.file_object, "Accuracy for svc:" + str(self.svc_score))  # Log AUC
            else:
                self.svc_score = roc_auc_score(test_y, self.prediction_svc)  # AUC for SVC
                # fpr, tpr, thresholds = roc_curve(test_y, self.prediction_svc)
                self.logger_object.db_log(self.file_object, "AUC for svc:" + str(
                                              self.svc_score))  # Log AUC



            # create best model for Decision tree
            self.decisiontree = self.get_best_params_for_Decision_Tree(train_x, train_y)
            self.prediction_decisiontree = self.decisiontree.predict(test_x)  # Predictions using the Decision tree Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.decisiontree_score = accuracy_score(test_y, self.prediction_decisiontree)
                self.logger_object.db_log(self.file_object, "Accuracy for decisiontree:" + str(self.decisiontree_score))  # Log AUC
            else:
                self.decisiontree_score = roc_auc_score(test_y, self.prediction_decisiontree)  # AUC for Decision tree
                self.logger_object.db_log(self.file_object, "AUC for Decision tree:" + str(self.decisiontree_score))  # Log AUC

            # create best model for XGBoost
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.db_log(self.file_object, "Accuracy for XGBoost:" + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.db_log(self.file_object, "AUC for XGBoost:" + str(self.xgboost_score)) # Log AUC


            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                self.logger_object.db_log(self.file_object, "Accuracy for RF:" + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest) # AUC for Random Forest
                self.logger_object.db_log(self.file_object, "AUC for RF:" + str(self.random_forest_score))


            #comparison of models
            model_name = [("XGBoost", self.xgboost, self.xgboost_score), ("RandomForest", self.random_forest, self.random_forest_score),
                          ("DecisionTree", self.decisiontree, self.decisiontree_score), ("SVC", self.svc, self.svc_score)]
            model_scores = [self.xgboost_score, self.random_forest_score, self.decisiontree_score, self.svc_score]

            best_model = model_name[model_scores.index(max(model_scores))]


            return best_model[0], best_model[1], best_model[2]
            # #comparing the two models
            # if(self.random_forest_score <  self.xgboost_score):
            #     return 'XGBoost',self.xgboost
            # else:
            #     return 'RandomForest',self.random_forest

        except Exception as e:
            self.logger_object.db_log(self.file_object, "Exception occured in get_best_model method of the Model_Finder class. Exception message:  " + str(
                                       e))
            self.logger_object.db_log(self.file_object, "Model Selection Failed. Exited the get_best_model method of the Model_Finder class")
            raise Exception()

