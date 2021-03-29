"""
This is the Entry point for Training the Machine Learning Model.

Written By: iNeuron Intelligence
Version: 1.0
Revisions: None

"""


# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing, clustering
from best_model_finder import tuner
# from application_logging import logger
from application_logging.DB_logger import DB_Logs
from Training_File_DB_operations.DataTypeValidation_db_insertion import DB_Operation, Model_saving_loading
from Mail_box import Send_mail

#Creating the common Logging object


class trainModel:

    def __init__(self):
        self.log_writer = DB_Logs(database_name="Training_Logs")
        self.db_object = "ModelTrainingLog"
        self.mail_log_obj = "MailSendLog"
        self.db_ops = DB_Operation(logger=self.log_writer)
        self.sendmail = Send_mail(self.mail_log_obj, self.log_writer, 'Training status')
        self.model_ops = Model_saving_loading(database_name="Models")

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.db_log(self.db_object, "Start of Training")
        try:
            # Getting the data from the source
            data_getter= data_loader.Data_Getter(self.db_object, self.log_writer)
            data=data_getter.get_data(Database_name="Training_Files", collection_name="Training_data")

            # print(data)
            """doing the data preprocessing"""
            preprocessor= preprocessing.Preprocessor(self.db_object, self.log_writer)
            data=preprocessor.remove_columns(data, ['Wafer']) # remove the unnamed column as it doesn't contribute to prediction.

            # create separate features and labels
            # data.rename(columns={"Good/Bad": "Output"}, inplace=True)
            X,Y=preprocessor.separate_label_feature(data, label_column_name='Output')

            # check if missing values are present in the dataset
            is_null_present=preprocessor.is_null_present(X, database_name="Data_Collection", collection_name="Null_values")

            # if missing values are there, replace them appropriately.
            if(is_null_present):
                X=preprocessor.impute_missing_values(X) # missing value imputation

            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # and they are giving the same output both for good and bad sensors
            # prepare the list of such columns to drop
            cols_to_drop=preprocessor.get_columns_with_zero_std_deviation(X)

            # drop the columns obtained above
            X=preprocessor.remove_columns(X,cols_to_drop)

            """ Applying the clustering approach"""

            kmeans= clustering.KMeansClustering(self.db_object, self.log_writer) # object initialization.
            number_of_clusters=kmeans.elbow_plot(X)  #  using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X=kmeans.create_clusters(X,number_of_clusters)

            #create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels']=Y

            # getting the unique clusters from our dataset
            list_of_clusters=X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data=X[X['Cluster']==i] # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
                cluster_label= cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)

                model_finder= tuner.Model_Finder(self.db_object, self.log_writer) # object initialization

                #getting the best model for each of the clusters
                best_model_name,best_model, model_score=model_finder.get_best_model(x_train,y_train,x_test,y_test)

                #saving the best model to the directory.
                # file_op = file_methods.File_Operation(self.db_object,self.log_writer)
                save_model=self.model_ops.save_model_db(model=best_model,model_name=best_model_name+str(i),
                                                        collection_name=best_model_name+str(i), model_score=model_score)

            # logging the successful Training
            self.log_writer.db_log(self.db_object, "Successful End of Training")
            # self.file_object.close()
            # self.sendmail.send_mail_content("bad-data")
        except Exception:
            # logging the unsuccessful Training
            self.log_writer.db_log(self.db_object, "Unsuccessful End of Training")
            # self.file_object.close()
            raise Exception