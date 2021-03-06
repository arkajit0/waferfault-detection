import pymongo
# from Training_File_DB_operations.DB_logger import Training_Logs
import pandas as pd
from file_operations.File_operations_Azure import File_Operations
import config as cfg
import pickle
from datetime import datetime


class DB_Operation:
    """
          This class shall be used for handling all the NOSQL operations.

          Written By: iNeuron Intelligence
          Version: 1.0
          Revisions: None

          """
    def __init__(self, logger, good_raw_folder=None, bad_raw_folder=None):
        # self.path = 'Training_Database/'
        self.client = pymongo.MongoClient(cfg.Mongodb)
        self.badFilePath = bad_raw_folder
        self.goodFilePath = good_raw_folder
        self.logger = logger
        # self.fileoperation = File_Operations(log=self.logger)


    def selectdatafromcollection(self, Database_name, collection_):
        """
                    Method Name: selectdatafromcollection
                    Description: Selecting data from collection as dataframe
                    Outcome: dataframe


        """

        # self.fileFromDB = storing_location
        # self.fileName = filename
        collection_name = "ExportToCsv"
        try:
            conn = self.client[Database_name]
            collection_db = conn[collection_]
            data_fromdb_df = pd.DataFrame(list(collection_db.find()))

            self.logger.db_log(collection_name, "File exported successfully!!!")
            return data_fromdb_df.drop(['_id'], axis=1)
        except Exception as e:
            self.logger.db_log(collection_name, "File exporting error!!!")


    def insert_into_collection(self, database_name, collection_name, csv_data):
        """
                            Method Name: insert_into_collection
                            Description: inserting chunk data into collection

                """
        conn = self.client[database_name]
        db_collection = conn[collection_name]
        collection_name = "ExportToCsv"
        try:
            records = csv_data.to_dict('records')
            db_collection.insert_many(records)
            self.logger.db_log(collection_name, {"message": "prediction result inserted"})
        except Exception as e:
            self.logger.db_log(collection_name, {"message": "prediction insertion error!!!"+str(e)})


    def insert_intermediate_data_intodb(self, database_name, collection_name, file_name, csv_data):
        """
                                    Method Name: insert_intermediate_data_intodb
                                    Description: inserting data from file storage to database


                        """
        conn = self.client[database_name]
        db_collection = conn[collection_name]
        collection_name = "Good_Bad_Data"
        now = datetime.now()
        try:
            records = csv_data.to_dict('records')
            data_insert = {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"), "filename": file_name,
                           "csv_data": records}
            if db_collection.count_documents({"filename": file_name})>0:
                db_collection.delete_one({"filename": file_name})
            db_collection.insert_one(data_insert)
            # self.logger.db_log(collection_name=collection_name,
            #                    message={"message": "prediction result inserted"})
        except Exception as e:

            self.logger.db_log(collection_name, "prediction insertion error!!!"+str(e))


    def get_files(self, database_name, collection_name):
        """
                                            Method Name: get_files
                                            Description: fetch all file names from a particular collection


                                """
        conn = self.client[database_name]
        db_collection = conn[collection_name]
        file_cursor = db_collection.find()
        getallfiles = [file_name["filename"] for file_name in file_cursor]
        collection_name = "Good_Bad_Data"
        return getallfiles


    def download_csv_file(self, database_name, collection_name, filename):
        """
                                                    Method Name: download_csv_file
                                                    Description: downloading csv files from a particular collection


                                        """
        conn = self.client[database_name]
        db_collection = conn[collection_name]
        csv_file = pd.DataFrame([file["csv_data"] for file in db_collection.find({"filename": filename})][0])
        return csv_file


    def move_csv(self, database_name, source_collection, destination_collection, filename, csv_data):
        """
                            Method Name: move_csv
                            Description: move files from one to another collection


                                                """
        conn = self.client[database_name]
        sourcecollection = conn[source_collection]
        destinationcollection = conn[destination_collection]
        if destinationcollection.count_documents({"filename": filename}) > 0:
            destinationcollection.delete_one({"filename": filename})
        self.insert_intermediate_data_intodb(database_name=database_name, collection_name=destination_collection,
                                             file_name=filename, csv_data=csv_data)
        sourcecollection.delete_one({"filename": filename})


    def insert_null_values(self, database_name, collection_name, csv_data):
        """
                                    Method Name: insert_null_values
                                    Description: inserting null values into a collection


                                                        """
        conn = self.client[database_name]
        collection = conn[collection_name]
        records = csv_data.to_dict('records')
        collection.insert_many(records)


    def insertintocollection_gooddata(self, Database_name, collection_, source_database, source_collection):
        """
                                            Method Name: insertintocollection_gooddata
                                            Description: inserting all data from good_data collection to another collection


                                                                """
        if Database_name in self.client.list_database_names():
            self.client.drop_database(Database_name)
        conn = self.client[Database_name]
        db_collection = conn[collection_]
        onlyfiles = self.get_files(database_name=source_database, collection_name=source_collection)
        # log_file = open("Training_Logs/DbInsertLog.txt", 'a+')
        collection_name = "DbInsertLog"
        # print(onlyfiles)
        for file in onlyfiles:
            csv_data = self.download_csv_file(database_name=source_database, collection_name=source_collection,
                                              filename=file)
            if "Good/Bad" in csv_data.columns:
                csv_data.rename(columns={"Good/Bad": "Output"}, inplace=True)
            records = csv_data.to_dict('records')
            db_collection.insert_many(records)
            self.logger.db_log(collection_name, "File loaded successfully!!!"+ str(file))


    def save_load_pickle_file(self, filename, database_name, collection_name, status, pickle_file=None):
        """
                            Method Name: save_load_pickle_file
                            Description: save and load pickle files


                                                """
        conn = self.client[database_name]
        collection = conn[collection_name]
        if status == "save":
            pickled_model = pickle.dumps(pickle_file)
            collection.insert_one({"filename": filename, "pickle_file": pickled_model})
        elif status == "load":
            data = collection.find({"filename": filename})

            for datas in data:
                json_data = datas

            pickled_model = json_data["pickle_file"]

            return pickle.loads(pickled_model)
        else:
            return "give load or save"


class Model_saving_loading:
    """
              This class shall be used for handling saving models  after training in Nosql and loading model from NOSQL

              Written By: iNeuron Intelligence
              Version: 1.0
              Revisions: None

              """


    def __init__(self, database_name):
        self.client = pymongo.MongoClient(cfg.Mongodb)
        self.database_name = database_name
        self.all_collections = self.client[self.database_name].collection_names()


    def save_model_db(self, collection_name, model, model_name, model_score=None):
        """
                    Method Name: save_model_db
                    Description: saving the model in database collection as binary


                                        """
        conn = self.client[self.database_name]
        collection = conn[collection_name]

        pickled_model = pickle.dumps(model)
        if collection.count_documents({"name": model_name}) > 0:
            collection.delete_one({"name": model_name})
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        if model_score == None:
            detail = {model_name: pickled_model, "name": model_name, "date": date,
                      "current_time": current_time}
        else:
            detail = {model_name: pickled_model, "name": model_name, "score": model_score, "date": date,
                      "current_time": current_time}

        collection.insert_one(detail)


    def load_model_db(self, collection_name, model_name):
        """
                            Method Name: load_model_db
                            Description: loading the model in database collection as binary


                                                """
        conn = self.client[self.database_name]
        collection = conn[collection_name]
        data = collection.find({"name": model_name})

        for datas in data:
            json_data = datas

        pickled_model = json_data[model_name]

        return pickle.loads(pickled_model)


    def find_correct_model_file(self, cluster_number):
        """
                                    Method Name: find_correct_model_file
                                    Description: finding the correct model according to cluster


                                                        """
        self.cluster_number = cluster_number
        for file in self.all_collections:
            try:
                if file.index(str(self.cluster_number))!=-1:
                    self.model_name = file
            except:
                continue

        self.model_name=self.model_name
        return self.model_name







