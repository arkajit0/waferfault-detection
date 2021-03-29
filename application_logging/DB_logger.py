import pymongo
from datetime import datetime
import config as cfg


#fetch training schema from database
class Fetch_Schema:
    def __init__(self, schema_collection):
        self.client = pymongo.MongoClient(cfg.Mongodb)
        self.db = self.client["Schema_Instances"]
        # self.col = self.db["Training_schema"]
        self.col = self.db[schema_collection]
        self.data = self.col.find_one()


class Drop_log_db:
    def __init__(self, database_name):
        self.client = pymongo.MongoClient(cfg.Mongodb)
        if database_name in self.client.list_database_names():  # check existence of database
            self.client.drop_database(database_name)

        # print("working")


class DB_Logs:
    def __init__(self, database_name):
        #establishing connection to mongodb server
        self.client = pymongo.MongoClient(cfg.Mongodb)
        self.db = self.client[database_name]          #creating database

    def db_log(self, collection_name, status_message):
        #create collection in db
        collection = self.db[collection_name]
        now = datetime.now()
        message = dict()
        message["date"] = now.strftime("%Y-%m-%d")
        message["current_time"] = now.strftime("%H:%M:%S")
        if type(status_message) == str:
            message["message"] = status_message
        else:
            message.update(status_message)
        insert_message = collection.insert_one(message)



