import pymongo
import json
# import config as cfg
Mongodb = "mongodb://wafferfault_detection:3smQ30jYQX38kxAm@arkajitcluster-shard-00-00.vwfky.mongodb.net:27017,arkajitcluster-shard-00-01.vwfky.mongodb.net:27017,arkajitcluster-shard-00-02.vwfky.mongodb.net:27017/myFirstDatabase?ssl=true&replicaSet=atlas-3k3zt1-shard-0&authSource=admin&retryWrites=true&w=majority"


#establishing connection to mongodb server
client = pymongo.MongoClient(Mongodb)
# if "Training_Instances" in client. list_database_names():
#     # print("exists")
#     client.drop_database("Training_Instances")
db = client["Schema_Instances"]
col1 = db["Training_schema"]
col2 = db["Prediction_schema"]

schema1 = json.load(open('C:\\Users\\ARKAJIT\\Inueron Class\\ML Final Projects\\waferFaultDetection\\code - Copy\\WaferFaultDetection_new\\schema_prediction.json',))
schema2 = json.load(open('C:\\Users\\ARKAJIT\\Inueron Class\\ML Final Projects\\waferFaultDetection\\code - Copy\\WaferFaultDetection_new\\schema_training.json',))

col1.insert_one(schema2)
col2.insert_one(schema1)
# print(schema_id.inserted_id)