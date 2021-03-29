from azure.storage.blob import BlobServiceClient
import pandas as pd
import time
from io import StringIO
import base64
import config as cfg
import pickle
# from Training_File_DB_operations.DB_logger import Training_Logs



class File_Operations:
    def __init__(self, log):
        self.connect_str = cfg.Azure_key
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
        self.logger = log
        self.collection_name = 'FileoFolderperations'



    def delete_container(self, container_name):
        container_obj = self.blob_service_client.list_containers()
        containers = [container.name for container in container_obj]

        if container_name in containers:
            # print('deleting')
            try:
                self.blob_service_client.delete_container(container_name)
                self.logger.db_log(self.collection_name, "Folder deleted successfully!!!"+ str(container_name))
            except Exception as e:
                # self.logger.db_log(collection_name=self.collection_name,
                #                    message={"Folder deletion error!!!": e})
                print(e)
            time.sleep(35)

            # raise Exception("error occurred in delete_container: No container found")


    def create_container(self, container_name):
        self.delete_container(container_name=container_name)

        try:
            self.blob_service_client.create_container(container_name)
            # self.logger.db_log(collection_name=self.collection_name,
            #                    message={"Folder created successfully!!!": container_name})
        except Exception as e:
            print(e)
            # self.logger.db_log(collection_name=self.collection_name,
            #                    message={"Error creating Folder": e})

    def getallFiles(self, container_name):
        container_obj = self.blob_service_client.list_containers()
        containers = [container.name for container in container_obj]
        if container_name in containers:
            file_object = self.blob_service_client.get_container_client(container_name)
            allfiles = [file.name for file in file_object.list_blobs()]
            return allfiles
        else:
            raise Exception("error occurred in getallFiles: No container found")

    def uploadfiles(self, container_name, filename, data):
        container_obj = self.blob_service_client.list_containers()
        containers = [container.name for container in container_obj]
        onlyfiles = self.getallFiles(container_name)
        if container_name in containers:
            if filename in onlyfiles:
                self.deletefiles(container_name, filename)
            blob_client = self.blob_service_client.get_blob_client(container_name, blob=filename)
            blob_client.upload_blob(data)
        else:
            raise Exception("error in uploadfiles:: no container name")

    def deletefiles(self, container_name, filename):
        container_obj = self.blob_service_client.list_containers()
        containers = [container.name for container in container_obj]
        if container_name in containers:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=filename)
            blob_client.delete_blob()
            self.logger.db_log(self.collection_name, {"File deleted successfull!!!": filename})
        else:
            self.logger.db_log(self.collection_name, {"File deletion error!!!": filename})
            raise Exception("error occurred in deletefiles: No container found")


    def movefiles(self, source, destination, filename):
        collection_name = 'FileTransfer'
        source_obj = self.blob_service_client.get_blob_client(source, blob=filename)
        dest_obj = self.blob_service_client.get_blob_client(destination, blob=filename)
        try:
            dest_obj.start_copy_from_url(source_obj.url)
            self.logger.db_log(collection_name, {"File Transfer successfull!!!": filename})
            self.deletefiles(source, filename)
        except Exception as e:
            self.logger.db_log(collection_name, {"File Transfer unsuccessfull!!!": filename})

    def copyfiles(self, source, destination, filename):
        collection_name = 'FileTransfer'
        source_obj = self.blob_service_client.get_blob_client(source, blob=filename)
        dest_obj = self.blob_service_client.get_blob_client(destination, blob=filename)
        try:
            dest_obj.start_copy_from_url(source_obj.url)
            self.logger.db_log(collection_name, {"File Transfer successfull!!!": filename})
            # self.deletefiles(source, filename)
        except Exception as e:
            self.logger.db_log(collection_name, {"File Transfer unsuccessfull!!!": filename})



    def downloadfiles(self, container_name, filename):
        container_obj = self.blob_service_client.list_containers()
        containers = [container.name for container in container_obj]
        if container_name in containers:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=filename)
            blob_file = blob_client.download_blob()
            file_df = pd.read_csv(StringIO(blob_file.readall().decode()))
            return file_df
        else:
            raise Exception("error occurred in downloadfiles: No container found")

    def savemodel(self, container_name, filename, model):
        collection = "ModelTrainingLog"
        model_file = self.getallFiles(container_name=container_name)
        if filename in model_file:
            self.deletefiles(container_name, filename)
        try:
            data = pickle.dumps(model)
            encoded = base64.b64encode(data)
            blob_client = self.blob_service_client.get_blob_client(container_name, blob=filename)
            blob_client.upload_blob(encoded)
        except Exception as e:
            print(e)
            # self.logger.db_log(collection, message={"Can't save Model": e})


    def loadmodel(self, container_name, filename):
        container_obj = self.blob_service_client.list_containers()
        containers = [container.name for container in container_obj]
        if container_name in containers:
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=filename)
            blob_file = blob_client.download_blob()
            decoded = base64.b64decode(blob_file.readall())
            model = pickle.loads(decoded)
            return model
        else:
            raise Exception("error occurred in loadmodel: No container found")


