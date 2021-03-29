from file_operations.File_operations_Azure import File_Operations
from Training_File_DB_operations.DataTypeValidation_db_insertion import DB_Operation

class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.

    Written By: iNeuron Intelligence
    Version: 1.0
    Revisions: None

    """
    def __init__(self, file_object, logger_object):
        self.training_file = 'inputfile.csv'      #downloaded csv filename
        self.fileFromDB = "training-filefrom-db"  #container in which the csv is downloaded from database
        self.logger_object = logger_object
        self.db_ops = DB_Operation(logger=self.logger_object)
        self.fileoperation = File_Operations(log=self.logger_object)
        self.file_object = file_object


    def get_data(self, Database_name, collection_name):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

         Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        self.logger_object.db_log(self.file_object, "Entered the get_data method of the Data_Getter class")
        try:

            # self.data= pd.read_csv(self.training_file) # reading the data file
            # self.data = self.fileoperation.downloadfiles(self.fileFromDB, self.training_file)
            self.data = self.db_ops.selectdatafromcollection(Database_name=Database_name, collection_=collection_name)
            self.logger_object.db_log(self.file_object,
                                      "Data Load Successful.Exited the get_data method of the Data_Getter class")
            return self.data
        except Exception as e:
            self.logger_object.db_log(self.file_object,
                                      "Exception occured in get_data method of the Data_Getter class. Exception message: "+str(e))
            self.logger_object.db_log(self.file_object,
                                      "Data Load Unsuccessful.Exited the get_data method of the Data_Getter class")
            raise Exception()


