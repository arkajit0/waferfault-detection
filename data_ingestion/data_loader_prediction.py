from file_operations.File_operations_Azure import File_Operations
class Data_Getter_Pred:
    """
    This class shall  be used for obtaining the data from the source for prediction.

    Written By: iNeuron Intelligence
    Version: 1.0
    Revisions: None

    """
    def __init__(self, file_object, logger_object, foldername, filename):
        self.folder_name = foldername
        self.prediction_file = filename
        self.file_object = file_object
        self.logger_object = logger_object
        self.fileoperations = File_Operations(log=self.logger_object)

    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

         Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        self.logger_object.db_log(self.file_object,{"message": "Entered the get_data method of the Data_Getter class"})
        try:
            # self.data= pd.read_csv(self.prediction_file) # reading the data file
            self.data = self.fileoperations.downloadfiles(container_name=self.folder_name, filename=self.prediction_file)
            self.logger_object.db_log(self.file_object,{"message":
                                                                     "Data Load Successful.Exited the get_data method of the Data_Getter class"})
            return self.data
        except Exception as e:
            self.logger_object.db_log(self.file_object,{"Error message":
                                                                     "Exception occured in get_data method of the Data_Getter class. Exception message: "+str(e)})
            self.logger_object.db_log(self.file_object,
                                               {"Error message": "Data Load Unsuccessful.Exited the get_data method of the Data_Getter class"})
            raise Exception()


