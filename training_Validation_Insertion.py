from Training_Raw_data_validation.rawValidation import Raw_Data_validation
# from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from Training_File_DB_operations.DataTypeValidation_db_insertion import DB_Operation
from DataTransform_Training.DataTransformation import dataTransform
# from application_logging import logger
from application_logging.DB_logger import DB_Logs, Drop_log_db

class train_validation:
    def __init__(self, path):
        self.drop_existing_log = Drop_log_db(database_name="Training_Logs")
        self.log_writer = DB_Logs(database_name="Training_Logs")
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()
        self.dBOperation = DB_Operation(logger=self.log_writer)
        self.file_object = "Training_Main_Log"


    def train_validation(self):
        try:
            self.log_writer.db_log(self.file_object, 'Start of Validation on files!!')
            # extracting values from prediction schema
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()
            # getting the regex defined to validate filename
            regex = self.raw_data.manualRegexCreation()
            # validating filename of prediction files
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
            # validating column length in the file
            self.raw_data.validateColumnLength(noofcolumns)
            # validating if any column has all values missing
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.db_log(self.file_object, "Raw Data Validation Complete!!")
            
            self.log_writer.db_log(self.file_object, "Starting Data Transforamtion!!")
            # replacing blanks in the csv file with "Null" values to insert in table
            self.dataTransform.replaceMissingWithNull()

            self.log_writer.db_log(self.file_object, "DataTransformation Completed!!!")

            self.log_writer.db_log(self.file_object, "Creating Training_Database and collection on the basis of given schema!!!")
            # create database with given name, if present open the connection! Create table with columns given in schema
            # collection_db_name = self.dBOperation.create_collection("Training_Files", "Training_data")
            # self.log_writer.training_logs(self.file_object, {"message": "Collection creation Completed!!"})
            self.log_writer.db_log(self.file_object, "Insertion of Data into Collection started!!!!")
            # # insert csv files in the table
            self.dBOperation.insertintocollection_gooddata(Database_name="Training_Files", collection_="Training_data",
                                                           source_database="Data_Collection", source_collection="Good_Data")
            self.log_writer.db_log(self.file_object, "Insertion in collection completed!!!")
            # self.log_writer.db_log(self.file_object, "Deleting Good Data Folder!!!")
            # Delete the good data folder after loading files in table
            # self.raw_data.deleteExistingGoodDataTrainingFolder()
            # self.log_writer.db_log(self.file_object, "Good_Data folder deleted!!!")
            # self.log_writer.db_log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            # Move the bad files to archive folder
            # self.raw_data.moveBadFilesToArchiveBad()
            # self.log_writer.db_log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.db_log(self.file_object, "Validation Operation completed!!")
            # self.log_writer.db_log(self.file_object, "Extracting csv file from collection")
            # export data in table to csvfile
            # self.dBOperation.selectdatafromcollection(Database_name="Training_Files", collection_="Training_data",
            #                                           storing_location="training-filefrom-db", filename="inputfile.csv")
            # self.file_object.close()

        except Exception as e:
            raise e









