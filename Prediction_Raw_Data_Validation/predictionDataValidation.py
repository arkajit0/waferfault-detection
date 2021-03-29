# from os import listdir
import re
# import shutil
# from application_logging.logger import App_Logger
from application_logging.DB_logger import *
from Training_File_DB_operations.DataTypeValidation_db_insertion import *
from file_operations.File_operations_Azure import File_Operations




class Prediction_Data_validation:
    """
               This class shall be used for handling all the validation done on the Raw Prediction Data!!.

               Written By: iNeuron Intelligence
               Version: 1.0
               Revisions: None

               """

    def __init__(self,path):
        self.Batch_Directory = path
        self.schema_path = Fetch_Schema(schema_collection="Prediction_schema").data
        self.logger = DB_Logs(database_name="Prediction_Logs")
        self.db_ops = DB_Operation(logger=self.logger)
        self.fileoperations = File_Operations(log=self.logger)


    def valuesFromSchema(self):
        """
                                Method Name: valuesFromSchema
                                Description: This method extracts all the relevant information from the pre-defined "Schema" file.
                                Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
                                On Failure: Raise ValueError,KeyError,Exception

                                 Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                                        """
        try:
            # with open(self.schema_path, 'r') as f:
            #     dic = json.load(f)
            #     f.close()
            dic = self.schema_path
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            file = "valuesfromSchemaValidationLog"
            message ={"LengthOfDateStampInFile": LengthOfDateStampInFile, "LengthOfTimeStampInFile": LengthOfTimeStampInFile,
                      "NumberofColumns": NumberofColumns}
            self.logger.db_log(file, {"message": message})

            # file.close()



        except ValueError:
            file = "valuesfromSchemaValidationLog"
            self.logger.db_log(file, "Value not found inside schema_training.json")
            # file.close()
            raise ValueError

        except KeyError:
            file = "valuesfromSchemaValidationLog"
            self.logger.db_log(file, "Key value error incorrect key passed")
            # file.close()
            raise KeyError

        except Exception as e:
            file = "valuesfromSchemaValidationLog"
            self.logger.db_log(file, "valuesFromSchema error message"+ str(e))
            # file.close()
            raise e

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns


    def manualRegexCreation(self):

        """
                                      Method Name: manualRegexCreation
                                      Description: This method contains a manually defined regex based on the "FileName" given in "Schema" file.
                                                  This Regex is used to validate the filename of the prediction data.
                                      Output: Regex pattern
                                      On Failure: None

                                       Written By: iNeuron Intelligence
                                      Version: 1.0
                                      Revisions: None

                                              """
        regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        return regex





    def validationFileNameRaw(self,regex,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        """
            Method Name: validationFileNameRaw
            Description: This function validates the name of the prediction csv file as per given name in the schema!
                         Regex pattern is used to do the validation.If name format do not match the file is moved
                         to Bad Raw Data folder else in Good raw data.
            Output: None
            On Failure: Exception

             Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

        """
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        onlyfiles = self.fileoperations.getallFiles(container_name=self.Batch_Directory)

        try:
            f = "nameValidationLog"
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:


                            self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection_Prediction",
                                                                        collection_name="Good_Data", file_name=filename,
                                                                        csv_data=self.fileoperations.downloadfiles(
                                                                            container_name=self.Batch_Directory,
                                                                            filename=filename))
                            message = {"message": "Valid File name!! File moved to GoodRaw Folder " + str(filename),
                                       "filename": str(filename), "status": "success"}
                            self.logger.db_log(f, message)

                        else:
                            # shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")

                            self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection_Prediction",
                                                                        collection_name="Bad_Data", file_name=filename,
                                                                        csv_data=self.fileoperations.downloadfiles(
                                                                            container_name=self.Batch_Directory,
                                                                            filename=filename))
                            message = {"message": "Invalid File Name!! File moved to Bad Raw Folder " + str(filename),
                                       "filename": str(filename), "status": "failure"}
                            self.logger.db_log(f, message)
                    else:
                        # shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")

                        self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection_Prediction",
                                                                    collection_name="Bad_Data", file_name=filename,
                                                                    csv_data=self.fileoperations.downloadfiles(
                                                                        container_name=self.Batch_Directory,
                                                                        filename=filename))
                        message = {"message": "Invalid File Name!! File moved to Bad Raw Folder " + str(filename),
                                   "filename": str(filename), "status": "failure"}
                        self.logger.db_log(f, message)
                else:
                    # shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")

                    self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection_Prediction",
                                                                collection_name="Bad_Data", file_name=filename,
                                                                csv_data=self.fileoperations.downloadfiles(
                                                                    container_name=self.Batch_Directory,
                                                                    filename=filename))
                    message = {"message": "Invalid File Name!! File moved to Bad Raw Folder " + str(filename),
                               "filename": str(filename), "status": "failure"}
                    self.logger.db_log(f, message)

            # f.close()

        except Exception as e:
            collection_name = "nameValidationLog"
            self.logger.db_log(collection_name, "Error occured while validating FileName"+ str(e))
            raise e




    def validateColumnLength(self,NumberofColumns):
        """
                    Method Name: validateColumnLength
                    Description: This function validates the number of columns in the csv files.
                                 It is should be same as given in the schema file.
                                 If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                                 If the column number matches, file is kept in Good Raw Data for processing.
                                The csv file is missing the first column name, this function changes the missing name to "Wafer".
                    Output: None
                    On Failure: Exception

                     Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None

             """
        try:
            collection_name = "columnValidationLog"
            # self.logger.log(f,"Column Length Validation Started!!")
            self.logger.db_log(collection_name, "Column Length Validation Started!!")
            # good_files = self.fileoperations.getallFiles(container_name="prediction-good-raw")
            good_files = self.db_ops.get_files(database_name="Data_Collection_Prediction", collection_name="Good_Data")
            for file in good_files:
                # csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                # csv = self.fileoperations.downloadfiles(container_name="prediction-good-raw", filename=file)
                csv = self.db_ops.download_csv_file(database_name="Data_Collection_Prediction", collection_name="Good_Data",
                                                    filename=file)
                if csv.shape[1] == NumberofColumns:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    # csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
                    self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection_Prediction",
                                                                collection_name="Good_Data", file_name=file,
                                                                csv_data=csv)
                else:
                    # shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file, "Prediction_Raw_Files_Validated/Bad_Raw")
                    # self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                    self.db_ops.move_csv(database_name="Data_Collection_Prediction", source_collection="Good_Data",
                                         destination_collection="Bad_Data", filename=file, csv_data=csv)
                    # self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                    message = {
                        "message": "Invalid Column Length for the file!! File moved to Bad Raw Folder " + str(file),
                        "filename": str(file), "status": "failure"}
                    self.logger.db_log(collection_name, message)

            self.logger.db_log(collection_name, "Column Length Validation Completed!!")
        except OSError:
            collection_name = "columnValidationLog"
            self.logger.db_log(collection_name,  "Error Occured while moving the file"+ str(OSError))
            raise OSError
        except Exception as e:
            collection_name = "columnValidationLog"
            self.logger.db_log(collection_name, "Error Occured"+ str(e))
            raise e

        # f.close()

    # def deletePredictionFile(self):
    #
    #     # if os.path.exists('Prediction_Output_File/Predictions.csv'):
    #     #     os.remove('Prediction_Output_File/Predictions.csv')
    #     # self.fileoperations.del

    def validateMissingValuesInWholeColumn(self):
        """
                                  Method Name: validateMissingValuesInWholeColumn
                                  Description: This function validates if any column in the csv file has all values missing.
                                               If all the values are missing, the file is not suitable for processing.
                                               SUch files are moved to bad raw data.
                                  Output: None
                                  On Failure: Exception

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                              """
        try:
            collection_name = "missingValuesInColumn"
            self.logger.db_log(collection_name, "Missing Values Validation Started!!")
            # good_files = self.fileoperations.getallFiles("prediction-good-raw")
            good_files = self.db_ops.get_files(database_name="Data_Collection_Prediction", collection_name="Good_Data")
            for file in good_files:
                # csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                # csv = self.fileoperations.downloadfiles(container_name="prediction-good-raw", filename=file)
                csv = self.db_ops.download_csv_file(database_name="Data_Collection_Prediction",
                                                    collection_name="Good_Data",
                                                    filename=file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count+=1
                        self.db_ops.move_csv(database_name="Data_Collection_Prediction", source_collection="Good_Data",
                                             destination_collection="Bad_Data", filename=file, csv_data=csv)
                        # self.logger.log(f,"Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        message = {
                            "message": "Invalid Column Length for the file!! File moved to Bad Raw Folder " + str(file),
                            "filename": str(file), "status": "failure"}
                        self.logger.db_log(collection_name, message)
                        break
                if count==0:
                    try:
                        csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)

                    except:
                        self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection_Prediction",
                                                                    collection_name="Good_Data", file_name=file,
                                                                    csv_data=csv)
        except OSError:
            collection_name = "missingValuesInColumn"
            self.logger.db_log(collection_name, "Error Occured while moving the file"+ str(OSError))
            raise OSError
        except Exception as e:
            collection_name = "missingValuesInColumn"
            self.logger.db_log(collection_name, "Error Occured"+ str(e))
            raise e
        # f.close()













