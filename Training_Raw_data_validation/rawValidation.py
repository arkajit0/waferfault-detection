import re
from application_logging.DB_logger import *
from Training_File_DB_operations.DataTypeValidation_db_insertion import *
# from application_logging.logger import App_Logger





class Raw_Data_validation:

    """
             This class shall be used for handling all the validation done on the Raw Training Data!!.

             Written By: iNeuron Intelligence
             Version: 1.0
             Revisions: None

             """

    def __init__(self,path):
        self.Batch_Directory = path
        self.schema_path = Fetch_Schema("Training_schema").data
        # self.logger = App_Logger()
        self.logger = DB_Logs(database_name="Training_Logs")
        self.db_ops = DB_Operation(logger=self.logger)
        self.fileoperation = File_Operations(log=self.logger)

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
            dic = self.schema_path
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            collection_name = "valuesfromSchemaValidationLog"
            message ={"LengthOfDateStampInFile": LengthOfDateStampInFile, "LengthOfTimeStampInFile": LengthOfTimeStampInFile,
                      "NumberofColumns": NumberofColumns}
            # self.logger.log(collection_name, message)
            self.logger.db_log(collection_name=collection_name, status_message=message)
            # file.close()



        except ValueError:
            # file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            collection_name = "valuesfromSchemaValidationLog"
            self.logger.db_log(collection_name,
                               {"ValueError": "Value not found inside schema_training.json"})
            # file.close()
            raise ValueError

        except KeyError:
            # file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            # self.logger.log(file, "KeyError:Key value error incorrect key passed")
            collection_name = "valuesfromSchemaValidationLog"
            self.logger.db_log(collection_name,
                               {"KeyError": "Key value error incorrect key passed"})
            # file.close()
            raise KeyError

        except Exception as e:
            # file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            # self.logger.log(file, str(e))
            # file.close()
            collection_name = "valuesfromSchemaValidationLog"
            self.logger.db_log(collection_name,
                               {"Error_message": e})
            raise e

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns


    def manualRegexCreation(self):
        """
                                Method Name: manualRegexCreation
                                Description: This method contains a manually defined regex based on the "FileName" given in "Schema" file.
                                            This Regex is used to validate the filename of the training data.
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
                    Description: This function validates the name of the training csv files as per given name in the schema!
                                 Regex pattern is used to do the validation.If name format do not match the file is moved
                                 to Bad Raw Data folder else in Good raw data.
                    Output: None
                    On Failure: Exception

                     Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None

                """

        #pattern = "['Wafer']+['\_'']+[\d_]+[\d]+\.csv"
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        # self.deleteExistingBadDataTrainingFolder()
        # self.deleteExistingGoodDataTrainingFolder()
        # #create new directories
        # self.createDirectoryForGoodBadRawData()
        # onlyfiles = [f for f in listdir(self.Batch_Directory)]
        onlyfiles = self.fileoperation.getallFiles(self.Batch_Directory)
        try:
            # f = open("Training_Logs/nameValidationLog.txt", 'a+')
            collection_name = "nameValidationLog"
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            # shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Good_Raw")
                            # self.fileoperation.copyfiles(source=self.Batch_Directory, destination="good-raw",
                            #                              filename=filename)
                            self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection",
                                                                        collection_name="Good_Data", file_name=filename,
                                                                        csv_data=self.fileoperation.downloadfiles(
                                                                            container_name=self.Batch_Directory,
                                                                            filename=filename))
                            # self.logger.log(f,"Valid File name!! File moved to GoodRaw Folder :: %s" % filename)
                            message = {"message": "Valid File name!! File moved to GoodRaw Folder "+ str(filename),
                                       "filename": str(filename), "status": "success"}
                            self.logger.db_log(collection_name, message)
                        else:
                            # shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                            # self.fileoperation.copyfiles(source=self.Batch_Directory, destination="bad-raw",
                            #                              filename=filename)
                            self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection",
                                                                        collection_name="Bad_Data", file_name=filename,
                                                                        csv_data=self.fileoperation.downloadfiles(
                                                                            container_name=self.Batch_Directory,
                                                                            filename=filename))
                            # self.logger.log(f,"Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                            message = {"message": "Invalid File Name!! File moved to Bad Raw Folder " + str(filename),
                                       "filename": str(filename), "status": "failure"}
                            self.logger.db_log(collection_name, message)
                    else:
                        # shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                        # self.fileoperation.copyfiles(source=self.Batch_Directory, destination="bad-raw",
                        #                              filename=filename)
                        self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection",
                                                                    collection_name="Bad_Data", file_name=filename,
                                                                    csv_data=self.fileoperation.downloadfiles(
                                                                        container_name=self.Batch_Directory,
                                                                        filename=filename))
                        # self.logger.log(f,"Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                        message = {"message": "Invalid File Name!! File moved to Bad Raw Folder " + str(filename),
                                   "filename": str(filename), "status": "failure"}
                        self.logger.db_log(collection_name, message)
                else:
                    # shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                    # self.fileoperation.copyfiles(source=self.Batch_Directory, destination="bad-raw",
                    #                              filename=filename)
                    self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection",
                                                                collection_name="Bad_Data", file_name=filename,
                                                                csv_data=self.fileoperation.downloadfiles(
                                                                    container_name=self.Batch_Directory,
                                                                    filename=filename))
                    # self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                    message = {"message": "Invalid File Name!! File moved to Bad Raw Folder " + str(filename),
                               "filename": str(filename), "status": "failure"}
                    self.logger.db_log(collection_name, message)

            # f.close()

        except Exception as e:
            # f = open("Training_Logs/nameValidationLog.txt", 'a+')
            # self.logger.log(f, "Error occured while validating FileName %s" % e)
            # f.close()
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
            # f = open("Training_Logs/columnValidationLog.txt", 'a+')
            collection_name = "columnValidationLog"
            # self.logger.log(f,"Column Length Validation Started!!")
            self.logger.db_log(collection_name, "Column Length Validation Started!!")
            # good_files = self.fileoperation.getallFiles("good-raw")
            good_files = self.db_ops.get_files(database_name="Data_Collection", collection_name="Good_Data")
            # for file in listdir('Training_Raw_files_validated/Good_Raw/'):
            for file in good_files:
                # csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                # csv = self.fileoperation.downloadfiles(container_name="good-raw", filename=file)
                csv = self.db_ops.download_csv_file(database_name="Data_Collection", collection_name="Good_Data",
                                                    filename=file)
                # print(csv.columns)
                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    # shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                    # self.fileoperation.movefiles(source="good-raw", destination="bad-raw", filename=file)
                    self.db_ops.move_csv(database_name="Data_Collection", source_collection="Good_Data",
                                         destination_collection="Bad_Data", filename=file, csv_data=csv)
                    # self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                    message = {"message": "Invalid Column Length for the file!! File moved to Bad Raw Folder " + str(file),
                               "filename": str(file), "status": "failure"}
                    self.logger.db_log(collection_name, message)

            # self.logger.log(f, "Column Length Validation Completed!!")
            self.logger.db_log(collection_name, "Column Length Validation Completed!!")
        except OSError:
            # f = open("Training_Logs/columnValidationLog.txt", 'a+')
            # self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            # f.close()
            collection_name = "columnValidationLog"
            self.logger.db_log(collection_name,  "Error Occured while moving the file"+ str(OSError))
            raise OSError
        except Exception as e:
            # f = open("Training_Logs/columnValidationLog.txt", 'a+')
            # self.logger.log(f, "Error Occured:: %s" % e)
            # f.close()
            collection_name = "columnValidationLog"
            self.logger.db_log(collection_name, "Error Occured"+ str(e))
            raise e
        # f.close()

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
            # f = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            # self.logger.log(f,"Missing Values Validation Started!!")
            collection_name = "missingValuesInColumn"
            self.logger.db_log(collection_name, "Missing Values Validation Started!!")
            # good_files = self.fileoperation.getallFiles("good-raw")
            good_files = self.db_ops.get_files(database_name="Data_Collection", collection_name="Good_Data")
            # for file in listdir('Training_Raw_files_validated/Good_Raw/'):
            for file in good_files:
                # csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                # csv = self.fileoperation.downloadfiles(container_name="good-raw", filename=file)
                csv = self.db_ops.download_csv_file(database_name="Data_Collection", collection_name="Good_Data",
                                                    filename=file)
                count = 0
                #print(csv.columns)
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count+=1
                        # shutil.move("Training_Raw_files_validated/Good_Raw/" + file,
                        #             "Training_Raw_files_validated/Bad_Raw")
                        # self.fileoperation.movefiles(source="good-raw", destination="bad-raw", filename=file)
                        self.db_ops.move_csv(database_name="Data_Collection", source_collection="Good_Data",
                                             destination_collection="Bad_Data", filename=file, csv_data=csv)
                        # self.logger.log(f,"Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        message = {
                            "message": "Invalid Column Length for the file!! File moved to Bad Raw Folder " + str(file),
                            "filename": str(file), "status": "failure"}
                        self.logger.db_log(collection_name, message)

                        break
                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    # self.fileoperation.uploadfiles(container_name="good-raw", filename=file, data=csv.to_csv(index=False))
                    self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection",
                                                                collection_name="Good_Data", file_name=file,
                                                                csv_data=csv)

                    # csv.to_csv("Training_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
        except OSError:
            # f = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            # self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            # f.close()
            collection_name = "missingValuesInColumn"
            self.logger.db_log(collection_name, "Error Occured while moving the file"+ str(OSError))
            raise OSError
        except Exception as e:
            # f = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            # self.logger.log(f, "Error Occured:: %s" % e)
            # f.close()
            collection_name = "missingValuesInColumn"
            self.logger.db_log(collection_name, "Error Occured"+ str(e))
            raise e
        # f.close()












