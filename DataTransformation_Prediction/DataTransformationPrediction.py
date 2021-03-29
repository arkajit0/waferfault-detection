# from application_logging.logger import App_Logger
import numpy as np
from application_logging.DB_logger import DB_Logs
from Training_File_DB_operations.DataTypeValidation_db_insertion import DB_Operation

class dataTransformPredict:

     """
                  This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.

                  Written By: iNeuron Intelligence
                  Version: 1.0
                  Revisions: None

                  """

     def __init__(self):
          self.logger = DB_Logs(database_name="Prediction_Logs")
          self.db_ops = DB_Operation(logger=self.logger)

     def replaceMissingWithNull(self):

          """
                                  Method Name: replaceMissingWithNull
                                  Description: This method replaces the missing values in columns with "NULL" to
                                               store in the table. We are using substring in the first column to
                                               keep only "Integer" data for ease up the loading.
                                               This column is anyways going to be removed during prediction.

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                                          """
          collection_name = "dataTransformLog"
          try:

               onlyfiles = self.db_ops.get_files(database_name="Data_Collection_Prediction", collection_name="Good_Data")
               for file in onlyfiles:
                    # csv = pandas.read_csv(self.goodDataPath+"/" + file)
                    # csv = self.fileoperations.downloadfiles(container_name=self.goodDataPath, filename=file)
                    csv = self.db_ops.download_csv_file(database_name="Data_Collection_Prediction", collection_name="Good_Data",
                                                        filename=file)
                    csv.fillna(np.nan,inplace=True)
                    # #csv.update("'"+ csv['Wafer'] +"'")
                    # csv.update(csv['Wafer'].astype(str))
                    csv['Wafer'] = csv['Wafer'].str[6:]
                    # csv.to_csv(self.goodDataPath+ "/" + file, index=None, header=True)
                    self.db_ops.insert_intermediate_data_intodb(database_name="Data_Collection_Prediction",
                                                                collection_name="Good_Data", file_name=file,
                                                                csv_data=csv)
                    self.logger.db_log(collection_name, "File Transformed successfully!!"+ str(file))
               #log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")

          except Exception as e:
               # self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
               # #log_file.write("Current Date :: %s" %date +"\t" +"Current time:: %s" % current_time + "\t \t" + "Data Transformation failed because:: %s" % e + "\n")
               # log_file.close()
               self.logger.db_log(collection_name,"Data Transformation failed because"+str(e))
               raise e
          # log_file.close()
