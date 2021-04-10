import pandas
from datetime import datetime
from data_preprocessing import preprocessing
from data_ingestion import data_loader
# from application_logging import logger
from application_logging.DB_logger import DB_Logs
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
from Training_File_DB_operations.DataTypeValidation_db_insertion import DB_Operation, Model_saving_loading
from file_operations.File_operations_Azure import File_Operations
from Mail_box import Send_mail


class prediction:

    def __init__(self,path):
        self.file_object = "Prediction_Log"
        self.log_writer = DB_Logs(database_name="Prediction_Logs")
        self.db_operation = DB_Operation(logger=self.log_writer)
        self.fileoperations = File_Operations(log=self.log_writer)
        if path is not None:
            self.pred_data_val = Prediction_Data_validation(path)
        self.mail_log_obj = "MailSendLog"
        self.sendmail = Send_mail(self.mail_log_obj, self.log_writer, 'Prediction status')
        self.model_ops = Model_saving_loading(database_name="Models")


    def predictionFromModel(self):

        try:
            # self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            # self.fileoperations.create_container("prediction-output-file")
            self.log_writer.db_log(self.file_object, "Start of Prediction")
            data_getter= data_loader.Data_Getter(self.file_object, self.log_writer)
            data=data_getter.get_data(Database_name="Prediction_Files", collection_name="Prediction_csv")

            #code change
            # wafer_names=data['Wafer']
            # data=data.drop(labels=['Wafer'],axis=1)

            preprocessor= preprocessing.Preprocessor(self.file_object, self.log_writer)
            is_null_present=preprocessor.is_null_present(data, database_name="Data_Collection_Prediction",
                                                         collection_name="Null_values")
            if(is_null_present):
                data=preprocessor.impute_missing_values(data)

            cols_to_drop=preprocessor.get_columns_with_zero_std_deviation(data)
            data=preprocessor.remove_columns(data,cols_to_drop)
            print(data.shape)
            #data=data.to_numpy()
            # file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans= self.model_ops.load_model_db(collection_name="KMeans", model_name="KMeans")

            ##Code changed
            #pred_data = data.drop(['Wafer'],axis=1)
            clusters=kmeans.predict(data.drop(['Wafer'],axis=1))#drops the first column for cluster prediction
            data['clusters']=clusters
            clusters=data['clusters'].unique()
            for i in clusters:
                cluster_data= data[data['clusters']==i]
                wafer_names = list(cluster_data['Wafer'])
                cluster_data=data.drop(labels=['Wafer'],axis=1)
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                # model_name = file_loader.find_correct_model_file(i)
                model_name = self.model_ops.find_correct_model_file(cluster_number=i)
                # model = file_loader.load_model(model_name)
                model = self.model_ops.load_model_db(collection_name=model_name, model_name=model_name)
                result=list(model.predict(cluster_data))
                result = pandas.DataFrame(list(zip(wafer_names,result)),columns=['Wafer','Prediction'])

                self.db_operation.insert_into_collection(collection_name="Prediction_Result",
                                                         database_name="Prediction_Files", csv_data=result)
                # result.to_csv("Prediction_Output_File/Predictions.csv",header=True,mode='a+') #appends result to prediction file
            # path = "prediction-csv"
            # self.fileoperations.create_container(path)
            predicted_data = self.db_operation.selectdatafromcollection(Database_name="Prediction_Files",
                                                                        collection_="Prediction_Result")
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")
            self.fileoperations.uploadfiles(container_name="Predicted_Result",
                                            filename="Result_"+str(date)+"_"+str(current_time),
                                            data=predicted_data.to_csv(index=False))
            self.log_writer.db_log(self.file_object, "End of Prediction")
            try:
                self.sendmail.send_mail_content("Data_Collection_Prediction", "Bad_Data")
            except:
                pass
        except Exception as ex:
            self.log_writer.db_log(self.file_object, "Error occured while running the prediction!! Error:: %s" % ex)
            raise ex
        return "success"




