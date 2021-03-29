# import shutil
from file_operations.File_operations_Azure import File_Operations

class File_Operation:
    """
                This class shall be used to save the model after training
                and load the saved model for prediction.

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.fileoperation = File_Operations(log=self.logger_object)
        self.model_directory = "models"

    def create_model_directory(self):
        self.logger_object.db_log(self.file_object,
                                  message={
                                             "message": "Creating directory to keep training models"})
        try:
            self.fileoperation.create_container(container_name=self.model_directory)
        except Exception as e:
            raise Exception()


    def save_model(self,model,filename):
        """
            Method Name: save_model
            Description: Save the model file to directory
            Outcome: File gets saved
            On Failure: Raise Exception

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None
"""
        self.logger_object.db_log(self.file_object,
                                  message={"message": "Entered the save_model method of the File_Operation class"})

        try:
            # self.fileoperation.create_container(container_name=self.model_directory)
            # path = os.path.join(self.model_directory,filename) #create seperate directory for each cluster
            # if os.path.isdir(path): #remove previously existing models for each clusters
            #     shutil.rmtree(self.model_directory)
            #     os.makedirs(path)
            # else:
            #     os.makedirs(path) #
            # with open(path +'/' + filename+'.sav',
            #           'wb') as f:
            #     pickle.dump(model, f) # save the model to file
            filename = filename+'.sav'
            self.fileoperation.savemodel(container_name=self.model_directory, filename=filename, model=model)
            self.logger_object.db_log(self.file_object,
                                      message={"success Message":
                                                          "Model File "+filename+" saved. Exited the save_model method of the Model_Finder class"})

            return 'success'
        except Exception as e:
            self.logger_object.db_log(self.file_object,
                                      message={"Error message":
                                                          "Exception occured in save_model method of the Model_Finder class. Exception message:  " + str(e)})
            self.logger_object.db_log(self.file_object,
                                      message={"Error while saving":
                                                          'Model File '+filename+' could not be saved. Exited the save_model method of the Model_Finder class'})
            raise Exception()

    def load_model(self,filename):
        """
                    Method Name: load_model
                    Description: load the model file to memory
                    Output: The Model file loaded in memory
                    On Failure: Raise Exception

                    Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None
        """
        self.logger_object.db_log(self.file_object,
                                  message={"load model message": "Entered the load_model method of the File_Operation class"})
        try:
            # with open(self.model_directory + filename + '/' + filename + '.sav',
            #           'rb') as f:
            if '.sav' not in filename:
                filename = filename+'.sav'
            model = self.fileoperation.loadmodel(container_name=self.model_directory, filename=filename)
            self.logger_object.db_log(self.file_object,
                                      message={"model loaded successfully": "Model File " + filename + "loaded. Exited the load_model method of the Model_Finder class"})
            # return pickle.load(f)
            return model
        except Exception as e:
            self.logger_object.db_log(self.file_object,
                                      message={"Model load Error":
                                                "Exception occured in load_model method of the Model_Finder class. Exception message:  " + str(e)})
            self.logger_object.db_log(self.file_object,
                                      message= {"Model Error message":
                                                           "Model File " + filename + " could not be saved. Exited the load_model method of the Model_Finder class"})
            raise Exception()

    def find_correct_model_file(self,cluster_number):
        """
                            Method Name: find_correct_model_file
                            Description: Select the correct model based on cluster number
                            Output: The Model file
                            On Failure: Raise Exception

                            Written By: iNeuron Intelligence
                            Version: 1.0
                            Revisions: None
                """
        self.logger_object.db_log(self.file_object,
                                  message={"correct model select": "Entered the find_correct_model_file method of the File_Operation class"})
        try:
            self.cluster_number= cluster_number
            self.folder_name=self.model_directory
            # self.list_of_model_files = []
            # self.list_of_files = self.fileoperation.getallFiles(container_name=self.folder_name)

            for self.file in self.list_of_files:
                try:
                    if (self.file.index(str( self.cluster_number))!=-1):
                        self.model_name = self.file
                except:
                    continue
            self.model_name=self.model_name
            self.logger_object.db_log(self.file_object,
                                      {"success message": "Exited the find_correct_model_file method of the Model_Finder class."})
            return self.model_name
        except Exception as e:
            self.logger_object.db_log(self.file_object,
                                      {"Model selector Error message":
                                                  "Exception occured in find_correct_model_file method of the Model_Finder class. Exception message:  " + str(e)})
            self.logger_object.db_log(self.file_object,
                                      {"Model selector Error message":
                                                  "Exited the find_correct_model_file method of the Model_Finder class with Failure"})
            raise Exception()