from file_operations.File_operations_Azure import File_Operations
import os
import pandas as pd


path = "Training_Batch_files/"

instance = File_Operations()
instance.create_container("prediction-batch-files")

onlyfiles = [f for f in os.listdir(path)]

for file in onlyfiles:
    csv = pd.read_csv(path+file)
    instance.uploadfiles(container_name="prediction-batch-files", filename=file, data=csv.to_csv(index=False))
    print("done")

