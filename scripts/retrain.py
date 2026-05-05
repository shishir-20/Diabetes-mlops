import os
import datetime

print(" Retraining pipeline started...")

start_time = datetime.datetime.now()

# Run training
os.system("python model/train_model.py")

end_time = datetime.datetime.now()

print(" Retraining completed!")
print(" Start Time:", start_time)
print(" End Time:", end_time)