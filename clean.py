import os 
for file in os.listdir("./checkpoints"):
    os.remove("./checkpoints/"+file)
