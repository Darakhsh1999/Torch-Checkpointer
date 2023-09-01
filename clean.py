import os 
def clean():
    for file in os.listdir("./checkpoints"):
        os.remove("./checkpoints/"+file)
    
if __name__ == "__main__":
    clean()
