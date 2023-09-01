import random
import torch
from torch.nn import Module
from torch.optim import Optimizer
from tree import CheckpointTree, Node

class TorchCheckpointer():

    def __init__(
            self,
            trainable,
            model: Module,
            optimizer: Optimizer,
            model_name: str,
            n_branches: int = 5,
        ):

        self.trainable = trainable
        self.model = model
        self.optimizer = optimizer
        self.n_branches = n_branches
        self.model_name = model_name
        self.seed = random.randint(0,10000)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def train(self, model: Module, optimizer: Optimizer, n_epochs: int = 10):
        """ Trainable: train epoch function """

        self.tree = CheckpointTree(tree_height=n_epochs, tree_branches=self.n_branches)
        status = 0 # 0 = keep training, 1 = new branch, -1 = complete

        while (True): 

            if (status == 1): # New branch

                # Load in model and optimizer
                load_path = f"checkpoints/{self.model_name}_{self.tree.old_key}"
                model.load_state_dict(torch.load(load_path+".pt"))
                optimizer.load_state_dict(torch.load(load_path+"_O.pt"))

                #  Change torch seed
                seed = random.randint(0,100000)
                self.seed = seed if self.seed != seed else seed+1
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

            # Train one epoch
            val = self.trainable(model, optimizer)

            # Insert value to tree
            status, key = self.tree.insert(val)

            # save model 
            save_name = f"checkpoints/{self.model_name}_{key}"
            torch.save(self.model.state_dict(), save_name+".pt")
            torch.save(self.optimizer.state_dict(), save_name+"_O.pt")

            if (status == -1): return