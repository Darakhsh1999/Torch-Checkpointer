import numpy as np
import matplotlib.pyplot as plt

class Node():
    """ Node class for the CheckpointTree """

    def __init__(self, val, epoch: int = None, diff: float = None, ID: str = None, branch: int = None):
        self.val = val
        self.epoch = epoch
        self.diff = diff 
        self.ID = ID
        self.branch = branch
        self.left = None
        self.right = None


class CheckpointTree():
    """" Check point tree for training neural network """

    def __init__(self, tree_height: int, tree_branches: int):
        self.tree: dict[int:Node] = {}
        self.tree_height = tree_height
        self.tree_branches = tree_branches
        self.pos = "1" # points to the next node
        self.old_key = None
        self.n_branches = 1
        self.branch_nodes = []
        self.branch_pos = []
        self.complete = False

    def find(self, pos):
        """ Finds node in tree given pos """
        node = self.root
        for c in pos[1:]:
            if c == "1":
                node = node.left
            else:
                node = node.right
        return node

    def insert(self, value: float):
        """ Insert value into the checkpoint tree """

        # Create node
        key = int(self.pos, 2) # binary -> dec
        epoch = len(self.pos) # height
        if self.old_key is None:
            diff = 0.0
        else:
            diff = value - self.tree[self.old_key].val # diff to parent
        node = Node(val=value, epoch=epoch, diff=diff, ID=self.pos, branch=self.n_branches)

        print(f"Branch = {self.n_branches}, epoch = {epoch}, value = {value:.3f}, diff = {diff:.5f}")

        if self.tree: # Add leaf
            self.tree[key] = node 
            if self.pos.endswith("1"):
                self.tree[self.old_key].left = node
            else:
                self.tree[self.old_key].right = node

        else: # Root node
            self.root = node
            self.tree[key] = node

        # Update key and position
        self.old_key = key
        self.pos = self.pos + "1" # update pos

        if (epoch == self.tree_height): # New branch
            self.n_branches += 1
            if (self.n_branches == 1+self.tree_branches): return -1, key # Tree complete
            self.find_branch_point()
            self.branch_nodes.append(self.find(self.branch_point))
            self.branch_pos.append(self.branch_point)
            return 1, key
        return 0, key
    
    
    def find_branch_point(self):
        """ Preorder traversal of binary tree """

        self.max_absolute_delta = 0.0
        self.max_absolute_delta_pos = "1"
        self.traverse(self.root)
        self.branch_point: str = self.max_absolute_delta_pos[:-1]
        self.pos = self.branch_point + "0" # branch right
        self.old_key = int(self.branch_point, 2)
    
    def traverse(self, node: Node):
        """ Recursive traversal """
        
        if (node.epoch == self.tree_height): return # Dont branch from end points

        if (abs(node.diff) > self.max_absolute_delta) and (node.ID[:-1] not in self.branch_pos):
            self.max_absolute_delta = node.diff
            self.max_absolute_delta_pos = node.ID
        
        if node.left: self.traverse(node.left)

        if node.right: self.traverse(node.right)

    def preorder_traversal(self, node=None, root=True):
        
        if root: node = self.root
        node_path = [node]
        if node.left: node_path +=  self.preorder_traversal(node.left, root=False)
        if node.right: node_path += self.preorder_traversal(node.right, root=False)

        return node_path
    
    def print_tree(self):

        tree_nodes = self.preorder_traversal()

        # Plot nodes
        for node in tree_nodes:
            ID = node.ID
            x = node.branch
            y = len(ID)
            plt.scatter(x, y, marker="o", c="blue", s=200, edgecolors="black")
            plt.text(x+0.25, y, s=f"{node.val:.2f}", verticalalignment="center", horizontalalignment="center", fontsize=14)

        # Draw lines
        branch_idx = 0
        for node_idx, node1 in enumerate(tree_nodes):

            if (node_idx == len(tree_nodes)-1): continue # Final node

            if (node1.epoch == self.tree_height): # End of branch
                branch_node: Node = self.branch_nodes[branch_idx]
                target_pos = branch_node.ID + "0"
                target_node: Node = self.find(target_pos)
                x1 = branch_node.branch
                x2 = target_node.branch

                y1 = len(branch_node.ID)
                y2 = target_node.epoch
                plt.plot([x1,x2],[y1,y2], "k-", linewidth=2)
                branch_idx += 1
                continue

            x = node1.branch
            y1 = node1.epoch
            y2 = tree_nodes[node_idx+1].epoch

            plt.plot([x,x],[y1,y2], "k-", linewidth=3)

        
        plt.xticks(np.arange(1,self.tree_branches+1))
        plt.yticks(np.arange(1,self.tree_height+1))
        plt.xlabel("branch")
        plt.ylabel("epoch")
        plt.grid()
        plt.show()
    
    def print_3d_tree(self):

        tree_nodes = self.preorder_traversal()

        ax = plt.figure().add_subplot(projection='3d')

        # Plot nodes
        for node in tree_nodes:
            ID = node.ID
            x = node.branch
            y = len(ID)
            z = node.val
            ax.scatter3D(x, y, z, marker="o", c="blue", s=200, edgecolors="black")

        # Draw lines
        branch_idx = 0
        for node_idx, node1 in enumerate(tree_nodes):

            if (node_idx == len(tree_nodes)-1): continue # Final node

            if (node1.epoch == self.tree_height): # End of branch
                branch_node: Node = self.branch_nodes[branch_idx]
                target_pos = branch_node.ID + "0"
                target_node: Node = self.find(target_pos)
                x1 = branch_node.branch
                x2 = target_node.branch
                y1 = len(branch_node.ID)
                y2 = target_node.epoch
                z1 = branch_node.val
                z2 = target_node.val
                ax.plot3D([x1,x2],[y1,y2],[z1,z2], "k-", linewidth=2)
                branch_idx += 1
                continue

            x = node1.branch
            y1 = node1.epoch
            y2 = tree_nodes[node_idx+1].epoch
            z1 = node1.val
            z2 = tree_nodes[node_idx+1].val

            ax.plot3D([x,x],[y1,y2],[z1,z2], "k-", linewidth=2)

        plt.xticks(np.arange(1,self.tree_branches+1))
        plt.yticks(np.arange(1,self.tree_height+1))
        ax.set_xlabel("branch")
        ax.set_ylabel("epoch")
        ax.set_zlabel("accuracy")
        plt.show()
    



if __name__ == "__main__":

   tree = CheckpointTree(tree_height=4, tree_branches=2)

   n1 = 0.1
   n2 = 0.3
   n3 = 0.6
   n4 = 0.65
   n5 = 0.4
   n6 = 0.7

   tree.insert(n1) 
   tree.insert(n2)
   tree.insert(n3)
   tree.insert(n4)
   tree.insert(n5)
   tree.insert(n6)

   tree.print_tree()

   """    
                n1
            n2
        n3
    n4
   
   """
