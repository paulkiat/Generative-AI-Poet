import random
from math import sqrt, log

class MCTS():
    
    def __init__(self, model, n_sims=500):
        self.model = model
        self.n_sims = n_sims
    
    def run_mcts(self, state):
        for i in range(self.n_sims):
            leaf = self.tree_search(state)
            reward = self.evaluate_leaf(leaf)
            self.backpropogate(leaf, reward)
            
    def tree_search(self, state):
        current_node = self.root
        while not current_node.is_leaf():
            current_node = self.select_best_child(current_node)
        if current_node.is_fully_expanded():
            return current_node
        else:
            return self.expand_node(current_node)
        
    def select_best_child(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            score = child.reward + sqrt(log(node.n_visits) / child.n_visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def expand_node(self, node):
        """
        Generate new chid nodes for current node by trying all possible actions from current state and adding to tree
        """
        for action in self.get_possible_actions(node.state):
            new_state = self.apply_action(node.state, action)
            new_node = Node(new_state, parent=node)
            node.children.append(new_node)
        return new_node
            
    def evaluate_leaf(self, leaf):
        """
        Runs model on input data for leaf node and returns accuracy
        """
        accuracy = self.model.evaluate(leaf.state)
        return accuracy
    
    def backpropagate(self, leaf, reward):
        """
        Updates rewards and numer of visits for nodes in path from leaf -> root
        """
        current_node = leaf
        while current_node is not None:
            current_node.n_visits += 1
            current_node.reward += reward
            current_node = current_node.parent

class HyperparameterTuner():
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid
        
    def tune_params(self):
        mcts = MCTS(self.model)
        current_state = self.get_random_state()
        mcts.run_mcts(current_state)
        best_leaf = self.get_best_leaf(mcts.root)
        best_params = self.get_params_from_leaf(best_leaf)
        return best_params
    
    def get_random_state(self):
        state = {}
        for key, values in self.param_grid.items():
            state[key] = random.choice(values)
        return state
    
    def get_best_leaf(self, root):
        best_score = -float('inf')
        best_leaf = None
        for leaf in root.leaves():
            if leaf.reward > best_score:
                best_score = leaf.reward
                best_leaf = leaf
        return best_leaf
    
    def get_params_from_leaf(self, leaf):
        return leaf.state
    
param_grid = {'learning_rate': [0.01, 0.05, 0.01],
                    'batch_size': [32, 64, 128],
                    'num_epochs': [10, 20, 30]}

tuner = HyperparameterTuner(nm_model, param_grid)
best_params = tuner.tune_params()
        