import numpy as np


class Operation():
    """
    An Operation is a node in a "Graph". 
    TensorFlow will also use this concept of a Graph.
    This Operation class will be inherited by other 
    classes that actually compute the specific 
    operation, such as adding or matrix multiplication.
    """

    def __init__(self, input_nodes=[]):
        """
        Intialize an Operation
        """
        self.input_nodes = input_nodes  # The list of input nodes
        self.output_nodes = []  # List of nodes consuming this node's output
        for node in input_nodes:
            node.output_nodes.append(self)
        _default_graph.operations.append(self)

    def compute(self):
        """ 
        This is a placeholder function. It will be 
        overwritten by the actual specific operation
        that inherits from this class.
        """
        pass


class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_var, b_var):
        self.inputs = [a_var, b_var]
        return a_var * b_var


class matmul(Operation):
    '''
    matrix multipy
    '''

    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_mat, b_mat):
        self.inputs = [a_mat, b_mat]
        return a_mat.dot(b_mat)


class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])

    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))


class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


class Placeholder():
    """
    A placeholder is a node that needs to be provided a 
    value for computing the output in the Graph.
    """

    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        """
        Sets this Graph instance as the Global Default Graph
        """
        global _default_graph
        _default_graph = self


def traverse_postorder(operation):
    """ 
    PostOrder Traversal of Nodes. Basically makes sure
    computations are done in the correct order (Ax first, 
    then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the 
    basic fundamentals of deep learning.
    """
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(operation)
    return nodes_postorder


class Session:
    def run(self, operation, feed_dict={}):
        """ 
          operation: The operation to compute
          feed_dict: Dictionary mapping placeholders to 
          input values (the data)  
        """
        # Puts nodes in correct order
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:  # Operation
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        # Return the requested node value
        return operation.output


if __name__ == "__main__":
    # z = Ax + b
    g = Graph()
    g.set_as_default()
    A = Variable(10)
    b = Variable(1)
    x = Placeholder()
    y = multiply(A, x)
    z = add(y, b)

    sess = Session()
    result = sess.run(operation=z, feed_dict={x: 10})

    print(result)
