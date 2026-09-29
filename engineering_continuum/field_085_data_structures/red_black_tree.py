"""Course 085: Self-Balancing Red-Black Binary Search Tree"""
class RBNode:
    def __init__(self, val, color="RED"):
        self.val = val
        self.color = color
        self.left = None
        self.right = None
        self.parent = None

class RedBlackTree:
    def __init__(self):
        self.NIL = RBNode(0, color="BLACK")
        self.root = self.NIL

    def insert(self, val):
        node = RBNode(val)
        node.left = self.NIL
        node.right = self.NIL
        y = None
        x = self.root
        while x != self.NIL:
            y = x
            if node.val < x.val:
                x = x.left
            else:
                x = x.right
        node.parent = y
        if y is None:
            self.root = node
        elif node.val < y.val:
            y.left = node
        else:
            y.right = node
        node.color = "RED"
        if self.root == node:
            node.color = "BLACK"

    def in_order(self) -> list:
        res = []
        def _walk(n):
            if n != self.NIL:
                _walk(n.left)
                res.append(n.val)
                _walk(n.right)
        _walk(self.root)
        return res
