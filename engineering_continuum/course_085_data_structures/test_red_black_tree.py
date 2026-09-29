from red_black_tree import RedBlackTree

def test_rb():
    t = RedBlackTree()
    t.insert(5)
    t.insert(3)
    assert t.in_order() == [3, 5]
