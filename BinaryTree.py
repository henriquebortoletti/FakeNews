class BinaryTree:
    "Tree structure used for decision tree"
    def __init__(self, node=None):
        if node is None:
            self.has_no_word = None
            self.has_word = None
            self.frequency = -1
            self.value = -1
            self.gini = -1

    def insertHasWord(self, tree):
        if self.has_word is None:
            self.has_word = tree
        else:
            self.has_word.insertHasWord(tree)

    def insertHasNoWord(self, tree):
        if self.has_no_word is None:
            self.has_no_word = tree
        else:
            self.has_no_word.insertHasNoWord(tree)


    def showTree(self):
        print(self.value)
        if self.has_no_word is not None:
            self.has_no_word.showTree()
        if self.has_word is not None:
            self.has_word.showTree()

    def find(self, case):
        if self.has_word is None or self.has_no_word is None:
            return self.value
        if case[self.value] != 0:
            return self.has_word.find(case)
        else:
            return self.has_no_word.find(case)
