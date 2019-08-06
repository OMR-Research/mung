from mung.grammar import DependencyGrammar
from mung.io import parse_node_classes

import os
import unittest


class GrammarTest(unittest.TestCase):
    def test_parse_grammar(self):
        filepath = os.path.dirname(
            os.path.dirname(__file__)) + u'/test/test_data/mff-muscima-classes-annot.deprules'
        node_classes_path = os.path.dirname(
            os.path.dirname(__file__)) + u'/test/test_data/mff-muscima-classes-annot.xml'
        node_classes = parse_node_classes(node_classes_path)
        node_classes_dict = {node_class.name for node_class in node_classes}
        dependency_graph = DependencyGrammar(grammar_filename=filepath, alphabet=node_classes_dict)
        self.assertEqual(646, len(dependency_graph.rules))


if __name__ == '__main__':
    unittest.main()
