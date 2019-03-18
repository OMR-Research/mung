import unittest

from mung.io import parse_node_classes
from mung.node_class import NodeClass


class NodeClassTest(unittest.TestCase):
    def test_note_class_to_string(self):
        # Arrange
        expected = """<NodeClass>
    <Id>1</Id>
    <Name>noteheadFull</Name>
    <GroupName>primitives</GroupName>
    <Color>#FF6689</Color>
</NodeClass>"""
        node_class = NodeClass(1, "noteheadFull", "primitives", "#FF6689")

        # Act
        actual = str(node_class)

        # Assert
        self.assertEqual(actual, expected)

    def test_node_class_parsing(self):
        # Arrange
        expected_number_of_classes = 158

        # Act
        node_classes = parse_node_classes("test/test_data/mff-muscima-classes-annot.xml")

        # Assert
        self.assertEqual(len(node_classes), expected_number_of_classes)
        self.assertEqual(node_classes[0].name, "noteheadFull")


if __name__ == '__main__':
    unittest.main()
