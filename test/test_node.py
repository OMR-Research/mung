import os
import unittest

from mung.io import read_nodes_from_file, export_node_list
from mung.node import Node


class NodeTest(unittest.TestCase):
    def test_bbox_to_integer_bounds(self):
        # Arrange
        expected = (44, 18, 56, 93)
        expected2 = (44, 18, 56, 93)

        # Act
        actual = Node.round_bounding_box_to_integer(44.2, 18.9, 55.1, 92.99)
        actual2 = Node.round_bounding_box_to_integer(44, 18, 56, 92.99)

        # Assert
        self.assertEqual(actual, expected)
        self.assertEqual(actual2, expected2)

    def test_overlaps(self):
        # Arrange
        node = Node(0, 'test', 10, 100, height=20, width=10)

        # Act and Assert
        self.assertEqual(node.bounding_box, (10, 100, 30, 110))

        self.assertTrue(node.overlaps((10, 100, 30, 110)))  # Exact match

        self.assertFalse(node.overlaps((0, 100, 8, 110)))  # Row mismatch
        self.assertFalse(node.overlaps((10, 0, 30, 89)))  # Column mismatch
        self.assertFalse(node.overlaps((0, 0, 8, 89)))  # Total mismatch

        self.assertTrue(node.overlaps((9, 99, 31, 111)))  # Encompasses Node
        self.assertTrue(node.overlaps((11, 101, 29, 109)))  # Within Node
        self.assertTrue(node.overlaps((9, 101, 31, 109)))  # Encompass horz., within vert.
        self.assertTrue(node.overlaps((11, 99, 29, 111)))  # Encompasses vert., within horz.
        self.assertTrue(node.overlaps((11, 101, 31, 111)))  # Corner within: top left
        self.assertTrue(node.overlaps((11, 99, 31, 109)))  # Corner within: top right
        self.assertTrue(node.overlaps((9, 101, 29, 111)))  # Corner within: bottom left
        self.assertTrue(node.overlaps((9, 99, 29, 109)))  # Corner within: bottom right

    def test_read_nodes_from_file(self):
        test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test', 'test_data')
        clfile = os.path.join(test_data_dir, '01_basic.xml')
        nodes = read_nodes_from_file(clfile)
        self.assertEqual(len(nodes), 48)

    def test_read_nodes_from_file_with_data(self):
        test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test', 'test_data')
        file = os.path.join(test_data_dir, '01_basic_binary.xml')
        nodes = read_nodes_from_file(file)
        self.assertEqual("G", nodes[0].data['pitch_step'])
        self.assertEqual(79, nodes[0].data['midi_pitch_code'])
        self.assertEqual([8, 17], nodes[0].data['precedence_outlinks'])


if __name__ == '__main__':
    unittest.main()
