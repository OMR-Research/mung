import os
import unittest

from mung.io import parse_cropobject_list, export_cropobject_list
from mung.node import Node


class NodeTest(unittest.TestCase):
    def test_bbox_to_integer_bounds(self):
        # Arrange
        expected = (44, 18, 56, 93)
        expected2 = (44, 18, 56, 93)

        # Act
        actual = Node.bbox_to_integer_bounds(44.2, 18.9, 55.1, 92.99)
        actual2 = Node.bbox_to_integer_bounds(44, 18, 56, 92.99)

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

    def test_parse_node_list(self):
        test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test', 'test_data',
                                     'cropobjects_xy_vs_topleft')
        clfile = os.path.join(test_data_dir, '01_basic_topleft.xml')
        cropobjects = parse_cropobject_list(clfile)
        self.assertEqual(len(cropobjects), 48)

        clfile_xy = os.path.join(test_data_dir, '01_basic_xy.xml')
        cropobjects_xy = parse_cropobject_list(clfile_xy)
        self.assertEqual(len(cropobjects_xy), 48)
        self.assertEqual(len(cropobjects), len(cropobjects_xy))

        export_xy = export_cropobject_list(cropobjects_xy)
        with open(clfile) as hdl:
            raw_data_topleft = '\n'.join([l.rstrip() for l in hdl])
        self.assertEqual(raw_data_topleft, export_xy)


if __name__ == '__main__':
    unittest.main()
