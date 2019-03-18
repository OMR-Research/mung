import unittest

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


if __name__ == '__main__':
    unittest.main()
