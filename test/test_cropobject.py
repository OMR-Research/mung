import unittest

from muscima.cropobject import CropObject


class CropObjectTest(unittest.TestCase):
    def test_bbox_to_integer_bounds(self):
        # Arrange
        expected = (44, 18, 56, 93)
        expected2 = (44, 18, 56, 93)

        # Act
        actual = CropObject.bbox_to_integer_bounds(44.2, 18.9, 55.1, 92.99)
        actual2 = CropObject.bbox_to_integer_bounds(44, 18, 56, 92.99)

        # Assert
        self.assertEqual(actual, expected)
        self.assertEqual(actual2, expected2)

    def test_overlaps(self):
        # Arrange
        crop_object = CropObject(0, 'test', 10, 100, height=20, width=10)

        # Act and Assert
        self.assertEqual(crop_object.bounding_box, (10, 100, 30, 110))

        self.assertTrue(crop_object.overlaps((10, 100, 30, 110)))  # Exact match

        self.assertFalse(crop_object.overlaps((0, 100, 8, 110)))  # Row mismatch
        self.assertFalse(crop_object.overlaps((10, 0, 30, 89)))  # Column mismatch
        self.assertFalse(crop_object.overlaps((0, 0, 8, 89)))  # Total mismatch

        self.assertTrue(crop_object.overlaps((9, 99, 31, 111)))  # Encompasses CropObject
        self.assertTrue(crop_object.overlaps((11, 101, 29, 109)))  # Within CropObject
        self.assertTrue(crop_object.overlaps((9, 101, 31, 109)))  # Encompass horz., within vert.
        self.assertTrue(crop_object.overlaps((11, 99, 29, 111)))  # Encompasses vert., within horz.
        self.assertTrue(crop_object.overlaps((11, 101, 31, 111)))  # Corner within: top left
        self.assertTrue(crop_object.overlaps((11, 99, 31, 109)))  # Corner within: top right
        self.assertTrue(crop_object.overlaps((9, 101, 29, 111)))  # Corner within: bottom left
        self.assertTrue(crop_object.overlaps((9, 99, 29, 109)))  # Corner within: bottom right


if __name__ == '__main__':
    unittest.main()
