"""Dataset unit test module."""
import os
import unittest
import numpy as np
import avod.tests as tests

from avod.builders.dataset_builder import DatasetBuilder
from avod.core import constants
from avod.datasets.kitti.kitti_dataset import KittiDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class KittiDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake_kitti_dir = tests.test_path() + "/datasets/Kitti/tracking"

    def get_fake_dataset(self, data_split, directory):
        dataset_config = DatasetBuilder.copy_config(
            DatasetBuilder.KITTI_TRACKING_UNITTEST)

        # Overwrite config values
        dataset_config.data_split = data_split
        dataset_config.dataset_dir = directory

        dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config)

        return dataset

    def test_data_loading(self):
        dataset = self.get_fake_dataset('train', self.fake_kitti_dir)

        indices_to_load = [1, 5, 9]
        expected_samples = [['010001', '010004'],
                            ['010005', '010008'],
                            ['010009', '010009']]

        # Load samples before shuffling
        samples = dataset.load_samples(indices_to_load)

        # Loop through indices
        for i in range(len(indices_to_load)):
            # Check image
            sample_image_input = samples[i].get(constants.KEY_IMAGE_INPUT)
            self.assertIsNotNone(sample_image_input)
            self.assertIsInstance(sample_image_input, np.ndarray)

            # Check indexing
            sample_name = samples[i].get(constants.KEY_SAMPLE_NAME)
            self.assertEqual(sample_name, expected_samples[i])

            # Check labels
            box_labels = samples[i].get(constants.KEY_LABEL_BOXES_3D)
            self.assertIsNotNone(box_labels)
            self.assertIsInstance(box_labels[0], np.ndarray)
            self.assertIsInstance(box_labels[0][0], np.ndarray)

            class_labels = samples[i].get(constants.KEY_LABEL_CLASSES)
            self.assertIsNotNone(class_labels)
            self.assertIsInstance(class_labels[0], np.ndarray)
            self.assertIsInstance(class_labels[0][0], np.int32)

    def test_data_splits(self):
        bad_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_UNITTEST)

        # Test invalid splits
        bad_config.data_split = "bad"
        self.assertRaises(ValueError, KittiDataset, bad_config)

        # Should be "train"
        bad_config.data_split = "training"
        self.assertRaises(ValueError, KittiDataset, bad_config)

        # Should be "val"
        bad_config.data_split = "validation"
        self.assertRaises(ValueError, KittiDataset, bad_config)

        # Should be "test"
        bad_config.data_split = "testing"
        self.assertRaises(ValueError, KittiDataset, bad_config)

        # Train split
        train_dataset = self.get_fake_dataset('train', self.fake_kitti_dir)
        self.assertEqual(train_dataset.num_samples, 10)

        # Validation split
        validation_dataset = self.get_fake_dataset('val', self.fake_kitti_dir)
        self.assertEqual(validation_dataset.num_samples, 8)

        # Train + validation split
        trainval_dataset = self.get_fake_dataset('trainval',
                                                 self.fake_kitti_dir)
        self.assertEqual(trainval_dataset.num_samples, 40)

        # Test split
        test_dataset = self.get_fake_dataset('test', self.fake_kitti_dir)
        self.assertEqual(test_dataset.num_samples, 3)

    def test_batch_loading(self):
        # Training split
        dataset = self.get_fake_dataset('train', self.fake_kitti_dir)

        batch = dataset.next_batch(3)
        self.assertEqual(len(batch), 3)

        # Validation split
        dataset = self.get_fake_dataset('val', self.fake_kitti_dir)

        batch = dataset.next_batch(3)
        self.assertEqual(len(batch), 3)

        # Testing split
        dataset = self.get_fake_dataset('test', self.fake_kitti_dir)

        batch = dataset.next_batch(3)
        self.assertEqual(len(batch), 3)

        # Test split should not return any labels
        self.assertIsNone(batch[0].get('label'))

    def test_batch_wrapping(self):
        dataset = self.get_fake_dataset('train', self.fake_kitti_dir)

        batch = dataset.next_batch(10)
        self.assertEqual(len(batch), 10)
        self.assertEqual(dataset.epochs_completed, 1)

        # Should not wrap
        batch = dataset.next_batch(3)
        self.assertEqual(len(batch), 3)
        self.assertEqual(dataset.epochs_completed, 1)

        # Should wrap, and provide a full batch
        batch = dataset.next_batch(8)
        self.assertEqual(len(batch), 8)
        self.assertEqual(dataset.epochs_completed, 2)
        self.assertEqual(dataset._index_in_epoch, 1)

if __name__ == '__main__':
    unittest.main()
