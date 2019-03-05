from avod.builders.dataset_builder import DatasetBuilder

from scripts.preprocessing import gen_tracking_mini_batches
from scripts.preprocessing import gen_label_clusters


def main():

    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.KITTI_UNITTEST)
    dataset_config.data_split = "trainval"
    unittest_dataset = DatasetBuilder.build_kitti_tracking_dataset(dataset_config)

    gen_label_clusters.main(unittest_dataset)
    gen_tracking_mini_batches.main(unittest_dataset)


if __name__ == '__main__':
    main()
