import argparse
import pathlib
import h5py
import numpy as np
from tqdm import tqdm


def create_dataset(
        in_file: pathlib.Path,
        out_file: pathlib.Path,
        input_length: int, 
        target_length: int, 
        rain_amount_thresh: float
    ) -> None:

    with h5py.File(in_file, "r", rdcc_nbytes=1024 ** 3) as orig_f:
        train_images = orig_f["train"]["images"]
        train_timestamps = orig_f["train"]["timestamps"]
        test_images = orig_f["test"]["images"]
        test_timestamps = orig_f["test"]["timestamps"]

        print("Train shape", train_images.shape)
        print("Test shape", test_images.shape)

        imgSize = train_images.shape[1]
        num_pixels = imgSize * imgSize

        with h5py.File(out_file, "w", rdcc_nbytes=1024 ** 3) as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")

            train_image_dataset = train_set.create_dataset(
                name="images",
                shape=(1, input_length + target_length, imgSize, imgSize),
                maxshape=(None, input_length + target_length, imgSize, imgSize),
                dtype='float32', 
                compression="gzip", 
                compression_opts=9
            )

            train_timestamp_dataset = train_set.create_dataset(
                name="timestamps", 
                shape=(1, input_length + target_length, 1),
                maxshape=(None, input_length + target_length, 1),
                dtype=h5py.special_dtype(vlen=str), 
                compression="gzip",
                compression_opts=9
            )
            
            test_image_dataset = test_set.create_dataset(
                name="images", 
                shape=(1, input_length + target_length, imgSize, imgSize),
                maxshape=(None, input_length + target_length, imgSize, imgSize),
                dtype='float32', 
                compression="gzip", 
                compression_opts=9
            )
            
            test_timestamp_dataset = test_set.create_dataset(
                name="timestamps", 
                shape=(1, input_length + target_length, 1),
                maxshape=(None, input_length + target_length, 1),
                dtype=h5py.special_dtype(vlen=str), 
                compression="gzip",
                compression_opts=9
            )

            origin = [[train_images, train_timestamps], [test_images, test_timestamps]]
            datasets = [[train_image_dataset, train_timestamp_dataset], [test_image_dataset, test_timestamp_dataset]]
            for origin_id, (images, timestamps) in enumerate(origin):
                image_dataset, timestamp_dataset = datasets[origin_id]
                first = True
                for i in tqdm(range(input_length + target_length, len(images))):
                    if np.sum(images[i] > 0) >= num_pixels * rain_amount_thresh:
                        imgs = images[i - (input_length + target_length):i]
                        timestamps_img = timestamps[i - (input_length + target_length):i]

                        if first:
                            first = False
                        else:
                            image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                            timestamp_dataset.resize(timestamp_dataset.shape[0] + 1, axis=0)

                        image_dataset[-1] = imgs
                        timestamp_dataset[-1] = timestamps_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-path",
        type=pathlib.Path,
        help="Path to store original dataset file.",
    )

    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        help="Path to store new dataset file.",
    )

    parser.add_argument(
        "--input-length",
        type=int,
        default=18,
        help="Number of precipitation maps in model input.",
    )

    parser.add_argument(
        "--target-length",
        type=int,
        default=12,
        help="Number of precipitation maps in prediction target.",
    )

    parser.add_argument(
        "--rain-threshold",
        type=float,
        default=0.5,
        help="Minimum fraction of rainy pixels in last prediction target.",
    )

    args = parser.parse_args()

    create_dataset(
        in_file=args.input_path,
        out_file=args.output_path,
        input_length=args.input_length,
        target_length=args.target_length,
        rain_amount_thresh=args.rain_threshold
    )