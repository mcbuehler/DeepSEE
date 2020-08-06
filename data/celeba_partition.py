import os
import pathlib


def run(path_partitions, path_in, path_out):
    assert os.path.exists(path_in), "Path does not exist: "+path_in
    print("Symlinking images from {} to {}...".format(path_in, path_out))
    splits = {'0': "train", '1': "val", '2': "test"}
    for split in splits.values():
        pathlib.Path(os.path.join(path_out, split)).mkdir(parents=True, exist_ok=True)

    counter = 0

    with open(path_partitions, 'r') as f:
        for line in f:
            if counter % 1000 == 0:
                print("Counter: ", counter)
            filename, split = line.split()
            os.symlink(os.path.join(path_in, filename), os.path.join(path_out, splits[split], filename))
            counter += 1
    print("Done. {} Files linked.".format(counter))


if __name__ == "__main__":
    # Use absolute paths!
    # YOUR CELEBA_PATH/CelebA should contain the CelebA images
    path_partitions = "YOUR_CELEBA_PATH/CelebA_list_eval_partition.txt"
    path_in = "YOUR_CELEBA_PATH/CelebA/"
    path_out = "YOUR_DATA_PATH/CelebA/images/"
    if "YOUR_CELEBA_PATH" in path_partitions:
        print("Please update the variables to the correct paths.")
        exit()
    run(path_partitions, path_in, path_out)