import csv
import os

import pandas as pd


def get_hq_to_celeba_mapping(hq2celeba_mapping_file):
    # Returns a dict that maps HQ --> CelebA
    mapping = {}
    with open(hq2celeba_mapping_file, 'r') as f:
        next(f)
        for row in f:
            idx, _, orig_file = row.split()
            # print(idx, orig_idx, orig_file)
            celeba_id = orig_file[:-4]
            hq_id = idx
            mapping[hq_id] = celeba_id
    return mapping


def get_celeba_identities(celeba_identities_file):
    identities = {}
    with open(celeba_identities_file, 'r') as f:
        for row in f:
            filename, identity = row.split()
            file_id = filename[:-4]
            identities[file_id] = identity
    return identities


def get_HQ_file2identity_dict(hq2celeba_mapping, identities):
    # Returns the identity for hq file id (without extension .jpg or .png)
    file2id = {}

    for file_id in hq2celeba_mapping.keys():
        # file_id = file.split('/')[-1][:-4]
        celeba_file_id = hq2celeba_mapping[file_id]
        identity = identities[celeba_file_id]
        file2id[file_id] = identity
    return file2id


def combine_write_csv(hq2celeba_mapping_file, celeba_identities_file, path_out):
    hq2celeba = get_hq_to_celeba_mapping(hq2celeba_mapping_file)
    identities = get_celeba_identities(celeba_identities_file)
    hq2identity = get_HQ_file2identity_dict(hq2celeba, identities)
    print("HQ2Identity computed")

    print("Computing counts...")
    counts = {}
    for file_id, identity in hq2identity.items():
        if identity not in counts:
            counts[identity] = 1
        else:
            counts[identity] += 1

    print("Preparing dataframe")
    data = []
    for hq_file_id in hq2identity:
        sample = {
            # 'split': split,
            'hq_file_id': hq_file_id,
            'celeba_file_id': hq2celeba[hq_file_id],
            'identity': hq2identity[hq_file_id],
            'count': counts[hq2identity[hq_file_id]]
        }
        data.append(sample)

    df = pd.DataFrame(data, columns=['hq_file_id', 'celeba_file_id', 'identity', 'count'])
    df.to_csv(path_out, quoting=csv.QUOTE_ALL)
    print("{} entries with more than a single count.".format(len(df[df['count'] > 1])))


if __name__ == "__main__":
    path_in = "PATHTOCELEBAMASK-HQ/CelebAMask-HQ"
    mapping_file = os.path.join("CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt")
    identities_file = os.path.join("PATH/identity_CelebA.txt")
    path_out = "YOURPATH/CelebAMask-HQ/identities_all.csv"

    if "YOURPATH" in path_out:
        print("Did you update the in- and output paths?")
        exit()

    combine_write_csv(mapping_file, identities_file, path_out=path_out)




