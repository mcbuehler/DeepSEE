import numpy as np
from evaluator.pytorch_fid.inception import InceptionV3


def get_inception_model():
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    return model


def calculate_statistics_from_act(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_batch_activations(model, batch):
    batch01 = (batch + 1) / 2
    pred = model(batch01)[0]
    return pred.cpu().data.numpy().reshape(pred.size(0), -1)