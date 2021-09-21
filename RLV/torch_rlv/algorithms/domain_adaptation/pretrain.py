"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
import RLV.torch_rlv.algorithms.domain_adaptation.params
from RLV.torch_rlv.models.utils import make_variable, save_model


def train_src(encoder, classifier, images, labels):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        # make images and labels variable
        images = make_variable(images)
        labels = make_variable(labels.squeeze_())

        # zero gradients for optimizer
        optimizer.zero_grad()

        # compute loss for critic
        preds = classifier(encoder(images))
        loss = criterion(preds, labels)

        # optimize source classifier
        loss.backward()
        optimizer.step()

    return encoder, classifier


def eval_src(encoder, classifier, images, labels):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    print(images.shape)

    images = make_variable(images, volatile=True)
    labels = make_variable(labels)

    preds = classifier(encoder(images.float()))
    loss += criterion(preds, labels.float()).data[0]

    pred_cls = preds.data.max(1)[1]
    acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(images)
    acc /= len(images)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))