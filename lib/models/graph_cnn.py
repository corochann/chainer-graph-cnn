#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer import reporter

from lib import coarsening
from lib.functions.pooling.graph_max_pooling import GraphMaxPoolingFunction
from lib.links.connection.graph_convolution import GraphConvolution


class GraphCNN(chainer.Chain):
    """Graph CNN example implementation.

    Uses the GC32-P4-GC64-P4-FC512 architecture as in the original paper.
    """

    def __init__(self, A, n_out=10):
        super(GraphCNN, self).__init__()

        # Precompute the coarsened graphs
        graphs, pooling_inds = coarsening.coarsen(A, levels=4)
        # In order to simulate 2x2 max pooling, combine the 4 levels
        # of graphs into 2 levels by combining pooling indices.
        graphs, pooling_inds = coarsening.combine(graphs, pooling_inds, 2)

        graph_conv_layers_list = []

        # sizes for graph conv layers
        graph_pool_list = []
        sizes = [32, 64]
        for i, (g, inds, s) in enumerate(zip(graphs, pooling_inds, sizes)):
            f = GraphConvolution(None, s, g, K=25)
            p = GraphMaxPoolingFunction(inds)
            graph_conv_layers_list.append(f)
            graph_pool_list.append(p)

        # sizes for linear layers
        sizes = [512]
        with self.init_scope():
            self.graph_conv_layers = chainer.ChainList(
                *graph_conv_layers_list)
            self.linear_layers = chainer.ChainList(
                *[L.Linear(None, s) for s in sizes])
            self.cls_layer = L.Linear(None, n_out)
        self.graph_pool_functions = graph_pool_list

    def __call__(self, x, *args):
        # x.shape = (n_batch, n_channels, h*w)
        dropout_ratio = 0.5

        h = x
        # Graph convolutional layers
        for f, p in zip(self.graph_conv_layers, self.graph_pool_functions):
            h = p(F.relu(f(h)))

        # Fully connected layers
        for f in self.linear_layers:
            h = F.relu(F.dropout(f(h), dropout_ratio))

        # Linear classification layer
        h = self.cls_layer(h)

        if args:
            labels = args[0]
            loss = F.softmax_cross_entropy(h, labels)
            acc = accuracy.accuracy(h, labels)
            reporter.report({
                'loss': loss,
                'accuracy': acc},
                self)

            return loss

        return h
