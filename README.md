StructPool
=============
The code for our ICLR paper: StructPool: Structured Graph Pooling via Conditional Random Fields

=============

The code is built based on DGCNN(https://github.com/muhanzhang/pytorch_DGCNN) and Graph UNet(https://github.com/HongyangGao/Graph-U-Nets). Thanks a lot for their code sharing! 

The proposed Pooling Layer
=============

We first employ GCNs to obtain u(x) for a batch. Next, perform pooling for each graph.

Please refer to "DGCNN_embedding.py" and "pool.py" for details.

Installation
------------

This implementation is based on Hanjun Dai's structure2vec graph backend. Under the "lib/" directory, type

    make -j4

to compile the necessary c++ files.

After that, under the root directory of this repository, type

    ./run_DGCNN.sh

to run DGCNN on dataset MUTAG with the default setting.

Or type 

    ./run_DGCNN.sh DATANAME FOLD

to run on dataset = DATANAME using fold number = FOLD (1-10, corresponds to which fold to use as test data in the cross-validation experiments).

If you set FOLD = 0, e.g., typing "./run_DGCNN.sh DD 0", then it will run 10-fold cross validation on DD and report the average accuracy.

Alternatively, type

    ./run_DGCNN.sh DATANAME 1 200

to use the last 200 graphs in the dataset as testing graphs. The fold number 1 will be ignored.

Check "run_DGCNN.sh" for more options.

How to use your own data
------------------------

The first step is to transform your graphs to the format described in "data/README.md". You should put your testing graphs at the end of the file. Then, there is an option -test_number X, which enables using the last X graphs from the file as testing graphs. You may also pass X as the third argument to "run_DGCNN.sh" by

    ./run_DGCNN.sh DATANAME 1 X

where the fold number 1 will be ignored.

Reference
---------

    @inproceedings{
    Yuan2020StructPool:,
    title={StructPool: Structured Graph Pooling via Conditional Random Fields},
    author={Hao Yuan and Shuiwang Ji},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=BJxg_hVtwH}
    }