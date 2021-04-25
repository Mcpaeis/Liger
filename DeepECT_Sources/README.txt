===================================================================================================
+++Content
===================================================================================================
The file Sources.tar.gz contains the sources code for the Embedded Cluster Tree (DeepECT),
described in the Paper "Embedded Cluster Tree", as well as implementations for the baseline methods.
The file pre-trained.tar.gz contains the pre-trained autoencoders we used for the experiments.

===================================================================================================
+++Setup
===================================================================================================
The following library versions are necessary to reproduce these results:
    Python 3.7
    Scikit-learn 0.20.2      
    Scipy 1.2.0
    Pytorch 1.0.0
    Torchvision 0.2.2
    Cuda 10.0.130
    Attr 19.1.0
    Matplotlib 3.0.3
    Imageio 2.5.0-2
    Requests 2.22.0 

    
In scripts/Config.py three paths need to be set. A directory for the datasets, a directory for the results and a directory in which the pre-trained autoencoders are stored.

===================================================================================================
+++Preprocessing and Pre-training
===================================================================================================
Preprocessing the reuters dataset, can be done by first adding the sources above to PYTHONPATH.
Then run 'scripts/preprocessing/preprocess_reuters.py' with python

A new set of autoencoders can be pretrained for each dataset via 'scripts/preprocessing/pretrain_ae.py'.
The used dataset can be set as a command line argument. Either use "mnist", "usps", "reuters", or "fashion-mnist".
The parameters we used are then set accordingly. This is true for all scripts.

===================================================================================================
+++Reproducing the Experiments
===================================================================================================
The scripts in the directory 'scripts/projection_problem' reproduces the problems described in the paper when a compression loss without projection is used for (IDEC and unprojected DeepECT) and the impact of the projection used in DeepECT.
The experiments were conducted on a standard Personal Computer with an AMD Ryzen 7 2700X Eight-Core Processor, 64GB RAM and a GeForce GTX 1080 Ti GPU.


Note: The exepriments also show---next to the results shown in the paper---results for measures called NMI(best p-tree) and ACC (best p-tree) restults.
These are the results we get, if we ignore the split-order and based would collapse the tree based on the ground truth.
WE DID NOT USE THESE RESULTS IN THE PAPER! 
It would be an unfair advantage for DeepECT over the baseline methods, however, these results show that the tree partitions the data in a meaningful way.
And that by switching the split-order (with the knowledge of the ground truth) we cloud get better results.


