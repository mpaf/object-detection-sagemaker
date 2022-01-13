# object-detection-sagemaker

This sample workshop is based on https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_amazon_algorithms/object_detection_birds/object_detection_birds.html .

## Dataset

The workshop uses the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset, which is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations. For detailed information about the dataset, please see the technical report linked below.

Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001. [download pdf](http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf)

## Setup

The notebooks in this workshop should be run in SageMaker Studio,using the **Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)** kernel. This kernel already includes a couple of needed libraries (MXNET, OpenCV) for image manipulation and packing images into RecordIO data records. A list of available SageMaker Studio pre-baked Kernels can be found at https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-kernels.html .

Start the workshop by jumping to [Notebook 1](01-data%20preparation.ipynb) !