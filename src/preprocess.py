import mxnet
import tarfile
import utils
import os
import cv2
import shutil
import argparse
import urllib.request
import urllib
import subprocess
    
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)     
# download('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz')
# CalTech's download is (at least temporarily) unavailable since August 2020.

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-ratio', type=float, default=0.8)
    args, _ = parser.parse_known_args()

    print('Received arguments {}'.format(args))
    
# Can now use one made available by fast.ai .
    download("https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz")
    
    print("Dataset downloaded")
    
    my_tar = tarfile.open('CUB_200_2011.tgz')
    my_tar.extractall('./') # specify which folder to extract to
    my_tar.close()
    
    print("done unpacking")
    
    print(os.listdir("./"))


    BASE_DIR = "./CUB_200_2011/"
    print(os.listdir(BASE_DIR))

    IMAGES_DIR = BASE_DIR + "images/"

    CLASSES_FILE = BASE_DIR + "classes.txt"
    BBOX_FILE = BASE_DIR + "bounding_boxes.txt"
    IMAGE_FILE = BASE_DIR + "images.txt"
    LABEL_FILE = BASE_DIR + "image_class_labels.txt"

    TRAIN_LST_FILE = "birds_ssd_train.lst"
    VAL_LST_FILE = "birds_ssd_val.lst"
    
    SIZE_COLS = ["idx", "width", "height"]
    SIZE_FILE = BASE_DIR + "sizes.txt"

# We need to generate this file with image sizes as it's not provided with the dataset
    utils.gen_image_size_file(IMAGES_DIR, IMAGE_FILE, SIZE_COLS, SIZE_FILE)
    
    CLASSES = [17, 36, 47, 68, 73]

    TRAIN_LST_FILE = "birds_ssd_sample_train.lst"
    VAL_LST_FILE = "birds_ssd_sample_val.lst"

    TRAIN_RATIO = 0.8

    IM2REC_SSD_COLS = [
        "header_cols",
        "label_width",
        "zero_based_id",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "image_file_name",
    ]

    train_df, val_df = \
        utils.gen_list_files(SIZE_FILE, BBOX_FILE, IMAGE_FILE, LABEL_FILE,
                       CLASSES,
                       IM2REC_SSD_COLS,
                       TRAIN_RATIO,
                       TRAIN_LST_FILE, VAL_LST_FILE)
    

    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py",
        "im2rec.py")
    
    RESIZE_SIZE = 256
    process = subprocess.Popen(["python3", "im2rec.py", "--resize", "{}".format(RESIZE_SIZE), "--pack-label", "birds_ssd_sample", IMAGES_DIR],
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # move files to be picked up by SageMaker Processing Job Outputs
    shutil.move('./birds_ssd_sample_train.rec', '/opt/ml/processing/output/train/birds_ssd_sample_train.rec')
    shutil.move('./birds_ssd_sample_val.rec', '/opt/ml/processing/output/train/birds_ssd_sample_val.rec')