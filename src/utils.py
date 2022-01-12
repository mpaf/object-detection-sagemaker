import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2

def show_species(images_dir, species_id):
    _im_list = os.listdir(os.path.join("./{}".format(images_dir), species_id))

    NUM_COLS = 6
    IM_COUNT = len(_im_list)

    print('Species ' + species_id + ' has ' + str(IM_COUNT) + ' images.')

    NUM_ROWS = int(IM_COUNT / NUM_COLS)
    if ((IM_COUNT % NUM_COLS) > 0):
        NUM_ROWS += 1

    fig, axarr = plt.subplots(NUM_ROWS, NUM_COLS)
    fig.set_size_inches(8.0, 16.0, forward=True)

    curr_row = 0
    for curr_img in range(IM_COUNT):
        # fetch the url as a file type object, then read the image
        f = images_dir + species_id + '/' + _im_list[curr_img]
        a = plt.imread(f)

        # find the column by taking the current index modulo 3
        col = curr_img % NUM_ROWS
        # plot on relevant subplot
        axarr[col, curr_row].imshow(a)
        if col == (NUM_ROWS - 1):
            # we have finished the current row, so increment row counter
            curr_row += 1

    fig.tight_layout()
    plt.show()

    # Clean up
    plt.clf()
    plt.cla()
    plt.close()
    

''' generate a file with all file sizes to which bounding box coordinates
    are relative '''
def gen_image_size_file(images_dir, image_file, size_cols, size_file):
    print("Generating a file containing image sizes...")
    images_df = pd.read_csv(
        image_file, sep=" ", names=["image_pretty_name", "image_file_name"], header=None)
    rows_list = []
    idx = 0
    for i in images_df["image_file_name"]:
        # TODO: add progress bar
        idx += 1
        img = cv2.imread(images_dir + i)
        dimensions = img.shape
        height = img.shape[0]
        width = img.shape[1]
        image_dict = {"idx": idx, "width": width, "height": height}
        rows_list.append(image_dict)

    sizes_df = pd.DataFrame(rows_list)
    print("Image sizes:\n" + str(sizes_df.head()))

    sizes_df[size_cols].to_csv(size_file, sep=" ", index=False, header=None)
    
''' split a Dataframe into train and validation datasets, with a certain fraction '''
def split_to_train_test(df, label_column, train_frac=0.8):
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        print(
            "\n{}:\n---------\ntotal:{}\ntrain_df:{}\ntest_df:{}".format(
                lbl, len(lbl_df), len(lbl_train_df), len(lbl_test_df)
            )
        )
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)
    return train_df, test_df

''' generate training and validation RecordIO lst files '''
def gen_list_files(size_file, bbox_file, image_file, label_file,
                   classes,
                   im2rec_ssd_cols,
                   train_ratio,
                   train_lst_file, val_lst_file,
                  ):
    # use generated sizes file
    sizes_df = pd.read_csv(
        size_file, sep=" ", names=["image_pretty_name", "width", "height"], header=None
    )
    bboxes_df = pd.read_csv(
        bbox_file,
        sep=" ",
        names=["image_pretty_name", "x_abs", "y_abs", "bbox_width", "bbox_height"],
        header=None,
    )

    images_df = pd.read_csv(
        image_file, sep=" ", names=["image_pretty_name", "image_file_name"], header=None
    )
    print("num images total: " + str(images_df.shape[0]))
    image_class_labels_df = pd.read_csv(
        label_file, sep=" ", names=["image_pretty_name", "class_id"], header=None
    )

    # Merge the metadata into a single flat dataframe for easier processing
    full_df = pd.DataFrame(images_df)
    full_df.reset_index(inplace=True)
    full_df = pd.merge(full_df, image_class_labels_df, on="image_pretty_name")
    full_df = pd.merge(full_df, sizes_df, on="image_pretty_name")
    full_df = pd.merge(full_df, bboxes_df, on="image_pretty_name")
    full_df.sort_values(by=["index"], inplace=True)

    # Define the bounding boxes in the format required by SageMaker's built in Object Detection algorithm.
    # the xmin/ymin/xmax/ymax parameters are specified as ratios to the total image pixel size
    full_df["header_cols"] = 2  # one col for the number of header cols, one for the label width
    full_df["label_width"] = 5  # number of cols for each label: class, xmin, ymin, xmax, ymax
    full_df["xmin"] = full_df["x_abs"] / full_df["width"]
    full_df["xmax"] = (full_df["x_abs"] + full_df["bbox_width"]) / full_df["width"]
    full_df["ymin"] = full_df["y_abs"] / full_df["height"]
    full_df["ymax"] = (full_df["y_abs"] + full_df["bbox_height"]) / full_df["height"]

    # object detection class id's must be zero based. map from
    # class_id's given by CUB to zero-based (1 is 0, and 200 is 199).

    # grab a small subset of species for testing
    criteria = full_df["class_id"].isin(classes)
    full_df = full_df[criteria]

    unique_classes = full_df["class_id"].drop_duplicates()
    sorted_unique_classes = sorted(unique_classes)

    id_to_zero = {}
    i = 0.0
    for c in sorted_unique_classes:
        id_to_zero[c] = i
        i += 1.0

    full_df["zero_based_id"] = full_df["class_id"].map(id_to_zero)

    full_df.reset_index(inplace=True)

    # use 4 decimal places, as it seems to be required by the Object Detection algorithm
    pd.set_option("display.precision", 4)

    train_df = []
    val_df = []

    # split into training and validation sets
    train_df, val_df = split_to_train_test(full_df, "class_id", train_ratio)

    train_df[im2rec_ssd_cols].to_csv(train_lst_file, sep="\t", float_format="%.4f", header=None)
    val_df[im2rec_ssd_cols].to_csv(val_lst_file, sep="\t", float_format="%.4f", header=None)

    print("num train: " + str(train_df.shape[0]))
    print("num val: " + str(val_df.shape[0]))
    return train_df, val_df

def visualize_detection(img_file, dets, classes=[], thresh=0.6):
    """
    visualize detections in one image
    Parameters:
    ----------
    img : numpy.array
        image, in bgr format
    dets : numpy.array
        ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
        each row is one object
    classes : tuple or list of str
        class names
    thresh : float
        score threshold
    """
    import random
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(img_file)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    num_detections = 0
    for det in dets:
        (klass, score, x0, y0, x1, y1) = det
        if score < thresh:
            continue
        num_detections += 1
        cls_id = int(klass)
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        xmin = int(x0 * width)
        ymin = int(y0 * height)
        xmax = int(x1 * width)
        ymax = int(y1 * height)
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=colors[cls_id],
            linewidth=3.5,
        )
        plt.gca().add_patch(rect)
        class_name = str(cls_id)
        if classes and len(classes) > cls_id:
            class_name = classes[cls_id]
        print("{},{}".format(class_name, score))
        plt.gca().text(
            xmin,
            ymin - 2,
            "{:s} {:.3f}".format(class_name, score),
            bbox=dict(facecolor=colors[cls_id], alpha=0.5),
            fontsize=12,
            color="white",
        )

    print("Number of detections: " + str(num_detections))
    plt.show()