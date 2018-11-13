import os
from PIL import Image
import numpy as np
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def make_dataset(dir, dir_ann):
    images = []
    ann = []
    dataset = []
    dir = os.path.expanduser(dir)
    dir_ann = os.path.expanduser(dir_ann)

    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        d1 = os.path.join(dir_ann, target)

        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        for root, _, fnames in sorted(os.walk(d1)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    ann.append(path)

    for i in range(len(images)):
        item = (images[i], ann[i])
        dataset.append(item)
    return dataset

def generator(datas, batch_size):
    n_files = len(datas)
    batch_image = []
    batch_mask = []
    while True:
        for i in range(n_files):
            for i in range(batch_size):
                image = Image.open(datas[i][0])
                resized_image = np.array(image.resize((256, 256), Image.ANTIALIAS))

                mask = Image.open(datas[i][1])
                resized_mask = mask.resize((256, 256), Image.ANTIALIAS)
                arr = np.array(resized_mask)
                arr[arr >= 1] = 1
                new_arr = arr.reshape(arr.shape + (1,))
                batch_image.append(resized_image)
                batch_mask.append(new_arr)
            yield (np.asarray(batch_image), np.asarray(batch_mask))



if __name__ == '__main__':

    dir_images = "Dataset/JPEGImages/480p"
    dir_annotations = "Dataset/Annotations/480p"

    classes,classidx = find_classes(dir_images)
    datas = make_dataset(dir_images, dir_annotations)
    generator(datas, 32)

