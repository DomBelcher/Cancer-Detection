import cv2
import csv
import numpy as np

def data_generator (image_dir, label_file, image_size, batch_size=16, sample_prob=1):
    with open(label_file) as labelcsv:
        label_reader = csv.reader(labelcsv)
        headers = label_reader.__next__()

        labels = []

        for row in label_reader:
            if np.random.uniform() < sample_prob:
                labels.append(row)

        labels = np.array(labels)

        step = 0
        label = 0
        total_labels = len(labels)
        while True:
            # batch_labels = []
            # while len(batch_labels) < batch_size:
            #     this_label = label_reader.__next__()
            #     if np.random.uniform() < sample_prob:
            #         batch_labels.append(this_label)
            if label + batch_size > total_labels:
                label = 0
            batch_labels = labels[label:label + batch_size]
            label += batch_size


            X = np.zeros((batch_size, image_size, image_size, 1))
            y = np.zeros((batch_size))

            for im in range(0, batch_size):
                # print(batch_labels[im])
                img = cv2.imread('{}/{}.tif'.format(image_dir, batch_labels[im][0]), 0)
                img_centre = (img.shape[0]/2 - 1, img.shape[1]/2 - 1)
                crop_corner = (int(img_centre[0] - image_size/2), int(img_centre[1] - image_size/2))
                # print(crop_corner)
                crop_img = img[crop_corner[0]:crop_corner[0] + image_size, crop_corner[1]:crop_corner[1] + image_size]
                X[im,:,:,0] = crop_img/255
                y[im] = batch_labels[im][1]

            yield (X, y)