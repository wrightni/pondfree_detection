# Script for developing and testing analysis techniques on processed images
import os
import glob
import csv
import time
import numpy as np
from scipy.stats import scoreatpercentile
from scipy import ndimage
from skimage import feature
from skimage.draw import line, circle
import gdal
from tqdm import tqdm


def main():

    # FILE PATHS MUST BE CHANGED PRIOR TO USE

    # Directories for input and output files
    dst_dir = '/.../data/2016_07_19/2016_07_19_out'
    clsf_dir = '/.../data/2016_07_19/2016_07_19_clsf'

    batch_process = True
    # To batch analyze many images:
    if batch_process:

        # Swap these two depending on if you are saving optical image examples
        raw_dir = '/.../data/2016_07_19/2016_07_19_raw/07192016_raw/*{}*.JPG'
        basename = '2016_07_19_0{}_distance_transform_{}.tif'
        # raw_dir = None
        # basename = None

        batch_analyze(clsf_dir, raw_dir, dst_dir, basename)

    # To analyze a single image for testing:
    else:
        basename = '2016_07_19_0{}_distance_transform_{}.tif'

        frame = 2885
        # Test Files
        src_file = '/.../data/2016_07_19/2016_07_19_clsf/2016_07_19_0{}_classified.tif'.format(frame)

        dst_name = assign_dstname(dst_dir, basename, frame)
        dst_file = os.path.join(dst_dir, dst_name)

        src_ds = gdal.Open('/.../data/2016_07_19/2016_07_19_raw/07192016_raw/2016_07_19_0{}.JPG'.format(frame))
        raw_im = src_ds.ReadAsArray()
        src_ds = None

        # Load the clsf data
        start = time.time()
        clsf_image = load_classified(src_file)
        print("Loading: {}".format(time.time() - start))

        # Turn the classified image into a binary where 0 is background (not ice) and 1 is foreground (ice)
        start = time.time()
        binary_image = clsf_to_binary(clsf_image, fg_class=[1])
        print("To Binary: {}".format(time.time() - start))

        # Calculate the euclidian distance transform
        start = time.time()
        edt_image = ndimage.distance_transform_edt(binary_image)
        print("EDT Transform: {}".format(time.time() - start))

        start = time.time()
        unponded_markers, all_peaks, connections = define_unponded(edt_image, binary_image)
        print("Find Unponded: {}".format(time.time() - start))

        # Radius of 27.5 m as strong point threshold
        strpt_thresh = np.pi * 275 ** 2

        # Color the various points in the demo image
        for u in unponded_markers:
            strength = u[3]
            w = int(np.sqrt(strength / np.pi))# / 5)
            try:
                if strength > strpt_thresh:
                    rr, cc = circle(u[0], u[1], w)
                    raw_im[0, rr, cc] = raw_im[0, rr, cc] + 10
                    raw_im[1, rr, cc] = raw_im[1, rr, cc] + 40
                    raw_im[2, rr, cc] = raw_im[2, rr, cc] + 50
                else:
                    rr, cc = circle(u[0], u[1], w)
                    # raw_im[0, rr, cc] = 50
                    # raw_im[1, rr, cc] = 255
                    # raw_im[2, rr, cc] = 10
            except IndexError:
                print("Error coloring circle...")
                continue

        for u in unponded_markers:
            strength = u[3]
            w = 20
            try:
                if strength > strpt_thresh:
                    rr, cc = circle(u[0], u[1], w)
                    raw_im[0, rr, cc] = 51
                    raw_im[1, rr, cc] = 160
                    raw_im[2, rr, cc] = 255
                else:
                    rr, cc = circle(u[0], u[1], w)
                    raw_im[0, rr, cc] = 255
                    raw_im[1, rr, cc] = 166
                    raw_im[2, rr, cc] = 25
            except IndexError:
                print("Error coloring circle...")
                continue

        print("Number of unponded regions: {}".format(len(unponded_markers)))

        # Save the demo image
        write_raster(dst_file, raw_im)


def batch_analyze(clsf_dir, raw_dir, dst_dir, basename):
    '''
    Takes a list of csv files (full file path) and merges them into a
    single output.
    '''
    # Create a list of all the md files in the given directory
    image_list = glob.glob(os.path.join(clsf_dir, '*.tif'))
    image_list.sort()

    output_csv = os.path.join(dst_dir, '20170725b.csv')

    print("Analyzing pond coverage for {} images...".format(len(image_list)))
    # Read each file and add it to the aggregate list
    for file in tqdm(image_list):

        frame = int(file.split('_')[-2])

        # Load the clsf data
        clsf_image = load_classified(file)
        # Turn the classified image into a binary where 0 is background (not ice) and 1 is foreground (ice)
        binary_image = clsf_to_binary(clsf_image, fg_class=[1])
        # Calculate the euclidian distance transform
        edt_image = ndimage.distance_transform_edt(binary_image)

        unponded_markers = define_unponded(edt_image, binary_image)


        if not os.path.isfile(output_csv):
            open_flag = 'w'
        else:
            open_flag = 'a+'

        # Circle with radius of 275 pixels (27.5m)
        strpt_thresh = np.pi * 275**2

        with open(output_csv, open_flag) as csvfile:
            writer = csv.writer(csvfile)
            # Save the total strength of all regions
            weak_points = []
            strong_points = []
            for u in unponded_markers:
                strength = u[3]

                if strength > strpt_thresh:
                    strong_points.append(strength)
                else:
                    weak_points.append(strength)

            smn, sq1, smd, sq3, smx = fivenum(strong_points)
            wmn, wq1, wmd, wq3, wmx = fivenum(weak_points)

            # Write the number of regions and the sum of their sizes
            writer.writerow([frame, len(strong_points), smn, sq1, smd, sq3, smx,
                             len(weak_points), wmn, wq1, wmd, wq3, wmx])

        # Save every nth image for inspection
        # if frame % 10 == 0:
        #     for u in unponded_markers:
        #         strength = u[3]
        #         w = int(np.sqrt(strength / np.pi))
        #         try:
        #             if strength > strpt_thresh:
        #                 rr, cc = circle(u[0], u[1], w)
        #                 raw_im[0, rr, cc] = 255
        #                 # raw_im[1, rr, cc] = 0
        #                 # raw_im[2, rr, cc] = 0
        #             else:
        #                 rr, cc = circle(u[0], u[1], w)
        #                 # raw_im[0, rr, cc] = 50
        #                 raw_im[1, rr, cc] = 255
        #                 # raw_im[2, rr, cc] = 10
        #         except IndexError:
        #             print("Error coloring circle...")
        #             continue
        #     dst_name = assign_dstname(dst_dir, basename, frame)
        #     dst_file = os.path.join(dst_dir, dst_name)
        #     write_raster(dst_file, raw_im)
        # raw_im = None


def fivenum(v):
    """Returns five number summary (minimum, 25q, median, 75q, maximum)
    for the input vector, a list or array of numbers based on 1.5 times the interquartile distance"""

    if len(v) == 0:
        return 0, 0, 0, 0, 0

    q1 = scoreatpercentile(v,25)
    q3 = scoreatpercentile(v,75)
    # iqd = q3-q1
    md = np.median(v)

    return np.min(v), q1, md, q3, np.max(v)


def overlapping_circle_area(r1, r2, d):
    # print((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    return ((r1 ** 2) * np.arccos(np.deg2rad((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))) +
            (r2 ** 2) * np.arccos(np.deg2rad((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))) -
            0.5 * np.sqrt((r1 + r2 - d) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
            )

def define_unponded(edt_image, binary_image):

    threshold = 10 * 17      # 10p = 1m, * x meters

    peaks = feature.peak_local_max(edt_image, threshold_abs=threshold, min_distance=threshold, exclude_border=False)

    if len(peaks) == 0:
        return []
    # List of the magnitude of each peak, same order as peaks
    peak_size = []
    # Append the size of each peak to the list of peaks
    for peak in peaks:
        radius = edt_image[peak[0], peak[1]]
        area = np.pi * radius**2
        peak_size.append([radius, area])

    peaks = np.append(peaks, peak_size, axis=1)
    # Remove any two peaks that are both close together and have no water along a
    # #   straight line connecting them.
    all_peaks = peaks
    connections = []        # To plot the lines connecting points
    to_remove = []
    for i in range(len(peaks)):
        for j in range(i+1, len(peaks)):
            if i in to_remove or j in to_remove:
                continue
            # Each point is [y, x, magnitude]
            # magnitude = radius of circle
            pt_a = peaks[i]
            pt_b = peaks[j]
            # Find the line between these two points
            rr, cc = line(int(pt_a[0]), int(pt_a[1]), int(pt_b[0]), int(pt_b[1]))
            dist = len(rr)
            size_a = pt_a[2]
            size_b = pt_b[2]
            if 5 < dist < size_a + size_b:
                connections.append([rr, cc])
                # If the sum = the length, all pixels are ice (value 1)
                if np.sum(binary_image[rr, cc]) == len(rr):
                    # Merge the areas of each circle, minus the area of overlap
                    area_a = pt_a[3]
                    area_b = pt_b[3]
                    overlap = overlapping_circle_area(size_a, size_b, dist)

                    if size_a > size_b:
                        to_remove.append(j)
                        # Add the area of the smaller circle (minus the overlap) to the sum
                        peaks[i][3] += (area_b - overlap)
                    else:
                        to_remove.append(i)
                        # Add the area of the smaller circle (minus the overlap) to the sum
                        peaks[j][3] += (area_a - overlap)

    peaks = np.delete(peaks, to_remove, axis=0)

    return peaks, all_peaks, connections


def clsf_to_binary(image_data, fg_class):
    # Converts a clsf image to binary (foreground/background)
    # The foreground is set to include all categories given in the fg_class list.
    binary_image = np.zeros_like(image_data)
    # Set the foreground pixels to 1
    for cat in fg_class:
        binary_image[image_data==cat] = 1

    # Set the borders to zero
    binary_image[:, 0] = 0
    binary_image[:, -1] = 0
    binary_image[0, :] = 0
    binary_image[-1, :] = 0

    return binary_image


def load_classified(filename):
    # Loads a classified file into a numpy array.
    try:
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
    except RuntimeError:
        print("Error opening {}".format(filename))
        return 0

    raster_data = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    return raster_data

def assign_dstname(dst_dir, basename, frame):
    # Temp files in the output folder are named sequentially
    filelist = glob.glob(os.path.join(dst_dir, basename.format(frame, '*')))
    next_index = len(filelist) + 1

    filename = basename.format(frame, next_index)

    return filename


def write_raster(filename, image_data):

    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)

    im_shape = np.shape(image_data)
    ndims = len(im_shape)

    # Different protocol for 2d vs 3d imagery
    if ndims == 3:
        nbands, ysize, xsize = im_shape
    else:
        ysize, xsize = im_shape
        nbands = 1

    dst_ds = driver.Create(filename, xsize=xsize, ysize=ysize, bands=nbands,
                           eType=gdal.GDT_Byte, options=["COMPRESS=LZW"])

    # Different protocol for 2d vs 3d imagery
    if ndims == 3:
        for b in range(1, nbands + 1):
            dst_ds.GetRasterBand(b).WriteArray(image_data[b - 1, :, :])
    else:
        dst_ds.GetRasterBand(1).WriteArray(image_data[:, :])

    dst_ds.FlushCache()
    dst_ds = None


if __name__ == '__main__':
    main()
