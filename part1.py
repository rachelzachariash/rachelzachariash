'''import math

from skimage.feature import peak_local_max

try:

    # Python program to demonstrate erosion and
    # dilation of images.
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image, ImageDraw
    from matplotlib import cm
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def find_tfl_lights(c_image: np.ndarray, some_threshold):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    ### WRITE YOUR CODE HERE ###
    ### USE HELPER FUNCTIONS ###
    # arr = rgb_to_grey(c_image)

    final_x = []
    final_y = []
    final_rx = []
    final_ry = []
    c_image = c_image[:int(len(c_image[:, 0, 0]) * 0.45), :]
    for i in range(1, 17, 2):
        hp_kernel = high_pass(i)
        mat_r = sg.convolve(c_image[:, :, 0], hp_kernel, mode='same')
        mat_g = sg.convolve(c_image[:, :, 1], hp_kernel, mode='same')
        r_coordinates = peak_local_max(mat_r, min_distance=20, num_peaks = 10)
        # print(r_coordinates)
        g_coordinates = peak_local_max(mat_g, min_distance=20, num_peaks=10)
        final_x = final_x + (g_coordinates[:, 1]).tolist()
        final_y = final_y + g_coordinates[:, 0].tolist()
        final_rx = final_rx + (r_coordinates[:, 1]).tolist()
        final_ry = final_ry + r_coordinates[:, 0].tolist()
        if i == 9:
            c_kernel = unit_circle(i)
            r_coordinates = peak_local_max(sg.convolve(mat_r, c_kernel, mode='same'),
                                           min_distance=20, num_peaks=10)
            g_coordinates = peak_local_max(sg.convolve(mat_g, c_kernel, mode='same'), min_distance=20,
                                           num_peaks=10)
            final_x = final_x + (g_coordinates[:, 1]).tolist()
            final_y = final_y + g_coordinates[:, 0].tolist()
            final_rx = final_rx + (r_coordinates[:, 1]).tolist()
            final_ry = final_ry + r_coordinates[:, 0].tolist()
    return reduction(c_image, final_rx, final_ry, final_x, final_y)


def find_color(c_image, x_red, y_red, x_green, y_green):
    k = 0
    while k < len(x_green):
        j = x_green[k]
        i = y_green[k]
        if len(c_image) <= i or len(c_image[0]) <= j:
            k += 1
        # elif not (160 > c_image[i, j, 0] > 100 and c_image[i, j, 1] > 200 and 210 > c_image[i, j, 2] > 140):
        elif not (c_image[i, j, 1] > 150 and abs(c_image[i, j, 0] - c_image[i, j, 2]) <= 30):
            y_green.pop(k)
            x_green.pop(k)
        else:
            k += 1
    k = 0
    while k < len(x_red):
        j = x_red[k]
        i = y_red[k]
        # if not (c_image[i, j, 0] > 230 and 210 > c_image[i, j, 1] > 100 and 150 > c_image[i, j, 2] > 60):
        if not (c_image[i, j, 0] > 150 and abs(c_image[i, j, 1] - c_image[i, j, 2]) <= 30):
            y_red.pop(k)
            x_red.pop(k)
        else:
            k += 1
    return reduction(c_image, x_red, y_red, x_green, y_green)


def reduction(c_image, x_red, y_red, x_green, y_green):
    radius = 40
    i, j, k = 0, 0, 0
    while i < len(x_green):
        j = 0
        while j < len(x_green):
            if i != j:
                if math.dist([x_green[i], y_green[i]], [x_green[j], y_green[j]]) <= radius:
                    del x_green[j]
                    del y_green[j]
                    j -= 1
            j += 1

        k = 0
        while k < len(x_red):
            if math.dist([x_green[i], y_green[i]], [x_red[k], y_red[k]]) <= radius:
                # if c_image[x_green[i], y_green[i], 1] > c_image[x_red[k], y_red[k], 0]:
                # del x_green[i]
                # del y_green[i]
                #     i -= 1
                # else:
                del x_red[k]
                del y_red[k]
                k -= 1
            k += 1
        i += 1
    i = 0
    while i < len(x_red):
        j = 0
        while j < len(x_red):
            if i != j:
                if math.dist([x_red[i], y_red[i]], [x_red[j], y_red[j]]) <= radius:
                    del x_red[j]
                    del y_red[j]
                    j -= 1
            j += 1
        i += 1

    return np.array(x_red), np.array(y_red), np.array(x_green), np.array(y_green)


def high_pass(s):
    width, height = 2 * s + 1, 2 * s + 1
    map_ = [[-1 / ((2 * s + 1) * (2 * s + 1)) for x in range(width)] for y in range(height)]
    map_[s + 1][s + 1] = (1 / ((2 * s + 1) * (2 * s + 1)) * (((2 * s + 1) ** 2) - 1))
    return np.array(map_)


def unit_circle(r):
    width, height = 2 * r + 1, 2 * r + 1

    EPSILON = 2.2

    map_ = [[-0.2 for x in range(width)] for y in range(height)]
    counter = 0
    for y in range(height):
        for x in range(width):
            if math.sqrt((y - r) ** 2 + (x - r) ** 2) <= r:
                counter += 1
    for y in range(height):
        for x in range(width):
            if math.sqrt((y - r) ** 2 + (x - r) ** 2) <= r:
                map_[y][x] = 1 / counter
    return np.array(map_)


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
    return red_x, red_y, green_x, green_y

def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'data'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        red_x, red_y, green_x, green_y = test_find_tfl_lights(image, json_fn)
        return red_x, red_y, green_x, green_y
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)

def find_lights(img_path, fig, title):
    image = np.array(Image.open(img_path))
    red_x, red_y, green_x, green_y = find_tfl_lights(image,some_threshold=42)
    candidates = [(x, y) for x, y in zip(red_x, red_y)]
    candidates += [(x, y) for x, y in zip(green_x, green_y)]
    auxiliary = ["red"] * len(red_x) + ["green"] * len(green_x)

    plt.plot.mark_tfl(image, np.array(candidates), fig, len(red_x), title)
    # show_image_and_gt(np.array(Image.open(img_path)), None, None)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
    # plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    # plt.show(block=True)
    return {"candidates": candidates, "auxiliary": auxiliary}
'''
try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt

    import phase4.plot as plot

except ImportError:
    print("Need to fix the installation")
    raise


def high_pass_filter(img):
    highpass_filter = np.array([[-1 / 9, -1 / 9, -1 / 9],
                                [-1 / 9, 8 / 9, -1 / 9],
                                [-1 / 9, -1 / 9, -1 / 9]])
    return sg.convolve2d(img.T, highpass_filter, boundary='symm', mode='same')


def filter_by_color(img, color):
    return high_pass_filter(img[:, :, color])


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    highpass_red_filter = filter_by_color(c_image, 0)
    max_red_filter = maximum_filter(highpass_red_filter, 20)
    red_candidates = np.array([[i, j] for i in range(0, len(max_red_filter)) \
                               for j in range(0, len(max_red_filter[0])) \
                               if max_red_filter[i][j] == highpass_red_filter[i][j] \
                               and max_red_filter[i][j] > 30])
    x_red, y_red = [rc[0] for rc in red_candidates], [rc[1] for rc in red_candidates]

    highpass_green_filter = filter_by_color(c_image, 1)
    max_green_filter = maximum_filter(highpass_green_filter, 20)
    green_candidates = np.array([[i, j] for i in range(0, len(max_green_filter)) \
                                 for j in range(0, len(max_green_filter[0])) \
                                 if max_green_filter[i][j] == highpass_green_filter[i][j] \
                                 and max_green_filter[i][j] > 30])
    x_green, y_green = [gc[0] for gc in green_candidates], [gc[1] for gc in green_candidates]
    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    open_image = Image.open(image_path)
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(image, open_image=open_image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../../data'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


def find_lights(img_path, fig, title):
    image = np.array(Image.open(img_path))

    image = image[:int(len(image[:, 0, 0]) * 0.85), :]
    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    candidates = [(x, y) for x, y in zip(red_x, red_y)]
    candidates += [(x, y) for x, y in zip(green_x, green_y)]
    auxiliary = ["red"] * len(red_x) + ["green"] * len(green_x)

    plot.mark_tfl(image, np.array(candidates), fig, len(red_x), title)
    # show_image_and_gt(np.array(Image.open(img_path)), None, None)
    # plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
    # plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    # plt.show(block=True)
    return {"candidates": candidates, "auxiliary": auxiliary}
