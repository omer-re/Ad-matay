# -*- coding: utf-8 -*-

'''
    For the given path, get the List of all files in the directory tree
'''
import os
import time
import timeit

import cv2
import imutils
import numpy as np
from PIL import Image

import align_faces
import trim_margin
import upload_to_g_photos

failed = 0


def calc_angle_of_skewed_rectangle(contours_dots_array):
    c = max(contours_dots_array, key=cv2.contourArea)

    ## points not as tuple
    left = c[c[:, :, 0].argmin()][0]
    print("left ", left)
    right = c[c[:, :, 0].argmax()][0]
    top = c[c[:, :, 1].argmin()][0]
    print("top ", top)
    bottom = c[c[:, :, 1].argmax()][0]
    ## use left and top
    angle = np.arctan2((top[1] - left[1]), (top[0] - left[0]))  # radians
    angle_in_deg = angle * 180 / np.pi
    return angle_in_deg


def straighten_image(entry_path):
    # Load image, grayscale, Gaussian blur, threshold
    image = cv2.imread(entry_path)
    img_for_angle = cv2.imread(entry_path, cv2.IMREAD_GRAYSCALE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # blur = cv2.bilateralFilter(blur, 9, 75, 75)
    cv2.imwrite("blurred.jpg", blur)

    thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imwrite("threshed.jpg", thresh)

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    ## points not as tuple
    left = c[c[:, :, 0].argmin()][0]
    # print("left ", left)
    right = c[c[:, :, 0].argmax()][0]
    top = c[c[:, :, 1].argmin()][0]
    # print("top ", top)
    bottom = c[c[:, :, 1].argmax()][0]
    ## use left and top
    angle = np.arctan2((top[1] - left[1]), (top[0] - left[0]))  # radians
    print(angle)
    angle_in_deg = angle * 180 / np.pi
    print(angle_in_deg)

    deskewed_im = imutils.rotate_bound(image, int(angle_in_deg))
    deskewed_im2 = imutils.rotate_bound(image, 14)
    just_dir, fname = os.path.split(entry_path)
    cv2.imwrite("entry_path2.jpg", deskewed_im)
    cv2.imwrite("entry_path3.jpg", deskewed_im2)
    cv2.imwrite(entry_path, deskewed_im)


def check_valid_size(image_path, desired_bigger_dim, desired_smaller_dim):
    img = Image.open(image_path)
    curr_width, curr_height = img.size

    # find smaller dim and compress bigger to it by proper ratio
    if (curr_width < curr_height):
        new_height = int(float(curr_width / desired_bigger_dim) * desired_smaller_dim)
        new_width = curr_width
        img = img.resize((new_height, new_width), Image.ANTIALIAS)
    else:
        new_height = curr_height
        new_width = int(float(curr_height / desired_smaller_dim) * desired_bigger_dim)
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

    img.save(image_path)


def is_ratio_valid_image(image_path):
    img = Image.open(image_path)
    curr_width, curr_height = img.size

    # find smaller dim and compress bigger to it by proper ratio
    if (curr_width < curr_height):
        if float(curr_width / curr_height) < 0.7 or float(curr_width / curr_height) > 0.8:
            # might be invalid cropping
            return False
    elif (curr_width > curr_height):
        if float(curr_height / curr_width) < 0.7 or float(curr_height / curr_width) > 0.8:
            # might be invalid cropping
            return False
    else:
        return True


"""# make sure it is big enough to be cropped, and cropping if it does
def trim(im_PIL_obj):
    print(im_PIL_obj)
    #im_PIL_obj = Image.open(open(path,'rb'))
    #try:
    bg = Image.new(im_PIL_obj.mode, im_PIL_obj.size, im_PIL_obj.getpixel((0, 0)))
    diff = ImageChops.difference(im_PIL_obj, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    dim1 = bbox[2] - bbox[0]
    dim2= bbox[3] - bbox[1]
    MIN_DIM = 1000
    if bbox:
        print (bbox, dim1, dim2)
        # validate box isn't too small or cropping isn't reasonable proportions
        if dim1 > MIN_DIM and dim2 > MIN_DIM and is_ratio_valid_coordinates(dim1, dim2):
            print("return bbox #144")
            return im_PIL_obj.crop(bbox)
    else:
        print("no crop")
        return
    # except:
    #    print("An exception occurred ")
    #except Exception as exception:
        traceback.print_exc()
        # print("Exception: {}".format(type(exception).__name__))
        # print("Exception message: {}".format(exception))
        #return
"""


def orig_getListOfFiles(dirName, recursive_search):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    ##

    ##
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath) and recursive_search == 1:
            allFiles = allFiles + getListOfFiles(fullPath, int(recursive_search))
        # elif (os.path.isfile(fullPath) and entry.endswith(".jpg")):
        elif (entry.endswith('.jpg') or entry.endswith('.jpeg') or entry.endswith('.png')):
            allFiles.append(fullPath)

    return allFiles


def getListOfFiles(dirName, recursive_search):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    ##
    ##
    allFiles = []
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath) and recursive_search == 1:
            allFiles = allFiles + getListOfFiles(fullPath, int(recursive_search))
        # elif (os.path.isfile(fullPath) and entry.endswith(".jpg")):
        elif (entry.endswith('.jpg') or entry.endswith('.jpeg') or entry.endswith('.png')):
            allFiles.append(os.path.join(fullPath))

    return allFiles


def resize_to_ratio(image_path, desired_bigger_dim, desired_smaller_dim):
    img = Image.open(image_path)
    curr_width, curr_height = img.size

    # find smaller dim and compress bigger to it by proper ratio
    if (curr_width < curr_height):
        new_height = int(float(curr_width / desired_bigger_dim) * desired_smaller_dim)
        new_width = curr_width
        img = img.resize((new_height, new_width), Image.ANTIALIAS)
    else:
        new_height = curr_height
        new_width = int(float(curr_height / desired_smaller_dim) * desired_bigger_dim)
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

    img.save(image_path)


def print_unsaved_files():
    L = "Following files had problems: "
    print(L)
    for file in unsaved_files:
        print(file)
    file1 = open("scan_handler log.txt", "w")
    # file1.write("Hello \n")
    file1.writelines(L)
    file1.close()  # to change file access modes


def upload_to_google_photos(album_name, photos_list):
    auth_file_path = 'client_id.json'
    upload_to_g_photos.logging.basicConfig(format='%(asctime)s %(module)s.%(funcName)s:%(levelname)s:%(message)s',
                                           datefmt='%m/%d/%Y %I_%M_%S %p',
                                           filename='log_file.txt',
                                           level=upload_to_g_photos.logging.INFO)

    session = upload_to_g_photos.get_authorized_session(auth_file_path)

    upload_to_g_photos.upload_photos(session, photos_list, album_name)

    # As a quick status check, dump the albums and their key attributes

    print("{:<50} | {:>8} | {} ".format("PHOTO ALBUM", "# PHOTOS", "IS WRITEABLE?"))

    for a in upload_to_g_photos.getAlbums(session):
        print(
            "{:<50} | {:>8} | {} ".format(a["title"], a.get("mediaItemsCount", "0"), str(a.get("isWriteable", False))))
    print("Done uploading :)")


def handler(crop_counter, rotate_counter, dirName, recursive_search, opt_crop):
    # handles the hebrew names of folders
    counter = 0
    for it in os.scandir(dirName):
        counter += 1
        if it.is_dir():
            head, _folder = os.path.split(it)
            source = it
            dest = os.path.join(head, 'temp_name{}'.format(counter))
            counter += counter
            print('\x1b[0;33;40m' + 'head= {}\n_folder= {}\n source= {}\ndest= {}'.format(head, _folder, source,
                                                                                          dest) + '\x1b[0m')

            folder_old_name = _folder
            os.rename(source, dest)
            time.sleep(1)
            print("old name was: {}".format(_folder))
            handler(crop_counter, rotate_counter, dest, recursive_search, opt_crop)
            time.sleep(2)
            # change back
            # _source = os.path.join(dirName, "_")
            _source = source
            os.rename(dest, source)
            time.sleep(2)
            name_restored = os.path.exists(source)
            if name_restored:
                print("{} completed, \tcounter is {}\n".format(_folder, counter))
            else:
                print('\x1b[0;30;41m' + 'folder name restoration failed!\t{}'.format(folder_old_name) + '\x1b[0m')

    ###
    start = timeit.default_timer()

    # Get the list of all files in directory tree at given path
    # listOfFiles = getListOfFiles(dirName, recursive_search)
    listOfFiles = getListOfFiles(dirName, 0)

    num_of_files = len(listOfFiles)
    print('Files found: ' + str(num_of_files))

    # Print the files
    num_passed = 1
    if opt_crop == 1 or opt_rotate == 1:
        for elem in listOfFiles:
            just_dir, fname = os.path.split(elem)

            print(fname)
            print(os.path.dirname(elem))

            # Check if it's a blank image (backside of a scan for example)
            image = cv2.imread(elem)
            valid = 0
            try:
                image.shape
                print("checked for shape".format(image.shape))
                valid = 1
            except AttributeError:
                print("shape not found")
                print("Empty image: " + elem)
                continue
                # code to move to next frame

            # if check_if_blank.check_if_blank(elem) == "image":
            if valid == 1:
                print("not a blank")
                # if opt_skew==1:
                #    straighten_image(elem)
                os.access(elem, os.R_OK)
                if opt_crop == 1:
                    trim_margin.trim_n_print(elem, unsaved_files, crop_counter)
                # if big_dim!=0 and small_dim!=0:
                #   resize_to_ratio(elem,big_dim, small_dim)

                # fix rotation
                if opt_rotate == 1:
                    align_faces.find_rotation(elem)
                    rotate_counter = rotate_counter + 1

            print("Passed :", num_passed, " out of ", num_of_files, "\n")
            num_passed += 1

    stop_crop = timeit.default_timer()
    #
    # if opt_upload == 1:
    #     if opt_album_name != "0":
    #         g_p_album_name = opt_album_name
    #     else:
    #         g_p_album_name = 0
    #
    #     upload_to_google_photos(g_p_album_name, listOfFiles)

    # stop_upload = timeit.default_timer()

    print('Time for local process: ', stop_crop - start)
    # print('Time for upload process: ', stop_upload - (stop_crop - start))
    print('Folder {} completed'.format(dirName))


def main():
    global unsaved_files
    global crop_counter
    global rotate_counter
    crop_counter = 0
    rotate_counter = 0

    unsaved_files = []
    print("This script allows multiple tools to handle scans: \n")
    print("-Trimming blank borders.\n -Adjusting dimensions to ratio. \n -Deskew images.\n -Upload to Google Photos\n")
    dirName = input("Enter root folder full path: ")
    # recursive_search = int(input("Include sub-folders?[0 / 1] "))
    recursive_search = 0
    # opt_skew=int(input("Is aligning needed?[0 / 1] "))
    opt_crop = int(input("Is crop margin needed?[0 / 1] "))
    # big_dim = int(input("Enter ratio of bigger dim (or 0 to decline): "))
    # small_dim = int(input("Enter ratio of smaller dim (or 0 to decline): "))

    global opt_rotate
    opt_rotate = int(input("Is rotate fixing needed?[0 / 1] "))

    # opt_upload = int(input("Is uploading to Google Photos needed?[0 / 1] "))
    # if opt_upload == 1:
    #     opt_album_name = input("Enter album name for Google Photos or 0 to use folder's name as a default: ")
    #     handler(crop_counter, rotate_counter, dirName, recursive_search, opt_crop, opt_upload, opt_album_name)

    # if os.path.isdir(dirName):
    head, _folder = os.path.split(dirName)
    source = dirName
    dest = os.path.join(head, 'temp_name_top')
    print('\x1b[0;33;40m' + 'head= {}\n_folder= {}\n source= {}\ndest= {}'.format(head, _folder, source,
                                                                                  dest) + '\x1b[0m')
    folder_old_name = _folder
    os.rename(source, dest)
    time.sleep(1)
    print("old name was: {}".format(_folder))
    handler(crop_counter, rotate_counter, dest, recursive_search, opt_crop)
    time.sleep(1)
    # change back
    # _source = os.path.join(dirName, "_")
    _source = source
    os.rename(dest, source)
    time.sleep(2)
    name_restored = os.path.exists(source)

    if not name_restored:
        print('\x1b[5;30;41m' + 'folder name restoration failed!\t{}'.format(folder_old_name) + '\x1b[0m')
        input('\x1b[0;30;43m' + 'press to continue' + '\x1b[0m')

    print_unsaved_files()
    print("crop_counter: ", crop_counter)
    print("rotate_counter: ", rotate_counter)
    input("done")


if __name__ == '__main__':
    main()
