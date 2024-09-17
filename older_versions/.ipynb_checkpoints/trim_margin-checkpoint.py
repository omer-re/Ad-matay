import os
import traceback

from PIL import Image, ImageChops


def is_ratio_valid_coordinates(dim1, dim2):
    # find smaller dim and compress bigger to it by proper ratio
    # print(float(dim1 / dim2), float(dim2 / dim1))

    # for square images
    if float(dim2 / dim1) < 0.53 or float(dim2 / dim1) > 0.47:
        print("SQUARE IMAGE")


    elif (dim1 < dim2):
        # print(float(dim1 / dim2))
        if float(dim1 / dim2) < 0.65 or float(dim1 / dim2) > 0.8:
            # might be invalid cropping
            # print("false1")
            return False
    elif (dim1 > dim2):
        # print(float(dim2 / dim1))

        if float(dim2 / dim1) < 0.65 or float(dim2 / dim1) > 0.8:
            # might be invalid cropping
            # print("false2")
            return False

    # print("128=True")
    return True


# make sure it is big enough to be cropped, and cropping if it does

def trim(im, entry_path):
    if entry_path is None:
        return
    im = Image.open(entry_path)
    try:
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        dim1 = bbox[2] - bbox[0]
        dim2 = bbox[3] - bbox[1]
        MIN_DIM = 2000
        if bbox is not None:
            # print("bbox")
            # print(dim1 , dim2 ,is_ratio_valid_coordinates(dim1, dim2) )
            if dim1 > MIN_DIM and dim2 > MIN_DIM and is_ratio_valid_coordinates(dim1, dim2):
                res = im.crop(bbox)
                res.save(entry_path)
                # print("cropped")
                return entry_path
            else:
                print("no crop - dim problem: ", dim2, dim1, "ratio: ", float(dim2 / dim1))
            return
        else:
            print("no crop bbox problem")
            return
    # except:
    #    print("An exception occurred ")
    except Exception as exception:
        traceback.print_exc()
        # print("Exception: {}".format(type(exception).__name__))
        # print("Exception message: {}".format(exception))
        # print("161")
        return


def trim_n_print(entry_path, unsaved_files=None, crop_counter=0):
    just_dir, fname = os.path.split(entry_path)
    image = Image.open(entry_path)
    res = trim(trim(image, entry_path), entry_path)
    if res != 0:

        crop_counter = crop_counter + 1
        unsaved_files.append(entry_path)
    else:
        print("Don't save")
