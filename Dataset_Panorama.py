import sys
from assets import stitched_approaches, margin_crop
import cv2 as cv
from pathlib import Path
import os
from Feature_Extraction import Pipeline
from Panorama import Stitch, superpoint_superglue_homography
import imutils
import gc
import time
import time
from itertools import count
from multiprocessing import Process
import stopit


def stitched_final(dir_reverse=False, dir_name="isiqa_release/constituent_images/12", output_dir="output_stitched/isqa/12", approach=stitched_approaches[0], pipeline_stages="AKAZE-FREAK-BF-USAC", width=1024, iter=0):
    try:
        gc.enable()
        # if Path.is_file(Path(f"{output_dir}/{approach}-{pipeline_stages}-{width}.png")):
        #     print("Image Already Exists")
        #     return "N/A"
        img_names, h_names = dir_img_names(dir_name, output_dir)
        imgs = []
        cut_off = 4
        if len(img_names) > cut_off:
            img_names = img_names[:cut_off]
        min_height = sys.maxsize
        if dir_reverse:
            img_names.reverse()
        for img_name in img_names:
            img = cv.imread(f"{dir_name}/{img_name}")
            if "isiqa" in dir_name:
                pass
                # img = imutils.resize(img, width=int(width))
            imgs.append(img)
            min_height = min(min_height, img.shape[0])
        imgs_formatted = []
        for img in imgs:
            imgs_formatted.append(img[:min_height-1,:,:])
        imgs, img = None, None
        stitched_img, duration = rec_stitched(imgs_formatted, approach, pipeline_stages=pipeline_stages)
        if stitched_img.shape[0] == imgs_formatted[0].shape[0] - margin_crop and stitched_img.shape[1] == imgs_formatted[0].shape[1] - margin_crop:
            raise Exception
        # cv.imwrite(f"{output_dir}/{approach}-{pipeline_stages}-{width}.png", stitched_img)
        folder_chosen = output_dir.split("/")[-1]
        cv.imwrite(f"{folder_chosen}-Sports-{pipeline_stages}.png", stitched_img)
        return duration*0.000001
    except MemoryError as e:
        if iter == 1:
            return "N/A"
        print(e)
        print("Sleeping for 30 seconds...")
        time.sleep(30)
        print("Continuing process...")
        return stitched_final(dir_reverse, dir_name, output_dir, approach, pipeline_stages, width, iter=iter+1)
    except Exception as e:
        print(e)
        return "N/A"


def rec_stitched(imgs, approach, pipeline_stages="", durations=0):
    # returns a single numpy array
    stitched_imgs_formatted = []
    # It will only crop an image, at the last layer of the pyramid.
    if len(imgs) == 2:
        rectangled = True
    else:
        rectangled = True
    # rectangled = False
    # with ThreadPoolExecutor() as executor:
    #     future = executor.submit(stitch_par, imgs, rectangled, minsize)
    #     stitch_par_values = future.result()
    # for _ in range(os.cpu_count()-1):
        # threading.Thread(target=stitch_par, args=(imgs,(len(imgs)-2), rectangled, minsize)).start()
    with stopit.ThreadingTimeout(200) as to_ctx_mgr:
        assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
        stitch_imgs, durations, minsize = stitch_par(imgs, rectangled, durations, approach, pipeline_stages=pipeline_stages)

    if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
        print("Timed Out")
        raise Exception

    for stitched_img in stitch_imgs:
        f_stitched_img = imutils.resize(stitched_img, width=minsize[1])
        stitched_imgs_formatted.append(f_stitched_img)
        minsize[0] = min(minsize[0], f_stitched_img.shape[0])
    stitch_imgs = []
    for stitched_img in stitched_imgs_formatted:
        stitch_imgs.append(stitched_img[:minsize[0]-1, :minsize[1]-1,:])

    if len(stitch_imgs) == 1:
        return stitch_imgs[0], durations
    else:
        gc.collect()
        return rec_stitched(stitch_imgs, approach, durations=durations, pipeline_stages=pipeline_stages)


def stitch_par(imgs, rectangled, durations, approach, pipeline_stages):
    minsize = [sys.maxsize, sys.maxsize]
    stitch_imgs = []
    for img_index in range(len(imgs)-1):
        # status, stitched_img, duration, homography/mdlt
        """if len(imgs)-1 == 3 and img_index + 1 == 1:
            matching_vis = True
            rectangled = True
        else:
            matching_vis = False"""

        status, stitched, duration, _ = all_stitched_approaches(imgs[img_index], imgs[img_index+1], approach,
                                                                rectangled=rectangled, pipeline_stages=pipeline_stages, matching=False)
        if status == cv.STITCHER_OK:
            print(f"Stitched Round {len(imgs)-1}: {img_index + 1}")
            # cv.imwrite(f"figures/Example_2_SR-{len(imgs)-1}-{img_index + 1}.png", stitched)
            # y, x
            minsize = [min(minsize[0], stitched.shape[0]), min(minsize[1], stitched.shape[1])]
            stitch_imgs.append(stitched)
            durations += duration.microseconds
        else:
            print("Did not stitch")
            raise Exception
    return stitch_imgs, durations, minsize


def dir_img_names(dir_name, output_dir=None):
    # create the directory, if it does not exist
    if output_dir is not None:
        if not Path(output_dir).is_dir():
            os.mkdir(output_dir)
            print(f"Directory Created: {output_dir}")
        else:
            print(f"{output_dir}")
    names = os.listdir(dir_name)
    img_names = []
    h_names = []

    for img_name in names:
        if ".png" in img_name or ".jpg" in img_name or ".ppm" in img_name:
            img_names.append(img_name)
        elif "H" in img_name:
            h_names.append(f"{dir_name}/{img_name}")
    return img_names, h_names


def all_stitched_approaches(img_1, img_2, approach, pipeline_stages, rectangled=False, matching=False):
    h_mdlt = None
    if approach == stitched_approaches[0]:
        status, stitched, duration, h_mdlt = pipeline_chosen(img_1, img_2, rectangled=rectangled, pipeline_stages=pipeline_stages, matching=matching)
    elif approach == stitched_approaches[1]:
        status, stitched, duration, h_mdlt = apap(img_1, img_2, rectangled=rectangled)
    elif approach == stitched_approaches[2]:
        status, stitched, duration, h_mdlt = apap(img_1, img_2, rectangled=rectangled, dl=True)
    elif approach == stitched_approaches[3]:
        status, stitched, duration, h_mdlt = auto(img_1, img_2, rectangled=rectangled)
    elif approach == stitched_approaches[4]:
        status, stitched, duration, h_mdlt = superpoint_superglue_homography(img_1, img_2, rectangled=rectangled)
    else:
        status, stitched, duration = open_stitched(img_1, img_2, rectangled=rectangled)
    return status, stitched, duration, h_mdlt


# region stitching algorithms
def pipeline_chosen(img_1, img_2, rectangled=False, pipeline_stages="AKAZE-SIFT-BF KNN-USAC", matching=None):
    pipeline = Pipeline(img_1, img_2)
    try:
        stages_split = pipeline_stages.split("-")
        # AKAZE, FREAK, BF, USAC - works for adobe panoramas
        pipeline.stage_1(f"{stages_split[0]}")
        pipeline.stage_2(f"{stages_split[1]}")
        pipeline.stage_3(f"{stages_split[2]}", matching_vis=matching)
        pipeline.warped_stitched_images(f"{stages_split[3]}", apap_poss=False)
    except LookupError as e:
        print(e)
        return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", None
    if rectangled:
        try:
            stitched_img = pipeline.get_rectangled_image()
            return cv.STITCHER_OK, stitched_img, pipeline.get_duration(), pipeline.get_homography()
        except Exception as e:
            print(e)
            return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", None
    stitched_img = pipeline.get_stitched_image()
    return cv.STITCHER_OK, stitched_img, pipeline.get_duration(), pipeline.get_homography()


def apap(img_1, img_2, rectangled=True, dl=False):
    stitcher = Stitch([img_2, img_1])
    try:
        if dl:
            status, apap_stitched, duration_apap, h_duration_apap, mdlt = stitcher.apap_deep_learning_stitching(rectangled=rectangled)
        else:
            status, apap_stitched, duration_apap, h_duration_apap, mdlt = stitcher.apap_stitching(rectangled=rectangled)
        return status, apap_stitched, h_duration_apap, mdlt
    except cv.error as e:
        print("Unable to stitch image using APAP")
        print(e)
        return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", None


def auto(img_1, img_2, rectangled=True):
    stitcher = Stitch([img_2, img_1])
    try:
        status, auto_stitched, duration, h_duration, homography = stitcher.auto_stitching(rectangled=rectangled)
        return status, auto_stitched, h_duration, homography
    except cv.error as e:
        print("Unable to stitch image using Auto")
        print(e)
        return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", None


def open_stitched(img_1, img_2, rectangled=True):
    stitcher = Stitch([img_1, img_2])
    try:
        status, open_stitch, duration = stitcher.open_stitching(rectangled=rectangled)
        return status, open_stitch, duration
    except Exception as e:
        print("Unable to stitch image using Open Stitching")
        print(e)
    finally:
        return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A"
# endregion
