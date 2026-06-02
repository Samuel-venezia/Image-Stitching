import cv2 as cv
import numpy as np
# from stitching import Stitcher
from AutoStitch import Panaroma
import datetime
from Deep_Homography import superpoint_superglue
from Feature_Extraction import HomographyEstimation
from assets import rectangling, crop_img
# For APAP
import APAP.matchers
from APAP.ransac import RANSAC
from APAP.apap import APAP_stitching, get_mdlt_final_size
from APAP.imagewarping import imagewarping
from APAP.config import *
# from APAP.homography import homography_fit, get_hom_final_size


# Creating a panoramic view from multiple images
def get_images(dataset="M", no_images=7):
    # list of images
    pan_images = []
    raw_stitching_folder = "stitching_img/"
    for index in range(1, no_images + 1):
        pan_images.append(cv.imread(f"{raw_stitching_folder}{dataset}{index}.jpg"))
    return pan_images


def superpoint_superglue_homography(img_1, img_2, stage_4_algorithm="RANSAC", rectangled=False):
    _, _, matches, duration = superpoint_superglue(img_1, img_2)
    homography_est = HomographyEstimation([img_1, img_2], None, matches)
    homography_est.dl_orig_dest_mtx()
    timer_start = datetime.datetime.now()

    if stage_4_algorithm == "USAC":
        homography_est.usac()
    elif stage_4_algorithm == "MAGSAC":
        homography_est.magsac()
    elif stage_4_algorithm == "PROSAC":
        homography_est.prosac()
    else:
        homography_est.ransac()
    h_duration = (datetime.datetime.now() - timer_start) + duration
    # timer_start = datetime.datetime.now()
    stitched, _ = homography_est.perspective_warping()  # Second value is warped
    # h_duration = datetime.datetime.now() - timer_start
    if rectangled:
        stitched = rectangling(stitched)
    else:
        stitched = crop_img(stitched)
    if stitched.shape[0] == 0 or stitched.shape[1] == 0:
        return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", None, None
    return cv.STITCHER_OK, stitched, h_duration, homography_est.get_homography_matrix()


# Stitching algorithm
class Stitch:
    def __init__(self, images=None):
        if images is None:
            self.__images = []
        self.__images = images

    def set_images(self, images):
        self.__images = images

    def open_cv_stitching(self):
        # Using the image stitcher from Open CV
        timer_start = datetime.datetime.now()
        open_cv_stitcher = cv.Stitcher()
        status, result = open_cv_stitcher.create(mode=cv.STITCHER_PANORAMA).stitch(self.__images)
        timer_end = datetime.datetime.now()
        duration = timer_end - timer_start
        return status, result, duration

    def open_stitching(self, detector="orb", rectangled=True):
        # Implement the pre-existing stitcher from Open Stitching.
        # This will be an alternative stitcher to the pipeline already used and an alternative to Opencv's stitcher
        # try:
        timer_start = datetime.datetime.now()
        # pan = Stitcher(detector=detector, crop=False, finder="no", blend_strength=0)  # , confidence_threshold=0.4705882221464)  # When Crop is True 0.43103449 warper_type="plane",
        # panorama = pan.stitch(self.__images)
        panorama = None
        if rectangled:
            panorama = rectangling(panorama)
        else:
            panorama = crop_img(panorama)
        timer_end = datetime.datetime.now()
        return cv.STITCHER_OK, panorama, timer_end - timer_start

    def auto_stitching(self, rectangled=False):
        # Implementation of the Autostitch algorithm from the 2007 paper,
        # “Automatic panoramic image stitching using invariant features“ Brown and Lowe
        try:
            timer_start = datetime.datetime.now()
            pan = Panaroma()
            stitched, h_duration, homography = pan.image_stitch(self.__images, rectangled=rectangled)
            timer_end = datetime.datetime.now()
            duration = timer_end - timer_start
        except cv.error as e:
            print(e)
            return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", "N/A", None
        return cv.STITCHER_OK, stitched, duration, h_duration, homography

    def apap_stitching(self, rectangled=False):
        # This is the implementation of the As Projective as Possible image stitching approach. 2013
        try:
            timer_start = datetime.datetime.now()
            stitched, mdlt_duration, mdlt = self.__apap(rectangled=rectangled)
            timer_end = datetime.datetime.now()
            duration = timer_end - timer_start
        except cv.error as e:
            print(e)
            return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", "N/A", None
        return cv.STITCHER_OK, stitched, duration, mdlt_duration, mdlt

    def apap_deep_learning_stitching(self, rectangled=False):
        # This is the implementation of the As Projective as Possible image stitching approach.
        # Furthermore, this is coupled with the attachment of the Deep Learning algorithms for SuperGlue and SuperPoint.
        try:
            timer_start = datetime.datetime.now()
            stitched, mdlt_duration, mdlt = self.__apap_deep_learning(rectangled=rectangled)
            timer_end = datetime.datetime.now()
            duration = timer_end - timer_start
        except cv.error as e:
            print(e)
            return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", "N/A", None
        except IndexError as e:
            print(e)
            return cv.Stitcher_ERR_NEED_MORE_IMGS, None, "N/A", "N/A", None
        return cv.STITCHER_OK, stitched, duration, mdlt_duration, mdlt

    def __apap(self, rectangled=False):
        matcher_obj = APAP.matchers.matchers()
        kp1, ds1 = matcher_obj.getFeatures(self.__images[0])
        kp2, ds2 = matcher_obj.getFeatures(self.__images[1])
        matches = matcher_obj.match(ds1, ds2)
        src_orig = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_orig = np.float32([kp2[m.trainIdx].pt for m in matches])
        src_orig = np.vstack((src_orig.T, np.ones((1, len(matches)))))
        dst_orig = np.vstack((dst_orig.T, np.ones((1, len(matches)))))
        ransac = RANSAC(M, thr, visual=False)
        src_fine, dst_fine = ransac(self.__images[0], self.__images[1], src_orig, dst_orig)
        # Generating mesh for MDLT
        X, Y = np.meshgrid(np.linspace(0, self.__images[0].shape[1] - 1, C1),
                           np.linspace(0, self.__images[0].shape[0] - 1, C2))
        # Mesh (cells) vertices' coordinates
        Mv = np.array([X.ravel(), Y.ravel()]).T
        # Perform Moving DLT
        apap = APAP_stitching(gamma, sigma)
        Hmdlt = apap(dst_fine, src_fine, Mv)
        timer_start = datetime.datetime.now()
        # Hmdlt - The homography moving direct linear transformation calculated from the source and destination points.
        min_x, max_x, min_y, max_y = get_mdlt_final_size(self.__images[0], self.__images[1], Hmdlt, C1, C2)
        warped_img1, warped_img2, warped_mask1, warped_mask2 = imagewarping(self.__images[0], self.__images[1], Hmdlt,
                                                                            min_x, max_x, min_y, max_y, C1, C2)
        translate = [-round(min_x), -round(min_y)]
        h2, w2 = self.__images[1].shape[:2]
        warped_2_output_canvas = np.zeros_like(warped_img2)
        warped_1_output_canvas = np.zeros_like(warped_img1)
        for c in range(warped_img2.shape[2]):
            warped_2_output_canvas[:, :, c] = (warped_img2[:, :, c] * warped_mask2)
            warped_1_output_canvas[:, :, c] = (warped_img1[:, :, c] * warped_mask1)
        warped_2_output_canvas[translate[1]:h2 + translate[1], translate[0]:w2 + translate[0]] = \
            warped_1_output_canvas[translate[1]:h2 + translate[1], translate[0]:w2 + translate[0]]
        timer_end = datetime.datetime.now()
        if not rectangled:
            output_cropped = crop_img(warped_2_output_canvas)
        else:
            output_cropped = rectangling(warped_2_output_canvas)
        return output_cropped, (timer_end - timer_start), Hmdlt

    def __apap_deep_learning(self, rectangled):

        _, _, matches = superpoint_superglue(self.__images[0], self.__images[1])
        src_orig = []
        dst_orig = []
        for m in matches:  # (kpt0, kpt1)
            src_orig.append(m[0])
            dst_orig.append(m[1])
        src_orig = np.array(src_orig)
        dst_orig = np.array(dst_orig)

        src_orig = np.vstack((src_orig.T, np.ones((1, src_orig.shape[0]))))
        dst_orig = np.vstack((dst_orig.T, np.ones((1, dst_orig.shape[0]))))
        ransac = RANSAC(M, thr, visual=False)
        src_fine, dst_fine = ransac(self.__images[0], self.__images[1], src_orig, dst_orig)
        # Generating mesh for MDLT
        X, Y = np.meshgrid(np.linspace(0, self.__images[0].shape[1] - 1, C1),
                           np.linspace(0, self.__images[0].shape[0] - 1, C2))
        # Mesh (cells) vertices' coordinates
        Mv = np.array([X.ravel(), Y.ravel()]).T
        # Perform Moving DLT
        apap = APAP_stitching(gamma, sigma)
        Hmdlt = apap(dst_fine, src_fine, Mv)
        # Hmdlt - The homography moving direct linear transformation calculated from the source and destination points.
        timer_start = datetime.datetime.now()
        min_x, max_x, min_y, max_y = get_mdlt_final_size(self.__images[0], self.__images[1], Hmdlt, C1, C2)
        warped_img1, warped_img2, warped_mask1, warped_mask2 = imagewarping(self.__images[0], self.__images[1], Hmdlt,
                                                                            min_x, max_x, min_y, max_y, C1, C2)
        translate = [-round(min_x), -round(min_y)]
        h2, w2 = self.__images[1].shape[:2]
        warped_2_output_canvas = np.zeros_like(warped_img2)
        warped_1_output_canvas = np.zeros_like(warped_img1)
        for c in range(warped_img2.shape[2]):
            warped_2_output_canvas[:, :, c] = (warped_img2[:, :, c] * warped_mask2)
            warped_1_output_canvas[:, :, c] = (warped_img1[:, :, c] * warped_mask1)
        warped_2_output_canvas[translate[1]:h2 + translate[1], translate[0]:w2 + translate[0]] = \
            warped_1_output_canvas[translate[1]:h2 + translate[1], translate[0]:w2 + translate[0]]
        timer_end = datetime.datetime.now()
        if not rectangled:
            output_cropped = crop_img(warped_2_output_canvas)
        else:
            output_cropped = rectangling(warped_2_output_canvas)
        return output_cropped, (timer_end - timer_start), Hmdlt
