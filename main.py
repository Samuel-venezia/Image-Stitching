import pandas as pd
from assets import stitched_approaches
import os
from Dataset_Panorama import stitched_final

if __name__ == "__main__":
    import gc
    import faulthandler
    gc.enable()
    faulthandler.enable()
    # pass
    stage_1 = ["SIFT", "BRISK", "AKAZE", "ORB"]
    # stage_1 = ["BRISK"]
    stage_2 = ["SIFT", "BRISK", "AKAZE", "ORB", "FREAK"]
    stage_3 = ["BF", "BF KNN"]
    stage_4 = ["RANSAC", "USAC"]
    approaches = []

    for s_4 in stage_4:
        for s_3 in stage_3:
            for s_2 in stage_2:
                for s_1 in stage_1:
                    approaches.append(f"{s_1}-{s_2}-{s_3}-{s_4}")

    # Last on: ORB - ORB - BF - RANSAC
    # approaches = approaches[6:12]

    # To be done afterwards
    # approaches = [approaches[34]]
    # approaches = [""]
    # approaches = approaches[33:41]
    # stitched_pipelines = ["AKAZE-SIFT-BF KNN-RANSAC", "AKAZE-FREAK-BF-RANSAC", "AKAZE-BRISK-BF-RANSAC", "AKAZE-SIFT-BF KNN-USAC", "AKAZE-SIFT-BF-USAC", "SIFT-SIFT-BF KNN-USAC"]  #
    # stitched_pipelines = ["AKAZE-FREAK-BF-USAC"]  #
    # stitched_pipelines = list({"SIFT-SIFT-BF-RANSAC", "AKAZE-BRISK-BF-RANSAC", "BRISK-SIFT-BF KNN-RANSAC", "AKAZE-SIFT-BF KNN-RANSAC", "BRISK-BRISK-BF KNN-RANSAC", "AKAZE-BRISK-BF-USAC", "SIFT-SIFT-BF KNN-USAC", "AKAZE-SIFT-BF KNN-USAC"})
    # stitched_pipelines = ["SIFT-SIFT-BF-RANSAC"]
    # stitched_pipelines = ["AKAZE-BRISK-BF-RANSAC"]
    # GH
    # "SIFT-SIFT-BF-RANSAC", "BRISK-SIFT-BF-RANSAC", "BRISK-BRISK-BF-RANSAC", "AKAZE-BRISK-BF-RANSAC",
    #                 "SIFT-SIFT-BF KNN-RANSAC", "AKAZE-SIFT-BF KNN-RANSAC", "BRISK-BRISK-BF KNN-RANSAC",
    #                 "AKAZE-FREAK-BF KNN-RANSAC", "SIFT-SIFT-BF-USAC", "AKAZE-FREAK-BF-USAC", "SIFT-SIFT-BF KNN-USAC",
    #                 "AKAZE-BRISK-BF KNN-USAC", "AKAZE-FREAK-BF KNN-USAC",


    # approaches = ["AKAZE-BRISK-BF KNN-USAC", "AKAZE-FREAK-BF-USAC", "AKAZE-SIFT-BF KNN-USAC", "BRISK-BRISK-BF KNN-USAC", "BRISK-FREAK-BF KNN-USAC", "BRISK-FREAK-BF-USAC", "BRISK-SIFT-BF KNN-USAC", "ORB-SIFT-BF KNN-USAC", "SIFT-SIFT-BF KNN-USAC"]
    # stitched_pipelines = [stitched_pipelines[5]]
    # Do last 3 for halfdome

    dataset_names = ["adobe_panoramas", "isiqa_release/constituent_images"]
    dataset_name = dataset_names[0]
    dir_names = os.listdir(dataset_name)
    # dir_names = dir_names[5:]
    # dir_names = [dir_names[4]]
    # dir_names = [dir_names[0]]
    if dataset_name == dataset_names[1]:
        dir_numbers = []
        for dir_name in dir_names:
            dir_numbers.append(int(dir_name))
        dir_names = dir_numbers
        dir_names.sort()

    # dir_names = [dir_names[0]]
    new_dirs = []
    for dir_name in dir_names:
        if "F_" in dir_name:  # "F_20200220" in dir_name and
            new_dirs.append(dir_name)
    dir_names = new_dirs
    # dir_names = ["fishbowl"]
    for i in range(1, 2):
        durations_scenes = []
        for approach in [stitched_approaches[0]]:
            for dir_ind, dir_name in enumerate(dir_names):
                durations = []
                for pipeline in [stitched_pipelines[0]]:
                    run_stitch = True
                    if dataset_name == dataset_names[0]:
                        output_dir = f"gh_output_stitched/{dataset_name}/{dir_name}"
                        if dir_name == "office":
                            pass
                            dir_reverse = True
                            # run_stitch = False
                            # continue
                        elif dir_name == "diamondhead":
                            dir_reverse = False  # False
                        elif dir_name == "hotel":
                            dir_reverse = False  # False
                        elif dir_name == "rio":
                            dir_reverse = True
                        elif dir_name == "yard":
                            dir_reverse = False
                        elif dir_name == "goldengate":
                            dir_reverse = True
                        elif dir_name == "carmel":
                            dir_reverse = False  # False
                        elif dir_name == "fishbowl":
                            dir_reverse = False
                        elif dir_name == "shanghai":
                            dir_reverse = True
                        elif dir_name == "halfdome":
                            dir_reverse = True
                        elif "F_20200220" in dir_name:
                            dir_reverse = False
                        else:
                            run_stitch = True
                            if dir_ind % 2 == 0:
                                dir_reverse = True  # False
                            else:
                                dir_reverse = False  # True
                    else:
                        output_dir = f"gh_output_stitched/{dataset_name[:5]}/{dir_name}"
                        dir_reverse = False
                        if 17 <= dir_ind <= 19 or dir_ind == 1:
                            run_stitch = False
                            continue
                    if run_stitch:
                        # for width in range(2600, 2800, 25):
                        # dir_reverse = not dir_reverse
                        duration_approach = stitched_final(dir_reverse=dir_reverse, dir_name=f"{dataset_name}/{dir_name}",
                                                           output_dir=output_dir, approach=approach,
                                                           width=1024, pipeline_stages=pipeline)
                        durations.append(duration_approach)
                        # duration approach can be appended into a list,
                        # and used within a dataframe as a comparison with other approaches
                        print(f"Duration of {i}:: {approach}: {pipeline} for {dir_name}: {duration_approach} seconds")
                    else:
                        durations.append("N/A")
                durations_scenes.append(durations)
            duration_df = pd.DataFrame(durations_scenes, index=dir_names, columns=stitched_pipelines) # , columns=["Duration"])
            if dataset_name == dataset_names[1]:
                pass
                # duration_df.to_csv(f"output_stitched/{dataset_name[:5]}/durations_{approach}.csv")
            else:
                pass
                print(duration_df.to_string())
                # duration_df.to_csv(f"durations_{approach}_{i}_APAP.csv")

"""
    Love it:
    Unable to allocate 20.8 PiB for an array with shape (42547875, 183242479, 3) and data type uint8
"""
