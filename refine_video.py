# Use idu_refine to refine the video

import os
import cv2
import argparse
import idu_refine


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--video_path", type=str, required=True)

    args = arg_parser.parse_args()

    video_name = os.path.basename(args.video_path).split(".")[0]
    output_path = os.path.join("outputs", video_name)

    flow_edit_refine = idu_refine.FlowEditRefineIDU(output_path)

    cap = cv2.VideoCapture(args.video_path)

    print("Reading video...")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    print("Done reading video.")
    frames = frames[::10]  # Only process every 10 frames
    refine_imgs = flow_edit_refine.run(
        imgs=frames,
        src_prompt="Street view image of a city street with building facades, sidewalks, street furniture, and parked vehicles. The image has distortions, with blurring and warping artifacts visible on building edges and street objects.",
        tar_prompt="Clear street view image of a city street with sharp building facades, smooth sidewalks, and crisp street furniture and parked vehicles. The image should be free of distortions, with natural lighting and well-defined textures, creating a realistic street scene.",
        n_min=10,
        n_max=24,
    )