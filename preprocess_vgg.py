import os
import sys
import cv2
import argparse
from insightface_func.face_detect_crop_single import Face_detect_crop
from pathlib import Path
from tqdm import tqdm
import numpy as np

def main(args):
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    crop_size = 224

    dirs = os.listdir(args.path_to_dataset)
    dirs.sort()
    for i in tqdm(range(len(dirs))):
        d = os.path.join(args.path_to_dataset, dirs[i])
        dir_to_save = os.path.join(args.save_path, dirs[i])
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
        dir_to_save_kps = os.path.join(dir_to_save, "detections")
        Path(dir_to_save_kps).mkdir(parents=True, exist_ok=True)
        image_names = os.listdir(d)
        for image_name in image_names:
            try:
                image_path = os.path.join(d, image_name)
                image = cv2.imread(image_path)
                cropped_image, _, kpss = app.get_crop_and_kps(image, crop_size)
                #cv2.imwrite(os.path.join(dir_to_save, image_name), cropped_image[0])
                with open(os.path.join(dir_to_save_kps, os.path.splitext(image_name)[0])+".txt", "w") as file:
                    for i in range(len(kpss[0])):
                        file.write(f"{kpss[0][i, 0]:.2f} {kpss[0][i, 1]:.2f}\n")
                #print(dir_to_save_kps)
                #assert False
            except:
                #assert False
                pass
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='./VggFace2/VGG-Face2/data/preprocess_train', type=str)
    parser.add_argument('--save_path', default='./VggFace2-crop', type=str)
    
    args = parser.parse_args()
    
    main(args)
