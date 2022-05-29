import mediapipe as mp
import glob
import math
import PIL
from io import BytesIO
import imageio
import os
import cv2
from PIL.ExifTags import TAGS
import PIL.ImageOps as ImageOps

import numpy as np

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


MAX_W = 1600
MAX_H = 1600

def get_exif_date(fn):
    ret = {}
    i = PIL.Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    return ret['DateTime']


def expand_and_scale(img):
    mask = img.sum(axis=2) == 0 # evil hack... the overlay is done by copying
    img[mask] = [1, 1, 1]       # non black pixels. so let's make sure we have no black pixel in the image
    img = np.asarray(ImageOps.expand(PIL.Image.fromarray(img), border=20, fill=0xffffff)).copy()
    
    h, w, _ = img.shape
    if max(h,w) > 1200:
        scale = 1200. / float(max(h,w))
        w = int(w * scale)
        h = int(h * scale)
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
        #img = skimage.util.img_as_ubyte(rescale(img, 1200. / float(max(h,w)), mode='constant', order=3))
    
    return img

def get_face_location(img, face_detection):
    results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(results)
    for detection in results.detections:
        left_eye = mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.LEFT_EYE) 
        right_eye = mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

        left_x = left_eye.x * img.shape[1]
        left_y = left_eye.y * img.shape[0] 
        right_x = right_eye.x * img.shape[1] 
        right_y = right_eye.y * img.shape[0] 

        distance = math.sqrt(math.pow(left_x - right_x, 2) + math.pow(left_y - right_y, 2))

        return (
            left_x, left_y, 
            right_x, right_y,
            distance)

    return None


with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:

    ref_scale = -1
    ref_eye_x = -1
    ref_eye_y = -1
    ref_ratio = -1

    ref_w = -1
    ref_h = -1

    output_extensions = ["gif", "mp4"]
    video_writers = [
            imageio.get_writer("movie.{}".format(ext), mode='I', fps=4)
            for ext in output_extensions]




    files = glob.glob('./*.JPG') + glob.glob('./*.jpg')
    files.sort(key=get_exif_date)
    print(files)
    curr_image = None
    for count, f in enumerate(files):
        img = cv2.imread(f)
        
        img = expand_and_scale(img) # add a white frame around it, and rescale if it's too big.
        (h, w, _) = img.shape
    
        print('detecting faces') 
        result = get_face_location(img, face_detection)
        if result is None:
            print("no valid face")
            continue
        left_x, left_y, right_x, right_y, eye_distance = result

        if ref_eye_x < 0: # first iteration, we have not found eyes before
            ref_eye_x = float(left_x) / w
            ref_eye_y = float(left_y) / h

        # TODO: this is not nice, but otherwise alignment is off when head is tilted to the side
        eye_distance = 1.0
        if ref_scale < 0:
            ref_scale = eye_distance

                    
        if max(eye_distance, ref_scale) / min(eye_distance, ref_scale) > 2.2:
            print('skipping face')
            continue
        
        scaling = ref_scale / eye_distance # this should be used for rescaling I think!





        deg = math.degrees(math.atan(float(right_y - left_y) / float(right_x - left_x)))
        
        M = np.float32([[1,0,ref_eye_x * MAX_W - left_x],[0,1,ref_eye_y * MAX_H - left_y]])
        img = cv2.warpAffine(img,M,(MAX_W,MAX_H),flags=cv2.INTER_LANCZOS4)

        center = (ref_eye_x * MAX_W, ref_eye_y * MAX_H)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=deg, scale=scaling)
        img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(MAX_W, MAX_H))
        #cv2.imwrite("helper.jpg", img)


        print(img.shape)
        if curr_image is None:
            curr_image = img
        mask = img.sum(axis=2) == 0
        img[mask] = curr_image[mask]
        

        #io.imsave(f+".jpg", img)
        #cv2.imwrite(f+".jpg", img)
        curr_image = img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for writer in video_writers:
            writer.append_data(img)
    #    break


            
    for writer in video_writers:
        writer.close()
