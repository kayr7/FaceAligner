import dlib
from skimage import io
from skimage import transform
import glob
import math
import PIL
from io import BytesIO
from skimage.transform import warp, rescale, resize
from skimage.transform import SimilarityTransform, AffineTransform
from skimage import transform as tf
from skimage import util
import skimage
import imageio
import os
import cv2
from PIL.ExifTags import TAGS
import PIL.ImageOps as ImageOps

import numpy as np

MAX_W = 2000
MAX_H = 2000

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
        img = skimage.util.img_as_ubyte(rescale(img, 1200. / float(max(h,w)), mode='constant', order=3))
    return img

def get_face_location(img):
    dets = cnn_face_detector(img, 1)
    if len(dets) != 1:
        print("problem with f: {} - found {} faces".format(f, len(dets)))
        return None
    
    return dets

def get_eye_locations(img, face):
    shape = predictor(img, face.rect)
    right_x = (shape.part(0).x + shape.part(1).x) / 2.
    right_y = (shape.part(0).y + shape.part(1).y) / 2.
    left_x = (shape.part(2).x + shape.part(3).x) / 2.
    left_y = (shape.part(2).y + shape.part(3).y) / 2.
    eye_distance = math.sqrt(
                        math.pow(left_x
                                 - right_x, 2) 
                        + math.pow(left_y 
                                   - right_y, 2))
    return (right_x, right_y, left_x, left_y, eye_distance)


cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

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
curr_image = None
for count, f in enumerate(files):
    img = io.imread(f)
    
    expand_and_scale(img) # add a white frame around it, and rescale if it's too big.
    (h, w, _) = img.shape
  
    print('detecting faces') 
    dets = get_face_location(img)
    if dets is None:
        continue

    print('get eye locations')
    assert(len(dets) == 1)
    right_x, right_y, left_x, left_y, eye_distance = get_eye_locations(img, dets[0])

    
    if ref_eye_x < 0: # first iteration, we have not found eyes before
        ref_eye_x = float(left_x) / w
        ref_eye_y = float(left_y) / h


    if ref_scale < 0:
        ref_scale = eye_distance
        
    if max(eye_distance, ref_scale) / min(eye_distance, ref_scale) > 2.2:
        print('skipping face')
        continue
    
    scaling = ref_scale / eye_distance # this should be used for rescaling I think!

    deg = math.atan(float(right_y - left_y) / float(right_x - left_x))
    tf_rotate = transform.SimilarityTransform(rotation=-deg)
    tf_shift = transform.SimilarityTransform(translation=[-(left_x), -(left_y)])
    tf_shift_inv = transform.SimilarityTransform(translation=[(ref_eye_x * 2880), (ref_eye_y * 2880)])
    img = transform.warp(img, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, output_shape=(2880, 2880), order=3)

    img = skimage.util.img_as_ubyte(rescale(img, 1./4., mode='constant', order=3))


    print(img.shape)
    if curr_image is None:
        curr_image = img
    mask = img.sum(axis=2) == 0
    img[mask] = curr_image[mask]
    

    io.imsave(f+".jpg", img)

    curr_image = img

    for writer in video_writers:
        writer.append_data(img)
#    break


        
for writer in video_writers:
    writer.close()
