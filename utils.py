import glob
import os.path as osp
import json
import h5py
import numpy as np
import cv2
from cnst import TIMESCALE
import matplotlib.pyplot as plt

class EventBuffer:
    def __init__(self, ev_f) -> None:
        self.f = h5py.File(ev_f, "r")
        self.x_f = self.f["x"]
        self.y_f = self.f["y"]
        self.p_f = self.f["p"]
        self.t_f = self.f["t"]

        self.fs = [self.x_f, self.y_f, self.p_f, self.t_f]

        self.n_retrieve = 100000
        self.x_cache = np.array([self.x_f[0]])
        self.y_cache = np.array([self.y_f[0]])
        self.t_cache = np.array([self.t_f[0]])
        self.p_cache = np.array([self.p_f[0]])

        self.caches = [self.x_cache, self.y_cache, self.t_cache, self.p_cache]

        self.curr_pnter = 1

    def update_cache(self):

        rx, ry, rp, rt = [e[self.curr_pnter:self.curr_pnter + self.n_retrieve] for e in self.fs]
        self.x_cache = np.concatenate([self.x_cache, rx])
        self.y_cache = np.concatenate([self.y_cache, ry])
        self.p_cache = np.concatenate([self.p_cache, rp])
        self.t_cache = np.concatenate([self.t_cache, rt])

        self.curr_pnter = min(len(self.t_f), self.curr_pnter + self.n_retrieve)

    def drop_cache_by_cond(self, cond):
        self.x_cache = self.x_cache[cond]
        self.y_cache = self.y_cache[cond]
        self.p_cache = self.p_cache[cond]
        self.t_cache = self.t_cache[cond]

    def retrieve_data(self, st_t, end_t):
        while self.t_cache[-1] <= end_t and (self.curr_pnter < len(self.t_f)):
            self.update_cache()

        ret_cond = ( st_t<= self.t_cache) & (self.t_cache <= end_t)
        ret_data = [self.t_cache[ret_cond], self.x_cache[ret_cond], self.y_cache[ret_cond], self.p_cache[ret_cond]]
        self.drop_cache_by_cond(~ret_cond)

        return ret_data  # (ts, xs, ys, ps)


def gen_triggers(n_frames):
    return np.array([i/n_frames*1e7 for i in range(n_frames)])

def evs_to_img(evs, win_size):
    h, w = win_size[:2]
    ev_img = np.zeros((h,w,3))
    ev_cnts = np.zeros((h, w))

    t, x, y, p = evs


    np.add.at(ev_cnts, (y, x), p)
    ev_img[ev_cnts > 0, 2] = 255
    ev_img[ev_cnts < 0, 0] = 255

    return ev_img, ev_cnts

def to_gray(image):
    if len(image.shape) == 3 and image.shape[-1] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    return gray_image


def load_jsons(fs):
        data = []
        for f in fs:
            with open(f, "r") as f_tmp:
                data.append(json.load(f_tmp))
        
        return data

def load_json(json_f):
    with open(json_f, "r") as f:
        return json.load(f)

def sobel_edge_detection(image, grad_thresh = None):
    # Convert the image to grayscale

    gray_image = to_gray(image)
    
    # Apply Sobel edge detection in both x and y directions
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, 3, 1)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, 3, 1)
    
    # Calculate the magnitude of the gradients
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize the gradient magnitude to the range [0, 255]
    normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, -1).astype(np.uint8)
    
    thresh = normalized_gradient.mean()
    thresh += (1 - thresh)/3
    used_thresh, thresholded = cv2.threshold(normalized_gradient, thresh, 255, cv2.THRESH_BINARY)

    return normalized_gradient, thresholded, used_thresh

def canny_detect_edges(gray_image, min_threshold=50, max_threshold=200):
    edges = cv2.Canny(gray_image, min_threshold, max_threshold)
    return edges

def make_edge_overlay(img1, img2, edge_method=canny_detect_edges):
        rgb_gray, pred_gray = to_gray(img1).squeeze(), to_gray(img2).squeeze()
        rgb_gray, pred_gray = np.clip(rgb_gray*255, 0, 255).astype(np.uint8), np.clip(pred_gray*255, 0, 255).astype(np.uint8)
        # gt_e, pred_e = detect_edges(rgb_gray), detect_edges(pred_gray)
        gt_e, pred_e = edge_method(rgb_gray), edge_method(pred_gray)
        gt_cond = (gt_e != 0)
        pred_cond = (pred_e != 0)
        cond = gt_cond | pred_cond

        overlay = np.ones((*rgb_gray.shape[:2], 3), dtype=np.uint8)*255
        overlay[cond] = 0

        overlay[gt_cond, 0] = 255
        overlay[pred_cond, 2] = 255
        return overlay

def make_overlay(img1, img2):
    img1, img2 = to_gray(img1), to_gray(img2)
    overlay_img = np.zeros((*img1.shape[:2],3))
    overlay_img[...,0] = img1
    overlay_img[...,2] = img2
    return overlay_img


def eimg_to_img(eimg, is_abs=True):
    img = np.zeros((*eimg.shape[:2],3))
    
    pos_cond = eimg > 0
    neg_cond = eimg < 0
    if is_abs:
        img[pos_cond, 2] = 255
        img[neg_cond, 0] = 255
    else:
        img[pos_cond, 2] = eimg[pos_cond]
        img[neg_cond, 0] = np.abs(eimg[neg_cond])
    
    return img


def create_video_from_images(image_list, output_video, frame_rate=30):
    # Get the dimensions of the first image (assuming all images have the same dimensions)
    height, width, layers = image_list[0].shape

    # Create a VideoWriter object to write the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Write each image to the video
    for img in image_list:
        out.write(img)

    # Release the VideoWriter
    out.release()

    print("Video saved as:", output_video)


def write_text_on_img(image, text):
    position = (10, 200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 10
    font_color = (255, 255, 0)  # White color in BGR
    line_thickness = 2

    # Put the text on the image
    cv2.putText(image, text, position, font, font_scale, font_color, line_thickness)
    return image



def make_vid(img_dir):
    ev_edge = cv2.imread(osp.join(img_dir, "ev_edge.png"))
    latent_img_fs = sorted(glob.glob(osp.join(img_dir, "latent_edges*.png")))
    phi_edges_img_fs = sorted(glob.glob(osp.join(img_dir, "phi_edge_img*.png")))
    get_c = lambda x:x.split("_")[-1].split(".png")[0]

    imgs = []
    
    for i, (latent_img_f, phi_edge_img_f) in enumerate(zip(latent_img_fs, phi_edges_img_fs)):

        img = np.concatenate([ev_edge, cv2.imread(latent_img_f), cv2.imread(phi_edge_img_f)], axis=1)
        img = write_text_on_img(img, get_c(latent_img_f))
        imgs.append(img)
    
    create_video_from_images(imgs, "out.mp4", frame_rate=15)