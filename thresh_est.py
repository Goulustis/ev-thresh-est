import os.path as osp
import numpy as np
import cv2
from utils import gen_triggers, EventBuffer
import glob
import os.path as osp
import json
from utils import load_json, sobel_edge_detection, make_vid
import matplotlib.pyplot as plt
import pickle
from cnst import TIMESCALE, SAVE_DIR
from private_cnst import SRC_DATA_DIR, FORMATTED_DATA_DIR
from tqdm import tqdm
import jax

class Dataset:
    def __init__(self) -> None:
        self.src_data_dir = SRC_DATA_DIR
        self.formatted_data_dir = FORMATTED_DATA_DIR

        self.src_linear_dir = osp.join(self.src_data_dir, "fine_frames", "linear")
        self.ecam_dir = osp.join(self.formatted_data_dir, "ecam_set")
        self.colcam_dir = osp.join(self.formatted_data_dir, "clear_linear_colcam_set")

        self.data_img_fs = sorted(glob.glob(osp.join(self.colcam_dir, "rgb", "1x", "*.png")))
        self.eimgs = np.load(osp.join(self.ecam_dir, "eimgs", "eimgs_1x.npy"), "r")
        self.event_buffer = EventBuffer(osp.join(self.src_data_dir, "events.hdf5"))

        self._build_img_ev_pairs()
        self.ev_img_trigs = self.get_ev_frame_triggers()


    def _build_img_ev_pairs(self):
        colcam_meta_f = osp.join(self.colcam_dir, "metadata.json")
        metadata = load_json(colcam_meta_f)

        self.img_to_ev_dic = {}
        for k, v in metadata.items():
            self.img_to_ev_dic[int(k)] = v["warp_id"]
        

    def get_src_img_triggers(self):
        return gen_triggers(len(glob.glob(osp.join(self.src_linear_dir, "*.png"))))

    def get_ev_frame_triggers(self):
        cam_dir = osp.join(self.ecam_dir, "camera")
        cam_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))

        ts = []
        for cam_f in cam_fs:
            with open(cam_f, "r") as f:
                data = json.load(f)
                ts.append(data["t"])
        
        return ts
    
    def get_formatted_img(self, idx):
        return cv2.imread(self.data_img_fs[idx], cv2.IMREAD_GRAYSCALE)
    
    def get_data_img_ev_pair(self, img_idx):
        ev_idx = self.img_to_ev_dic[img_idx]
        eimgs = self.eimgs[ev_idx:ev_idx + 1].squeeze()
        img = cv2.imread(self.data_img_fs[img_idx], cv2.IMREAD_GRAYSCALE)

        eimg_st, eimg_end = self.ev_img_trigs[ev_idx], self.ev_img_trigs[ev_idx+1]
        return img, eimgs, (eimg_st, eimg_end)

    def get_img_size(self):
        return self.get_formatted_img(0).shape[:2]

def make_ev_boundary_img(evs, t_c, img_size):
    ts, xs, ys, ps = evs
    t_c, ts = t_c/TIMESCALE, ts/TIMESCALE

    weights = np.exp(-np.abs(ts - t_c))*ps
    
    M = np.zeros(img_size)
    np.add.at(M, (ys, xs), weights)
    return M


def calc_phi_tv_edge(latent_img, ev_b_img, c_thresh=None, save=True):
    ev_grad, ev_edges, thresh = sobel_edge_detection(ev_b_img)
    latent_grad, latent_edges, _ = sobel_edge_detection(latent_img, thresh)
    # latent_grad, latent_edges, _ = sobel_edge_detection(latent_img)

    phi_edge_tmp = latent_edges.astype(np.float64) * ev_edges.astype(np.float64)
    if save:
        c_save = np.round(c_thresh, 4)
        cv2.imwrite(osp.join(SAVE_DIR ,f"latent_edges_{c_save}.png"), latent_edges)
        cv2.imwrite(osp.join(SAVE_DIR, "ev_edge.png"), ev_edges)
        cv2.imwrite(osp.join(SAVE_DIR, f"phi_edge_img_{c_save}.png"), (255*phi_edge_tmp/phi_edge_tmp.max()).astype(np.uint8))
    
    
    phi_tv = np.abs(latent_grad).sum()*0.15
    # latent_edges, ev_edges = latent_edges.astype(np.float64), ev_edges.astype(np.float64)
    phi_edge = -(phi_edge_tmp).sum()#/(np.linalg.norm(latent_edges.reshape(-1))*np.linalg.norm(ev_edges.reshape(-1)))
    return {"phi_tv": phi_tv ,
            "phi_edge":phi_edge,
            "phi":phi_tv + phi_edge}


def save_ev_b_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("ev_b_img.png")
    plt.clf()


def main():
    dataset = Dataset()
    img, ev_img, (t_st, t_end) = dataset.get_data_img_ev_pair(1)
    t_end = t_end
    cache_f = "cache.pkl"
    if not osp.exists(cache_f):
        raw_ev = dataset.event_buffer.retrieve_data(t_st, t_end)
        with open(cache_f, "wb") as f:
            pickle.dump(raw_ev, f)
    else:
        with open(cache_f, "rb") as f:
            raw_ev = pickle.load(f)
    
    ev_b_img = make_ev_boundary_img(raw_ev, (t_st + t_end)/2, dataset.get_img_size())
    save_ev_b_img(ev_b_img)
    
    log_img = np.log(img + 1e-7)
    cs = np.arange(0.1,0.5,0.01)
    phis = []
    for c in tqdm(cs, "calculating phis"):
        latent_log_img = log_img + c*ev_img
        latent_img = np.exp(latent_log_img)
        phis.append(calc_phi_tv_edge(latent_img, ev_b_img, c))
    
    phi_dict = jax.tree_map(lambda *x : np.array(x), *phis)
    print("found thresh:", cs[phi_dict["phi"].argmin()])

    fig, axes = plt.subplots(1, len(phi_dict))
    for i, (k, v) in enumerate(phi_dict.items()):
        axes[i].plot(cs, v)
        axes[i].set_title(k)

    plt.savefig("calc_vals.png")
    plt.clf()

    make_vid(SAVE_DIR)    

if __name__ == "__main__":
    main()