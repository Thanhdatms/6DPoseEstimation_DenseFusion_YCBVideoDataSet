import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import random
import scipy.io as scio
import numpy.ma as ma

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        # Dataset paths
        self.root = root
        self.num_pt = num_pt
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.refine = refine

        self.path = f'datasets/ycb/dataset_config/{mode}_data_list.txt'
        self.list, self.real, self.syn = self._load_data_list(self.path)

        # Load class point clouds
        self.cld = self._load_class_pointclouds('datasets/ycb/dataset_config/classes.txt')

        # Camera intrinsics
        self.cam_cx_1, self.cam_cy_1, self.cam_fx_1, self.cam_fy_1 = 312.9869, 241.3109, 1066.778, 1067.487
        self.cam_cx_2, self.cam_cy_2, self.cam_fx_2, self.cam_fy_2 = 323.7872, 279.6921, 1077.836, 1078.189

        self.xmap = np.arange(480)[:, None] * np.ones((480, 640))
        self.ymap = np.arange(640)[None, :] * np.ones((480, 640))

        # Image transforms
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Misc
        self.minimum_num_pt = 50
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.front_num = 2

        # Bounding box snapping
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.img_width = 480
        self.img_length = 640

        print(f"Loaded {len(self.list)} samples")

    def _load_data_list(self, path):
        data_list, real, syn = [], [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                data_list.append(line)
                if line.startswith('data/'):
                    real.append(line)
                else:
                    syn.append(line)
        return data_list, real, syn

    def _load_class_pointclouds(self, class_path):
        cld = {}
        class_id = 1
        with open(class_path) as f:
            for cls in f:
                cls = cls.strip()
                points = []
                with open(f'{self.root}/models/{cls}/points.xyz') as pf:
                    for line in pf:
                        x, y, z = map(float, line.strip().split(' '))
                        points.append([x, y, z])
                cld[class_id] = np.array(points)
                class_id += 1
        return cld

    def __len__(self):
        return len(self.list)

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh_large if self.refine else self.num_pt_mesh_small

    def __getitem__(self, index):
        sample_name = self.list[index]
        img, depth, label, meta = self._load_sample_files(sample_name)
        cam_cx, cam_cy, cam_fx, cam_fy = self._select_camera(sample_name)
        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))
        add_front, front_mask, front_img = False, None, None

        # Add front noise if required
        if self.add_noise:
            add_front, front_mask, front_img = self._add_front_noise(label, mask_back)

        obj_idx = meta['cls_indexes'].flatten().astype(np.int32)
        mask_label, idx = self._sample_object_mask(depth, label, obj_idx)
        if self.add_noise:
            img = self.trancolor(img)

        img_masked, mask_combined = self._crop_and_mask_image(
            img, mask_label, sample_name, mask_back, front_img, front_mask, add_front
        )

        choose, depth_masked, xmap_masked, ymap_masked = self._sample_points(
            mask_combined, depth, cam_cx, cam_cy, cam_fx, cam_fy
        )

        # Model points
        model_points = self._sample_model_points(obj_idx[idx])

        # Target transformation
        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        if self.add_noise:
            add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for _ in range(3)])
        else:
            add_t = np.zeros(3)

        target = np.dot(model_points, target_r.T)
        target = target + target_t + add_t

        # Convert everything to torch tensors
        cloud = np.concatenate((depth_masked, xmap_masked, ymap_masked), axis=1)
        return (torch.from_numpy(cloud.astype(np.float32)),
                torch.LongTensor(choose.astype(np.int32)),
                self.norm(torch.from_numpy(img_masked.astype(np.float32))),
                torch.from_numpy(target.astype(np.float32)),
                torch.from_numpy(model_points.astype(np.float32)),
                torch.LongTensor([int(obj_idx[idx]) - 1]))

    # ---------------- Helper Functions ----------------
    def _load_sample_files(self, name):
        img = Image.open(f'{self.root}/{name}-color.png')
        depth = np.array(Image.open(f'{self.root}/{name}-depth.png'))
        label = np.array(Image.open(f'{self.root}/{name}-label.png'))
        meta = scio.loadmat(f'{self.root}/{name}-meta.mat')
        return img, depth, label, meta

    def _select_camera(self, name):
        if name[:8] != 'data_syn' and int(name[5:9]) >= 60:
            return self.cam_cx_2, self.cam_cy_2, self.cam_fx_2, self.cam_fy_2
        return self.cam_cx_1, self.cam_cy_1, self.cam_fx_1, self.cam_fy_1

    def _add_front_noise(self, label, mask_back):
        for _ in range(5):
            seed = random.choice(self.syn)
            front = np.array(self.trancolor(
                Image.open(f'{self.root}/{seed}-color.png').convert("RGB")))
            front = np.transpose(front, (2, 0, 1))
            f_label = np.array(Image.open(f'{self.root}/{seed}-label.png'))
            front_labels = np.unique(f_label)[1:]
            if len(front_labels) < self.front_num:
                continue
            front_labels = random.sample(list(front_labels), self.front_num)
            mask_front = None
            for i, f_i in enumerate(front_labels):
                mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                mask_front = mk if i == 0 else mask_front * mk
            t_label = label * mask_front
            if len(t_label.nonzero()[0]) > 1000:
                return True, mask_front, front
        return False, None, None

    def _sample_object_mask(self, depth, label, obj):
        while True:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                return mask_label, idx

    def _crop_and_mask_image(self, img, mask_label, name, mask_back, front_img, front_mask, add_front):
        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        # Apply synthetic background if required
        if name[:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(
                Image.open(f'{self.root}/{seed}-color.png').convert("RGB")))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * front_mask[rmin:rmax, cmin:cmax] + \
                         front_img[:, rmin:rmax, cmin:cmax] * ~(front_mask[rmin:rmax, cmin:cmax])

        if name[:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        return img_masked, mask_label

    def _sample_points(self, mask, depth, cam_cx, cam_cy, cam_fx, cam_fy):
        rmin, rmax, cmin, cmax = get_bbox(mask)
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        cam_scale = 1000.0  # example default, replace with meta factor_depth if needed
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        return choose, pt2, pt0, pt1

    def _sample_model_points(self, obj_idx):
        points = self.cld[obj_idx]
        num_sample = self.num_pt_mesh_large if self.refine else self.num_pt_mesh_small
        if len(points) > num_sample:
            dellist = random.sample(range(len(points)), len(points) - num_sample)
            points = np.delete(points, dellist, axis=0)
        return points


# Bounding box helper (unchanged)
def get_bbox(label):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width, img_length = 480, 640
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b, c_b = rmax - rmin, cmax - cmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break

    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
        
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = max(center[0] - r_b // 2, 0)
    rmax = min(center[0] + r_b // 2, img_width)
    cmin = max(center[1] - c_b // 2, 0)
    cmax = min(center[1] + c_b // 2, img_length)
    return rmin, rmax, cmin, cmax
