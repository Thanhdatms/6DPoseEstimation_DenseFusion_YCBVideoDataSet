import os
import os.path
import random
import numpy as np
from scipy import io as scio
import torch.utils.data as data
from PIL import Image
import numpy.ma as ma
import torchvision.transforms as transforms


class PoseDataset(data.Dataset):
    def __init__(self, mode, num_point, add_noise, root, noise_trans, refine, ycb_root):
        self.mode = mode # train or test
        self.num_point = num_point # number of points sampled from point cloud
        self.add_noise = add_noise # whether to add noise to point cloud True/False
        self.root = root # root directory of dataset
        self.noise_trans = noise_trans # noise translation for data augmentation
        self.refine = refine # whether in refine stage True/False
        self.ycb_root = ycb_root # root directory of YCB models


        if mode == 'train':
            self.path = os.path.join(self.root, 'dataset_config', 'train_data_list.txt')
        else:
            self.path = os.path.join(self.root, 'dataset_config', 'test_data_list.txt')
        
        self.list, self.real, self.syn = self._load_image(self.path)

        # load point cloud data
        # root: dataset root
        # ycb_root: YCB Video Base root
        self.point_cloud = self._load_point_cloud(self.root, self.ycb_root)

        # setup camera intrinsics
        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        # Image transform
        self.transcolor = self.transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        # Misc
        self.minimun_num_pt = 50 # minimum number of points in point cloud
        self.symmetric_obj_idx = [12, 15, 18, 19, 20] # list of symmetric object indices (object đối xứng)
        self.num_pt_mesh_small = 500 # number of points in small object mesh (refine = False)
        self.num_pt_mesh_large = 2600 # number of points in large object mesh (refine = True)
        self.front_num = 2 # number of objects in front view for occlusion handling
        self.noise_img_loc = 0.0 # noise image location
        self.noise_img_scale = 7.0 # standard deviation of noise image

        # Bounding box snapping
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680] # list of border values for bounding box snapping
        self.img_width = 480
        self.img_length = 640
    
    # ------------- Helper Functions -------------

    def _load_image(self, path):
        data_list = []
        data_real = []
        data_syn = []

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                data_list.append(line)

                if 'data' in line:
                    data_real.append(line)
                else:
                    data_syn.append(line)
        
        return data_list, data_real, data_syn
    
    def _load_point_cloud(self, data_root, ycb_root):
        cld = {}
        class_id = 1

        point_path = os.path.join(ycb_root, 'models',)
        class_path = os.path.join(data_root, 'dataset_config', 'class_list.txt')

        with open(class_path, 'r') as f:
            for line in f:
                cls = line.strip()
                points = []

                with open(os.path.join(point_path, cls, 'points.xyz'), 'r') as pf:
                    for point_line in pf:
                        x, y, z = map(float, point_line.strip().split(' '))
                        points.append([x, y, z])
                cld[class_id] = np.array(points)
                class_id += 1
        
        return cld
    
    
    def __len__(self):
        return len(self.list)
    
    def get_sym_list(self):
        return self.symmetric_obj_idx
    
    def get_num_points_mesh(self):
        return self.num_pt_mesh_small if not self.refine else self.num_pt_mesh_large
    
    def __getitem__(self, index):
        sample_name = self.list[index]
        img = Image.open(os.path.join(self.root, f"{self.list[index]}-color.png"))
        depth = np.array(Image.open(os.path.join(self.root, f"{self.list[index]}-depth.png")))
        label = np.array(Image.open(os.path.join(self.root, f"{self.list[index]}-label.png")))
        meta = scio.loadmat(os.path.join(self.root, f"{self.list[index]}-meta.mat"))

        if 'data_syn' in self.list[index]:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1
        
        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front, front_mask, front_img = False, None, None

        if self.add_noise:
            add_front, front_mask, front_img = self._add_front_noise(label, mask_back)

        # Each frame has meta file containing id about objects in the frame ([ 5  7  9 10 16 18]) (N, 1) -> (N,)
        obj_idx = meta['cls_indexes'].flatten().astype(np.int32)

        # 
        mask_label, idx = self._sample_object_mask(depth, label, obj_idx)
        
        if self.add_noise:
            img = self.transcolor(img)

        img_masked, mask_conbined = self._crop_and_mask_image(
            img, mask_label, sample_name, mask_back, front_img, front_mask, add_front
        )
        
        # model_points = self._sample_model_points(obj_idx[idx])

        # # Target transfromation
        # target_r = meta['poses'][:, :, idx][:, :3] # rotation matrix 3x3 example: [[ 0.9998, -0.0176,  0.0083],
        #                                             #                      [ 0.0174,  0.9998, -0.0095],
        #                                             #                      [-0.0086,  0.0093,  0.9999]]
        # target_t = meta['poses'][:, :, idx][:, 3] # translation vector 3x1 example: [[ 0.1234],
        #                                             #                        [-0.0345],
        #                                             #                        [ 0.5678]]
        # if self.add_noise:
        #     add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for _ in range(3)])
        # else:
        #     add_t = np.array([0, 0, 0])

        # target = np.dot(model_points, target_r.T)
        # target = target + target_t + add_t

        # # Convert to torch tensors
        # cloud = np.concatenate((depth_masked, xmap_masked, ymap_masked), axis=1)


    # ------------- Helper Functions -------------
    
    def _add_front_noise(self, label, mask_back):
        """
        Add front view noise to simulate occlusion from other objects
        1. Randomly select a synthetic image from the synthetic dataset
        2. Check if the selected image has enough objects in the front view
        3. Create a mask for the front view objects
        4. Ensure that the main object is not overly occluded (at least 1000 points remain)
        5. Return the mask and front image if successful, otherwise return False    
        """
        for _ in range(0):
            seed = random.choice(self.syn)
            front = np.array(self.transcolor(
                Image.open(os.path.join(self.root, f"{seed}-color.png")).convert("RGB")
            ))
            front = np.transpose(front, (1, 0, 2))
            f_label = np.array(
                Image.open(os.path.join(self.root, f"{seed}-label.png"))
            )

            front_labels = np.unique(f_label)[1:]
            if (len(front_labels) < self.front_num):
                continue

            # Get only front_num objects ( 2 objects in front view)
            selected_front_labels = random.sample(list(front_labels), self.front_num)
            mask_front = None
            for i, f_i in enumerate(selected_front_labels):
                mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i)) # this is boolean mask (True is object, False is background)
                mask_front = mk if i == 0 else (mask_front * mk)

            t_label = label * mask_front
            # Không để ảnh thêm vào (front) che mất quá nhiều vật thể chính đảm bảo vật thể chính phải có ít nhất 1000 điểm
            if (t_label.nonzero()[0].size > 1000):
                return True, mask_front, front
            
        return False, None, None
    
    def _sample_object_mask(self, depth, label, obj_idx):

        for _ in range(10):
            idx = np.random.randint(len(obj_idx))
            obj_id = obj_idx[idx]

            mask_obj = (label == obj_id)
            mask_depth = (depth != 0)

            mask = mask_obj & mask_depth

            if mask.sum() > self.minimun_num_pt:
                return mask, idx
            
        # If no object has enough points, return the object with the maximum points
        num_points_list = [(label==idx) & (depth!=0) for idx in obj_idx]
        num_points_counts = [np.sum(num_points) for num_points in num_points_list]
        max_idx = np.argmax(num_points_counts)
        return (label == num_points_list[max_idx]), max_idx
            
    def _crop_and_mask_image(self, img, mask_label, name, mask_back, front_img, front_mask, add_front):
        """
        Crop and mask the image based on the object mask
        1. Find the bounding box of the object in the mask
        2. Snap the bounding box to predefined borders
        3. Crop the image and mask based on the bounding box
        4. Apply masking to the cropped image
        5. If add_front is True, blend in the front view noise
        6. Add Gaussian noise if add_noise is True
        """
        rmin, rmax, cmin, cmax= get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (1, 0, 2))[:, rmin:rmax, cmin:cmax] # delete alpha channel if exists

        # Apply synthetic backgroud if required
        if 'data_syn' in name:
            seed = random.choice(self.real)
            back = np.array(
                Image.open(os.path.join(self.root, f"{seed}-color.png")).convert("RGB")
            )
            back = np.transpose(back, (1, 0, 2))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] # apply background mask (mask_back = 1 1 1 0 0, back = 10 10 10 10 10, img  =  5  5  5 99 99 -> img_masked = 10 10 10 0 0)
            img_masked = img_masked + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * front_mask[rmin:rmax, cmin:cmax] + \
                         front_img[:, rmin:rmax, cmin:cmax] * ~(front_mask[rmin:rmax, cmin:cmax])

        if 'data_syn' in name:
            img_masked = img_masked + np.random.normal(loc=self.noise_img_loc, scale=self.noise_img_scale, size=img_masked.shape)
        
        return img_masked, mask_label

    def _sample_model_points(self, obj_id):
        points = self.point_cloud[obj_id]
        num_sample = self.num_pt_mesh_large if self.refine else self.num_pt_mesh_small
        if len(points) > num_sample:
            dellist = np.random.choice(len(points), len(points)-num_sample, replace=False)
            points = np.delete(points, dellist, axis=0)

        return points

def get_bbox(mask):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    # get bounding box of object
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_box, c_box = rmax - rmin, cmax - cmin

    for tt in range(len(border_list)):
        if r_box > border_list[tt] and r_box <= border_list[tt + 1]:
            r_box = border_list[tt + 1]
            break

    for tt in range(len(border_list)):
        if c_box > border_list[tt] and c_box <= border_list[tt + 1]:
            c_box = border_list[tt + 1]
            break
    
    # calculate center of bounding box
    r_center = [(rmin + rmax) // 2, (cmin + cmax) // 2]
    rmin = max(r_center[0] - r_box // 2, 0)
    rmax = min(r_center[0] + r_box // 2, img_width)
    cmin = max(r_center[1] - c_box // 2, 0)
    cmax = min(r_center[1] + c_box // 2, img_length)

    return rmin, rmax, cmin, cmax


