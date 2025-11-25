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
    
    # ------- Helper ----------

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
        #loa
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

    # ------------- Helper Functions -------------
    def _add_front_noise(self, label, mask_back):
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





