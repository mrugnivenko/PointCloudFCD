import imp
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from numpy.random import default_rng
import datasets.data_utils as d_utils


def make_coin_flip(threshold: float = 0.7) -> bool:
    """
    Function simulates flip of biased coin

    :params threshold: float, percent of False return

    :outpurs result: bool, True in (1-threshold)% of cases
    """

    return np.random.rand() >= threshold


def give_random_point(
    sagittal_shape: int, coronal_shape: int, axial_shape: int, crop_size: int
) -> tuple:
    """
    Function returns random point inside 3D tensor of shape
    [(sagittal_shape - crop_size) x (coronal_shape - crop_size) x (axial_shape - crop_size)].
    This point will be the min point of crop

    :params sagittal_shape: int, sagittal shape
    :params coronal_shape: int, coronal shape
    :params axial_shape: int, axial shape
    :params crop_size int: size of crop

    :outputs result: tuple(int, int, int), random point inside 3D tensor of shape
    [(sagittal_shape - crop_size) x (coronal_shape - crop_size) x (axial_shape - crop_size)]
    """

    return (
        random.randint(0, sagittal_shape - crop_size),
        random.randint(0, coronal_shape - crop_size),
        random.randint(0, axial_shape - crop_size),
    )


def give_random_point_inside_fcd(mask: np.ndarray) -> tuple:
    """
    Function returns random point lying inside FCD region

    :params mask: np.array, binary mask of FCD

    :outputs result: tuple(int, int, int), random point lying inside FCD region
    """

    xid, yid, zid = np.where(mask == 1)
    chosen_id = np.random.choice(len(xid))
    c_x, c_y, c_z = xid[chosen_id], yid[chosen_id], zid[chosen_id]
    return (c_x, c_y, c_z)


def get_random_crop(
    single_data_dict, crop_size=64, coin_flip_threshold=0.7
):
    """
    Function for getting random crop. With probability coin_flip_threshold crop will contain brain
    without FCD region and with probability (1-coin_flip_threshold) crop will contain brain with FCD region. 

    :params single_data_dict: dict, dictionary with one brain data - brain, label, ...
    :params crop_size int: size of output crop
    :params coin_flip_threshold: float, percent of False return (take crop inside FCD region) in flip of biased coin
    """

    sagittal_shape, coronal_shape, axial_shape = single_data_dict["brains"].shape

    # if there is no FCD region, then take random crop not inside FCD
    if single_data_dict["labels"].sum() == 0:
        coin_flip = True
    else:
        coin_flip = make_coin_flip(threshold=coin_flip_threshold)

    # take random crop not inside FCD
    if coin_flip:
        is_break = False
        while not is_break:
            min_point = give_random_point(
                sagittal_shape, coronal_shape, axial_shape, crop_size
            )

            brain_crop = single_data_dict["brains"][
                min_point[0] : min_point[0] + crop_size,
                min_point[1] : min_point[1] + crop_size,
                min_point[2] : min_point[2] + crop_size,
            ]
            mask_crop = single_data_dict["labels"][
                min_point[0] : min_point[0] + crop_size,
                min_point[1] : min_point[1] + crop_size,
                min_point[2] : min_point[2] + crop_size,
            ]

            if (brain_crop.sum() != 0) and mask_crop.sum() == 0:
                is_break = True
            else:
                is_break = False

        result_dict = {}
        for key in single_data_dict:
            result_dict[key] = single_data_dict[key][
                min_point[0] : min_point[0] + crop_size,
                min_point[1] : min_point[1] + crop_size,
                min_point[2] : min_point[2] + crop_size,
            ]

        return result_dict, min_point

    else:
        is_break = False
        while not is_break:
            min_point_inside_fcd = give_random_point_inside_fcd(
                single_data_dict["labels"]
            )

            sagittal_width = np.random.randint(0, crop_size)
            coronal_width = np.random.randint(0, crop_size)
            axial_width = np.random.randint(0, crop_size)

            brain_crop = single_data_dict["brains"][
                min_point_inside_fcd[0]
                - sagittal_width : min_point_inside_fcd[0]
                + (crop_size - sagittal_width),
                min_point_inside_fcd[1]
                - coronal_width : min_point_inside_fcd[1]
                + (crop_size - coronal_width),
                min_point_inside_fcd[2]
                - axial_width : min_point_inside_fcd[2]
                + (crop_size - axial_width),
            ]
            
            if ((brain_crop.sum() != 0) and (brain_crop.shape == (crop_size, crop_size, crop_size)) and ((brain_crop > 0).sum() >= 0)):
                is_break = True
            else:
                is_break = False

        result_dict = {}
        for key in single_data_dict:
            result_dict[key] = single_data_dict[key][
                min_point_inside_fcd[0]
                - sagittal_width : min_point_inside_fcd[0]
                + (crop_size - sagittal_width),
                min_point_inside_fcd[1]
                - coronal_width : min_point_inside_fcd[1]
                + (crop_size - coronal_width),
                min_point_inside_fcd[2]
                - axial_width : min_point_inside_fcd[2]
                + (crop_size - axial_width),
            ]

        return result_dict, [
            min_point_inside_fcd[0] - sagittal_width,
            min_point_inside_fcd[1] - coronal_width,
            min_point_inside_fcd[2] - axial_width,
        ]


def get_deterministic_crop(
    object_to_crop, sagittal_iter, coronal_iter, axial_iter, crop_size
):
    """
    Function returns crop with number (sagittal_iter, coronal_iter, axial_iter)
    
    :params object_to_crop: np.ndarray, data for cropping (can be brain, FCD mask and so on) 
    :param sagittal_iter: int, number of crop in sagittal direction
    :param coronal_iter: int, number of crop in coronal direction
    :param axial_iter: int, number of crop in axial direction
    :params crop_size int: size of output crop
    
    :outputs result: np.ndarray, required crop of the input data
    """

    return object_to_crop[
        sagittal_iter * crop_size : (sagittal_iter + 1) * crop_size,
        coronal_iter * crop_size : (coronal_iter + 1) * crop_size,
        axial_iter * crop_size : (axial_iter + 1) * crop_size,
    ]


def get_inference_crops(single_data_dict, crop_size=64, step_size=None):
    """
    Function returns dictionary with all non-empty in terms of brain crops for all data (brain, mask and so on).
    Also minimal coordinates for all crop are returned 
    
    :params single_data_dict: dict, dictionary with one brain data - brain, label, ...
    :params crop_size: int, size of output crops
    :params step_size: int, step of cropping, like stride in convolution
    
    :outputs single_data_crop_dict: dict, dictionary with all non-empty in terms of brain crops for all data (brain, mask and so on)
    :outputs center_coords: list, list with minimal coordinates of considered crops
    """
    
    # cropping without intersection
    if step_size is None:
        step_size = crop_size

    sagittal_shape, coronal_shape, axial_shape = single_data_dict["brains"].shape

    single_data_crop_dict = {}
    for key in single_data_dict.keys():
        single_data_crop_dict[key] = []
    center_coords = []

    for sagittal_iter in range(
        sagittal_shape // step_size - (crop_size // step_size) + 1
    ):
        for coronal_iter in range(
            coronal_shape // step_size - (crop_size // step_size) + 1
        ):
            for axial_iter in range(
                axial_shape // step_size - (crop_size // step_size) + 1
            ):
                cropped_brain = get_deterministic_crop(
                    single_data_dict["brains"],
                    sagittal_iter,
                    coronal_iter,
                    axial_iter,
                    crop_size,
                )
                if cropped_brain.sum() != 0.0:
                    for key in single_data_dict.keys():
                        single_data_crop_dict[key].append(
                            get_deterministic_crop(
                                single_data_dict[key],
                                sagittal_iter,
                                coronal_iter,
                                axial_iter,
                                crop_size,
                            )
                        )
                    center_coords.append(
                        (
                            sagittal_iter * crop_size,
                            coronal_iter * crop_size,
                            axial_iter * crop_size,
                        )
                    )

    return single_data_crop_dict, center_coords


def brain_and_mask_to_point_cloud_and_labels(
    single_crop_dict,
    crop_size=64,
    is_crop=True,
    return_min_point=True,
    return_air_mask=False,
    return_abs_coords=False,
    center=None,
    coin_flip_threshold=0.7,
    MEANS={},
    STDS={},
):
    """
    Function creates Point Cloud and corresponding labels (FCD or not) from single crop dict 
    (if the input is single data dict, then single crop dict is chosen randomly - used for training)
    
    :params single_crop_dict: dict, dictionary for single data (in case of train mode) or for single crop (in case of test mode)
    :params crop_size: int, size of crops for train/test. Crops will be of size [crop_size x crop_size x crop_size]
    :params is_crop: bool, whether the input data is dictionary for single data (train mode) or not (dictionary for single crop)
    :params return_min_point: bool, whether to return min point of crop (absolute or relative, depending on return_abs_coords parameter)
    :params return_air_mask: bool, whether to return mask of air 
    :params return_abs_coords: bool, whether to use absolute coordinates 
    :params center: list, list with absolute coordinates of crop
    :params coin_flip_threshold: float, percent of False return (take crop inside FCD region) in flip of biased coin
    :params MEANS: dict, dictionary with means of single data dictioanry elements
    :params STDS: dict, dictionary with stds of single data dictioanry elements
    
    :outputs
    """

    if is_crop: # if the input is single_data_dict 
        single_crop_dict, min_point = get_random_crop(
            single_crop_dict,
            crop_size=crop_size,
            coin_flip_threshold=coin_flip_threshold,
        )
        min_point = [x - crop_size for x in min_point] # coordinates before padding
    elif return_abs_coords: # if the input is single_crop_dict already and we consider absolute coordinates
        min_point = np.array(center)
    else: # if the input is single_crop_dict already and we consider relative coordinates 
        min_point = [0, 0, 0]

    size_abs = (241, 336, 283)
    size = single_crop_dict["brains"].shape
    if return_abs_coords:
        means = [x // 2 for x in size_abs]
        half_range = [x // 2 for x in size_abs]
    else:
        means = [x // 2 for x in size]
        half_range = [x // 2 for x in size]

    grid_x, grid_y, grid_z = torch.meshgrid(
        (torch.tensor(range(size[0])) + min_point[0] - means[0]) / half_range[0],
        (torch.tensor(range(size[1])) + min_point[1] - means[1]) / half_range[1],
        (torch.tensor(range(size[2])) + min_point[2] - means[2]) / half_range[2],
    )

    point_cloud = torch.cat(
        [
            grid_x.unsqueeze(-1).float(),
            grid_y.unsqueeze(-1).float(),
            grid_z.unsqueeze(-1).float(),
        ]
        + [
            ((torch.tensor(single_crop_dict[key]) - MEANS[key]) / STDS[key])
            .float()
            .unsqueeze(-1)
            .float()
            for key in single_crop_dict
            if key != "labels"
        ],
        -1,
    )

    point_cloud_fcd = point_cloud[
        (single_crop_dict["labels"] == 1) & (single_crop_dict["brains"] > 0), :
    ]
    pc_brain_without_fcd_air = point_cloud[
        (single_crop_dict["labels"] == 0) & (single_crop_dict["brains"] <= 0), :
    ]
    pc_brain_without_fcd_noair = point_cloud[
        (single_crop_dict["labels"] == 0) & (single_crop_dict["brains"] > 0), :
    ]

    without_fcd_shape = (
        pc_brain_without_fcd_air.shape[0] + pc_brain_without_fcd_noair.shape[0]
    )
    without_fcd_air_shape = pc_brain_without_fcd_air.shape[0]
    without_fcd_noair_shape = pc_brain_without_fcd_noair.shape[0]

    res = {
        "points": torch.cat(
            [point_cloud_fcd, pc_brain_without_fcd_noair, pc_brain_without_fcd_air]
        ),
        "labels": np.array([1] * point_cloud_fcd.shape[0] + [0] * without_fcd_shape),
    }

    if return_min_point:
        res["min_point"] = min_point
    if return_air_mask:
        res["air_mask"] = np.array(
            [0] * (point_cloud_fcd.shape[0] + without_fcd_noair_shape)
            + [1] * without_fcd_air_shape
        )

    return res


class BrainDataSegCrop:
    """
    Dataloader.
    Create loader with crops of the brain.
    """

    def __init__(
        self,
        task="train",
        crop_size=64,
        step_size=None,
        num_points=2048,
        transforms=None,
        return_center=False,
        is_folded=True,
        data_dict={},
        return_min_point_train=True,
        return_air_mask_test=False,
        return_abs_coords=False,
        return_pc_without_air_points=False,
        coin_flip_threshold=0.7,
        MEANS={"brains": 0, "curvs": 0, "thickness": 0, "sulc": 0},
        STDS={"brains": 1, "curvs": 1, "thickness": 1, "sulc": 1},
    ):

        """
        :params task : str, one of {'test', 'train'}. Shows the purpose for which the loader is created.
            Mode 'train' is used for training and validation, and mode 'test' is used for inference.
        :params crop_size: int, size of crops for train/test. Crops will be of size [crop_size x crop_size x crop_size]
        :params num_points: int, number of points to randomly pick out of all points
        :params transforms: obj, some sequence of transforms, created by transforms.Compose
        :params return_center: bool, this corresponds to whether we want to recover original crop position or not
        :params is_folded: bool, !!!!!!!!!!!!!!!!!!!!!!!
        :params data_dict: dict, dictionary with paths to the data
        :params return_min_point_train, bool whether to return min points of crops
        :params return_air_mask_test, bool whether to return air mask for inference 
        :params return_abs_coords, bool whether to return absolute coordinates
        :params return_pc_without_air_points, bool !!!!!!!!!!!!!!!!!!!!!!!
        :params coin_flip_threshold: float, percent of False return (take crop inside FCD region) in flip of biased coin
        :params MEANS: dict, dictionary with means of the data for normalization
        :params STDS: dict, dictionary with stds of the data for normalization
        """

        self.task = task
        self.crop_size = crop_size

        if step_size is None:
            step_size = crop_size

        self.num_points = num_points
        self.transforms = transforms
        self.return_center = return_center
        self.return_min_point_train = return_min_point_train
        self.return_air_mask_test = return_air_mask_test
        self.return_abs_coords = return_abs_coords
        self.return_pc_without_air_points = return_pc_without_air_points
        self.coin_flip_threshold = coin_flip_threshold
        self.MEANS = MEANS
        self.STDS = STDS

        if task == "train":
            self.data_dict = data_dict

        elif task == "test":
            data_dict_loaded = {}
            for key in data_dict.keys():
                data_dict_loaded[key] = [np.load(data_dict[key], allow_pickle=True)]

            self.center_coords = []
            self.points = []
            self.labels = []

            if return_air_mask_test or return_pc_without_air_points:
                self.air_masks = []

            for idx in range(len(data_dict_loaded["brains"])):
                sagittal_shape, coronal_shape, axial_shape = data_dict_loaded["brains"][
                    idx
                ].shape

                deltas = []
                for shape in [sagittal_shape, coronal_shape, axial_shape]:
                    if shape % crop_size != 0:
                        deltas.append((0, (shape // crop_size + 1) * crop_size - shape))
                    else:
                        deltas.append((0, 0))

                single_data_dict = {}
                for key in data_dict_loaded.keys():
                    single_data_dict[key] = np.pad(
                        data_dict_loaded[key][idx], deltas, "constant", constant_values=0
                    )

                single_data_dict, center_coords = get_inference_crops(
                    single_data_dict, crop_size=crop_size, step_size=step_size
                )

                if return_abs_coords:
                    self.center_coords += center_coords

                points_labels = [
                    brain_and_mask_to_point_cloud_and_labels(
                        {key: single_data_dict[key][idx_] for key in single_data_dict},
                        crop_size=self.crop_size,
                        is_crop=False,
                        return_air_mask=return_air_mask_test
                        or return_pc_without_air_points,
                        return_abs_coords=return_abs_coords,
                        center=center_coords[idx_],
                        MEANS=self.MEANS,
                        STDS=self.STDS,
                    )
                    for idx_ in range(len(single_data_dict["brains"]))
                ]

                self.points += [x["points"] for x in points_labels]
                self.labels += [x["labels"] for x in points_labels]
                if return_air_mask_test or return_pc_without_air_points:
                    self.air_masks += [x["air_mask"] for x in points_labels]
            if not is_folded:
                self.center_coords = self.center_coords[0]

    def __getitem__(self, idx):
        """
        Returns random point cloud of idx's brain if task == 'train' and the idx's crop of the only one 
        test brain if task == 'test' in the following format:
            current_points: np.array of shape (num_points, 4) 3d coords and feature for each point in point cloud
            mask: np.array of shape (num_points,) of 0 and 1, contains 1 for each unique point and 0 for every 2,3,4... repetition of the point
            current_points_labels: np.array of shape (num_points,) of 0 and 1, contains 1 for FCD point and 0 for non-FCD
            label: allways 0, needed for proper network functioning 
            center_coords: (int,int,int) - optional, will be returned if return_center == True, contains x,y,z coordinates of crops edge.

        :params idx: int, index of crop (inference) or brain (train)
        """
        
        if self.task == "train":
            rng = default_rng()
            
            single_data_dict = {key: self.data_dict[key][idx] for key in self.data_dict}
            for key in single_data_dict:
                single_data_dict[key] = np.load(
                    single_data_dict[key], allow_pickle=True
                )
            for key in single_data_dict:
                single_data_dict[key] = np.pad(
                    single_data_dict[key],
                    (
                        (self.crop_size, self.crop_size),
                        (self.crop_size, self.crop_size),
                        (self.crop_size, self.crop_size),
                    ),
                    "constant",
                    constant_values=0,
                )

            out = brain_and_mask_to_point_cloud_and_labels(
                single_data_dict,
                crop_size=self.crop_size,
                return_min_point=self.return_min_point_train,
                return_abs_coords=self.return_abs_coords,
                return_air_mask=self.return_pc_without_air_points,
                coin_flip_threshold=self.coin_flip_threshold,
                MEANS=self.MEANS,
                STDS=self.STDS,
            )

            current_points = out["points"]
            current_points_labels = out["labels"]
            if self.return_min_point_train:
                min_points = out["min_point"]
            if self.return_pc_without_air_points:
                current_air_masks = out["air_mask"]

        elif self.task == "test":
            rng = default_rng(42)
            current_points, current_points_labels = self.points[idx], self.labels[idx]
            if self.return_air_mask_test or self.return_pc_without_air_points:
                current_air_masks = self.air_masks[idx]
                
        cur_num_points = current_points.shape[0]
        if cur_num_points >= self.num_points:
            choice = rng.choice(cur_num_points, self.num_points)
            current_points = current_points[choice, :]
            current_points_labels = current_points_labels[choice]
            if self.return_air_mask_test or self.return_pc_without_air_points:
                current_air_masks = current_air_masks[choice]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            padding_num = self.num_points - cur_num_points
            shuffle_choice = rng.permutation(np.arange(cur_num_points))
            padding_choice = rng.choice(cur_num_points, padding_num)
            choice = np.hstack([shuffle_choice, padding_choice])
            current_points = current_points[choice, :]
            current_points_labels = current_points_labels[choice]
            if self.return_air_mask_test or self.return_pc_without_air_points:  
                current_air_masks = current_air_masks[choice]
            mask = torch.cat(
                [torch.ones(cur_num_points), torch.zeros(padding_num)]
            ).type(torch.int32)
            
        if self.return_pc_without_air_points:  # NNN start
            current_points_without_air = current_points[current_air_masks == 0]
            current_points_labels_without_air = current_points_labels[
                current_air_masks == 0
            ]
            current_air_masks_without_air = current_air_masks[current_air_masks == 0]
            mask_without_air = mask[current_air_masks == 0]

            cur_num_points = current_points_without_air.shape[0]
            padding_num = self.num_points - cur_num_points
            shuffle_choice = rng.permutation(np.arange(cur_num_points))
            padding_choice = rng.choice(cur_num_points, padding_num)
            choice = np.hstack([shuffle_choice, padding_choice])
            current_points = current_points_without_air[choice, :]
            current_points_labels = current_points_labels_without_air[choice]
            current_air_masks = current_air_masks_without_air[choice]
            current_mask = mask_without_air[choice]
            mask = torch.cat([current_mask, torch.zeros(padding_num)]).type(
                torch.int32
            )  # NNN end

        if self.transforms is not None:
            current_points = self.transforms(current_points)
            
        label = torch.from_numpy(np.array(0)).type(torch.int64)
        current_points_labels = torch.from_numpy(current_points_labels).type(
            torch.int64
        )
        
        res = {
            "current_points": current_points,
            "mask": mask,
            "current_points_labels": current_points_labels,
            "label": label,
        }
        
        if self.return_center:
            res["center"] = self.center_coords[idx]
        if self.return_air_mask_test:
            res["air_mask"] = current_air_masks
        if self.return_min_point_train and self.task == "train":
            res["min_point"] = min_points
            
        return res

    def __len__(self):
        """
        Returns dataloader length.
        Important to understand that if task == 'train' it will be equal to the number of brains
        in the train dataset and if task == 'test' it will be the number of crops for the only one test brain.
        """
        if self.task == "train":
            return len(self.data_dict["brains"])
        elif self.task == "test":
            return len(self.labels)


def dataset_to_best_weights(dataset, features: list, repeats: int = 2):
    """ 
    Function computes means and stds for all types of data for further normalization. Also it computes 
    weights for weighted loss function according to quantity of positive class.
    
    :params dataset: BrainDataSegCrop
    :params repeaets: int, number of calculations of means and std for random crop of each brain
    
    :outputs weights: tuple, (weight for negative class, weight for positive class)
    :outputs dict_means: dict, dictionary with means for all types of data
    :outputs dict_stds: dict, dictionary with stds for all types of data
    """

    means = []
    dict_means = {feature: [] for feature in features}
    dict_stds = {feature: [] for feature in features}
    dict_features_to_numbers = {feature: i+3 for i, feature in enumerate(features)}
    
    for _ in range(repeats):
        for element in dataset:
            points_labels = element["current_points_labels"]
            current_points = element["current_points"]
            for key in features:
                dict_means[key].append(torch.mean(current_points[:, dict_features_to_numbers[key]]).detach().cpu())
                dict_stds[key].append(torch.std(current_points[:, dict_features_to_numbers[key]]).detach().cpu())

            means.append(np.mean(np.array(points_labels.detach().cpu())))
            del element, points_labels, current_points
    mean = np.mean(means)

    for key in dict_means:
        dict_means[key] = np.mean(dict_means[key])
        dict_stds[key] = np.mean(dict_stds[key])

    return [mean, 1 - mean], dict_means, dict_stds


def get_loader_crop(
    config,
    num_points=4096,
    batch_size=16,
    crop_size=64,
    train_dict={},
    test_dict={},
    is_folded=True,
    return_min_point_train=True,
    return_abs_coords=False,
    return_pc_without_air_points=False,
    coin_flip_threshold=0.8,
    weighted_loss=False,
):

    """
    Function for creation dataloaders
    See BrainDataSegCrop for more details.
    The only difference here is that BrainDataSegCrop are put into batches.

    :params num_points: int, number of points to randomly pick out of all points
    :params batch_size: int, batch size for train/test loader
    :params crop_size: int, size of crops for train/test
    :params train_dict: dict, dictionary with paths to train data
    :params test_dict: dict, dictionary with paths to test data
    :params is_folded: bool, whether to use folede stratagy or leave-one-out
    :params return_min_point_train, bool whether to return min points of crops
    :params return_abs_coords, bool whether to return absolute coordinates
    :params return_pc_without_air_points: bool,  !!!!!!!!!!!!!!!!!!!!!!!
    :params coin_flip_threshold: float, percent of False return (take crop inside FCD region) in flip of biased coin
    :params weighted_loss: bool, whether to calculate weights for weighted loss function according to the quantity of positive class or not

    :outputs train_loader: torch.utils.data.DataLoader, train loader
    :outputs test_loader: torch.utils.data.DataLoader, test loader
    :outputs weights: tuple or None, (weight for negative class, weight for positive class)
    """

    transform_test = None
    transform_train = transforms.Compose(
        [
            d_utils.PointcloudRandomRotate(
                x_range=config.x_angle_range,
                y_range=config.y_angle_range,
                z_range=config.z_angle_range,
            ),
            d_utils.PointcloudScaleAndJitter(
                scale_low=config.scale_low,
                scale_high=config.scale_high,
                std=config.noise_std,
                clip=config.noise_clip,
                augment_symmetries=config.augment_symmetries,
            ),
        ]
    )

    train_dataset = BrainDataSegCrop(
        num_points=num_points,
        task="train",
        crop_size=crop_size,
        transforms=transform_train,
        data_dict=train_dict,
        is_folded=is_folded,
        return_min_point_train=return_min_point_train,
        return_abs_coords=return_abs_coords,
        return_pc_without_air_points=return_pc_without_air_points,
        coin_flip_threshold=coin_flip_threshold,
    )

    if weighted_loss:
        weights, MEANS, STDS = dataset_to_best_weights(train_dataset, [x for x in train_dict.keys() if 'labels' not in x])
        print(f"Weights are: {weights}")
        print(f"MEANS are: {MEANS}")
        print(f"STDS are: {STDS}")
            
        config['MEANS'] = MEANS
        config['STDS'] = STDS
        with open(f'../experiments/{config.EXP_NAME}/config.json', 'w') as file:
            json.dump(config, file)
        
        train_dataset = BrainDataSegCrop(
            num_points=num_points,
            task="train",
            crop_size=crop_size,
            transforms=transform_train,
            data_dict=train_dict,
            is_folded=is_folded,
            return_min_point_train=return_min_point_train,
            return_abs_coords=return_abs_coords,
            return_pc_without_air_points=return_pc_without_air_points,
            coin_flip_threshold=coin_flip_threshold,
            MEANS=MEANS,
            STDS=STDS,
        )
        
    print("Train dataset created")

    test_dataset = BrainDataSegCrop(
        num_points=num_points,
        task="train",
        crop_size=crop_size,
        data_dict=test_dict,
        transforms=transform_test,
        is_folded=is_folded,
        return_abs_coords=return_abs_coords,
        return_pc_without_air_points=return_pc_without_air_points,
        MEANS=MEANS,
        STDS=STDS,
    )
    print("Test dataset created")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
    )

    if weighted_loss:
        return train_loader, test_loader, weights
    else:
        return train_loader, test_loader