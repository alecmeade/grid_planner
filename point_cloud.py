import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

class PointCloud2D():

    def __init__(self, data: np.ndarray):
        assert (data.shape[1] == 2)
        self.data = data


    def plot(self, ax, sample_rate: float = 0.1):
        plot_idx = np.random.choice(np.arange(self.data.shape[0]), 
                                    int(self.data.shape[0] * sample_rate),
                                    replace = False)
        
        ax.scatter(self.data[plot_idx, 0], 
                    self.data[plot_idx, 1],
                    s = 0.5)


class PointCloud3D():

    def __init__(self, data: np.ndarray):
        assert (data.shape[1] == 3)
        self.data = data


    def to_2D(self, z_min: float, z_max: float) -> PointCloud2D:
        mask = np.logical_and(self.data[:, 2] >= z_min, self.data[:, 2] <= z_max)
        return PointCloud2D(self.data[mask, :2])


def read_point_cloud_log(path: str, row_size: int, double_precision: bool = True) -> np.ndarray:
    with open(path, 'rb') as f:
        data_type =  np.double if double_precision else np.single
        data = np.fromfile(f, data_type)
        return data.reshape((-1, row_size))
