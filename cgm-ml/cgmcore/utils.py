import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
import datetime
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
try:
    import vtk
except Exception as e:
    print("WARNING! VTK not available. This might limit the functionality.") 
from pyntcloud import PyntCloud
import pickle

    
def load_pcd_as_ndarray(pcd_path):
    """
    Loads a PCD-file. Yields a numpy-array.
    """
        
    return PyntCloud.from_file(pcd_path).points.values

    
def load_vtk(vtk_path):
    """
    Loads a VTK-file. Yields a numpy-array.
    """

    reader = vtk.vtkDataSetReader()
    reader.SetFileName(vtk_path)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()

    points = np.zeros((data.GetNumberOfPoints(), 3))

    for i in range(data.GetNumberOfPoints()):
        points[i] = data.GetPoint(i)

    return points


def render_pointcloud(points, title=None):
    """
    Renders a point-cloud.
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2], s=0.5, cmap="gray", alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if title != None:
        plt.title(title)

    plt.show()
    plt.close()


def ensure_voxelgrid_shape(voxelgrid, voxelgrid_target_shape):
    voxelgrid = pad_voxelgrid(voxelgrid, voxelgrid_target_shape)
    voxelgrid = crop_voxelgrid(voxelgrid, voxelgrid_target_shape)
    return voxelgrid


def pad_voxelgrid(voxelgrid, voxelgrid_target_shape):

    pad_before = [0.0] * 3
    pad_after = [0.0] * 3
    for i in range(3):
        pad_before[i] = (voxelgrid_target_shape[i] - voxelgrid.shape[i]) // 2
        pad_before[i] = max(0, pad_before[i])
        pad_after[i] = voxelgrid_target_shape[i] - pad_before[i] - voxelgrid.shape[i]
        pad_after[i] = max(0, pad_after[i])
    voxelgrid = np.pad(
        voxelgrid,
        [(pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])],
        'constant', constant_values=[(0, 0), (0, 0), (0, 0)]
    )

    return voxelgrid


def crop_voxelgrid(voxelgrid, voxelgrid_target_shape):

    while voxelgrid.shape[0] > voxelgrid_target_shape[0]:
        voxels_start = np.count_nonzero(voxelgrid[0,:,:] != 0.0)
        voxels_end = np.count_nonzero(voxelgrid[-1,:,:] != 0.0)
        if voxels_start > voxels_end:
            voxelgrid = voxelgrid[:-1,:,:]
        else:
            voxelgrid = voxelgrid[1:,:,:]

    while voxelgrid.shape[1] > voxelgrid_target_shape[1]:
        voxels_start = np.count_nonzero(voxelgrid[:,0,:] != 0.0)
        voxels_end = np.count_nonzero(voxelgrid[:,-1,:] != 0.0)
        if voxels_start > voxels_end:
            voxelgrid = voxelgrid[:,:-1,:]
        else:
            voxelgrid = voxelgrid[:,1:,:]

    while voxelgrid.shape[2] > voxelgrid_target_shape[2]:
        voxels_start = np.count_nonzero(voxelgrid[:,:,0] != 0.0)
        voxels_end = np.count_nonzero(voxelgrid[:,:,-1] != 0.0)
        if voxels_start > voxels_end:
            voxelgrid = voxelgrid[:,:,:-1]
        else:
            voxelgrid = voxelgrid[:,:,1:]

    return voxelgrid


def center_crop_voxelgrid(voxelgrid, voxelgrid_target_shape):

    # Center crop.
    crop_start = [0.0] * 3
    crop_end = [0.0] * 3
    for i in range(3):
        crop_start[i] = (voxelgrid.shape[i] - voxelgrid_target_shape[i]) // 2
        crop_start[i] = max(0, crop_start[i])
        crop_end[i] = target_shape[i] + crop_start[i]
    voxelgrid = voxelgrid[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]

    return voxelgrid


def render_voxelgrid(voxelgrid, title=None):
    """
    Renders a voxel-grid.
    """

    figsize = (5, 5)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    transformed_voxelgrid = np.flip(np.flip(voxelgrid, axis=2), axis=0)

    facecolors = np.zeros(transformed_voxelgrid.shape + (3,))
    for x, y, z in itertools.product(range(transformed_voxelgrid.shape[0]), range(transformed_voxelgrid.shape[1]), range(transformed_voxelgrid.shape[2])):
        color = (1.0 - y / 32)
        facecolors[x, y, z, 0] = color
        facecolors[x, y, z, 1] = color
        facecolors[x, y, z, 2] = color

    ax.voxels(transformed_voxelgrid, facecolors=facecolors, edgecolor="k")

    if title != None:
        plt.title(title)

    plt.show()
    plt.close()


def get_datetime_string():
    """
    Returns a datetime string.
    """

    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def get_latest_preprocessed_dataset(path=".", filter=""):
    """
    Retrieves the path of the latest preprocessed dataset. Takes into account a filter.
    """
    glob_search_path = os.path.join(path, "*.p")
    paths = [x for x in glob.glob(glob_search_path) if filter in x]
    if len(paths) == 0:
        raise Exception("No datasets found for filter " + filter + " at path " + os.path.abspath(path))
    return sorted(paths)[-1]


def get_latest_model(path=".", filter=""):
    """
    Retrieves the path of the latest preprocessed dataset. Takes into account a filter.
    """
    glob_search_path = os.path.join(path, "*.h5")
    paths = [x for x in glob.glob(glob_search_path) if filter in x]
    if len(paths) == 0:
        raise Exception("No models found for filter " + filter + " at path " + os.path.abspath(path))
    return sorted(paths)[-1]


def pointcloud_to_rgb_map(original_pointcloud, target_width=512, target_height=512, scale_factor=1.5, axis="vertical"):
    '''
    Maps a pointcloud to a RGB-image. Stores height, density and intensity as separate channels.
    '''
    if axis=="horizontal":
        
        # Transform to pixel-space.
        scale = np.array([target_width / scale_factor, target_width / scale_factor, target_width / scale_factor, target_width / scale_factor]) # TODO is this okay?
        translate = np.array([target_width / 2, target_height / 2, 0.0, 0.0])
        pointcloud = original_pointcloud * scale + translate

        # Crop the pointcloud.
        crop_mask = np.where(
            (pointcloud[:, 0] >= 0) & 
            (pointcloud[:, 0] < target_width) & 
            (pointcloud[:, 1] >= 0) & 
            (pointcloud[:, 1] < target_height))
        pointcloud = pointcloud[crop_mask]

        # Get indices and counts.
        _, indices, counts = np.unique(pointcloud[:,0:2], axis=0, return_index=True, return_counts = True)

        # Get unique pixel coordinates.
        pixel_coordinates = np.int_(np.array([[x, y] for x, y, _, _ in pointcloud[indices]]))

        # Create the height map.
        heights = pointcloud[indices][:,2]
        height_map = np.zeros((target_width, target_height))
        height_map[pixel_coordinates[:,0], pixel_coordinates[:,1]] = heights
        height_map /= target_width

        # Create the density map.
        densities = np.minimum(1.0, np.log(counts + 1)/np.log(64))
        density_map = np.zeros((target_width, target_height))
        density_map[pixel_coordinates[:,0], pixel_coordinates[:,1]] = densities

        # Create the intensity map.
        intensities = pointcloud[indices][:,3]
        intensity_map = np.zeros((target_width, target_height))
        intensity_map[pixel_coordinates[:,0], pixel_coordinates[:,1]] = intensities
        intensity_map /= target_width

        # Compose the RGB-map.
        rgb_map = np.zeros((target_width, target_height, 3))
        rgb_map[:,:,0] = height_map 
        rgb_map[:,:,1] = density_map
        rgb_map[:,:,2] = intensity_map

        return rgb_map
    
    elif axis=="vertical":
        
        # Transform to pixel-space.
        scale = np.array([target_width / scale_factor, target_width / scale_factor, target_width / scale_factor, target_width / scale_factor]) 
        # TODO is this okay?
        translate = np.array([target_height / 2, target_width / 2, -target_width/3, 0.0])
        pointcloud = original_pointcloud * scale + translate

        # Crop the pointcloud.
        crop_mask = np.where(
            (pointcloud[:, 1] >= 0) & 
            (pointcloud[:, 1] < target_width)
            & 
            (pointcloud[:, 2] >= 0) & 
            (pointcloud[:, 2] < target_height)
        )
        pointcloud = pointcloud[crop_mask]
        
        # Get indices and counts.
        _, indices, counts = np.unique(pointcloud[:,[1, 2]], axis=0, return_index=True, return_counts = True)

        # Get unique pixel coordinates.
        pixel_coordinates = np.int_(np.array([[-x, y] for _, y, x, _ in pointcloud[indices]]))

        # Create the height map.
        heights = pointcloud[indices][:,0]
        height_map = np.zeros((target_width, target_height))
        height_map[pixel_coordinates[:,0], pixel_coordinates[:,1]] = heights
        height_map /= target_width

        # Create the density map.
        densities = np.minimum(1.0, np.log(counts + 1)/np.log(64))
        density_map = np.zeros((target_width, target_height))
        density_map[pixel_coordinates[:,0], pixel_coordinates[:,1]] = densities

        # Create the intensity map.
        intensities = pointcloud[indices][:,3]
        intensity_map = np.zeros((target_width, target_height))
        intensity_map[pixel_coordinates[:,0], pixel_coordinates[:,1]] = intensities
        intensity_map /= target_width

        # Compose the RGB-map.
        rgb_map = np.zeros((target_width, target_height, 3))
        rgb_map[:,:,0] = height_map 
        rgb_map[:,:,1] = density_map
        rgb_map[:,:,2] = intensity_map

        return rgb_map
    
    else:
        raise Exception("Unknown axis: " + axis)

def show_rgb_map(rgb_map):
    '''
    Renders a RGB-map.
    '''
    
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.subplot(2, 2, 1)
    show_rgb_map_channel(rgb_map[::-1,:,:], "RGB")
    plt.subplot(2, 2, 2)
    show_rgb_map_channel(rgb_map[::-1,:,0], "Height", cmap="gray")
    plt.subplot(2, 2, 3)
    show_rgb_map_channel(rgb_map[::-1,:,1], "Density", cmap="gray")    
    plt.subplot(2, 2, 4)
    show_rgb_map_channel(rgb_map[::-1,:,2], "Intensity", cmap="gray")
    plt.show()
    return plt.gcf(), plt.gca()

def show_rgb_map_channel(data, title, cmap=None):
    '''
    Renders a channel of a RGB-map.
    '''
    
    fig = plt.imshow(data, cmap=cmap)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    
def find_timestamps_of_trained_models(root_path):
    '''
    Extracts the timestamps. Different timestamps will represent different models/trainings.
    '''
    all_paths = find_all_history_paths(root_path)
    date_times = []
    for path in all_paths:
        split = path.split("/")[-1].split("-")
        date = split[0]
        time = split[1]
        date_time = date + "-" + time
        date_times.append(date_time)
    date_times = sorted(list(set(date_times)))
    return date_times


def plot_date_times(date_times, all_history_paths, start_index, end_index = 100090, key_suffix=None):
    for date_time in date_times:

        # Load all histories for date-time.
        history_paths = [history_path for history_path in all_history_paths if date_time in history_path]
        histories = []
        for history_path in history_paths:
            history = pickle.load(open(history_path, "rb"))
            histories.append(history)

        # Plot the histories.
        for history, history_path in zip(histories, history_paths):
            split = history_path.split("/")[-1].split("-")
            for key in history.keys():
                if key_suffix != None and key_suffix in key:
                    plt.plot(history[key][start_index:end_index], label=key + " " + split[2] + " " + date_time)
    plt.legend()
    plt.show()
    plt.close()
    
def get_mean_error(date_times, all_history_paths, start_index, end_index = 100090, key_suffix=None):
    for date_time in date_times:

        # Load all histories for date-time.
        history_paths = [history_path for history_path in all_history_paths if date_time in history_path]
        histories = []
        for history_path in history_paths:
            history = pickle.load(open(history_path, "rb"))
            histories.append(history)

        # Print the average error of each model.
        for history, history_path in zip(histories, history_paths):
            split = history_path.split("/")[-1].split("-")
            for key in history.keys():
                if key_suffix != None and key_suffix in key:
                    lst = history[key][start_index:end_index]
                    avg_error = sum(lst) / len(lst)
                    print("Avg " + key + " " + split[2] + " " + date_time + " between epoch " + str(start_index) + " and " + str(end_index) + " = " + str(avg_error))

    
def find_all_history_paths(root_path):
    all_paths = glob.glob(os.path.join(root_path, "*.p"))
    history_paths = [path for path in all_paths if "history" in path]
    return history_paths