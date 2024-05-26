import pandas as pd

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from IPython.display import display
import os
import numpy as np
import random
from torchvision.transforms.functional import to_tensor, pil_to_tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import uniform_filter, gaussian_filter


class Patch:
    #Initializes the Patch object
    def __init__(self, x: int, y: int, image: np.ndarray, ground_truth: np.ndarray):

        self.x = x
        self.y = y
        self.img = image
        self.gt = ground_truth
        self.output = None

    def __repr__(self):

        return f"(x={self.x}, y={self.y})"


class Page:
    def __init__(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        ground_truth_precise: np.ndarray,
        name: str,
        patch_size: int = 832,
    ):
        self.img = image
        self.gt = ground_truth
        self.name = name
        self.grid = []
        self.precise_gt = ground_truth_precise
        self.coarse_segmentation = None
        self.refined_segmentation = None

        self.divide_image_into_square_patches(patch_size)

    def __repr__(self):
        return f"{self.name}"

    def __eq__(self, x):
        return self.name == x.name

    def divide_image_into_square_patches(self, patch_size):
        """
        Divides an image represented by a numpy array into square patches.

        Parameters:
        - patch_size: integer, the size of each square patch in pixels.

        Returns:
        - A list of Patches, each representing a square patch of the original image.
        """
        self.grid = []
        height, width, _ = self.img.shape

        # Calculate how many patches fit into the width and height
        patches_x = height // patch_size
        patches_y = width // patch_size
        # Iterate through the images and create the patches using the given size
        for i in range(patches_x):
            for j in range(patches_y):
                start_x = i * patch_size
                end_x = start_x + patch_size
                start_y = j * patch_size
                end_y = start_y + patch_size

                # Slice the image array to get the current patch and add it to the list
                content = self.img[start_x:end_x, start_y:end_y, :]
                gt = self.gt[start_x:end_x, start_y:end_y]
                new_patch = Patch(i, j, content, gt)
                self.grid.append(new_patch)

    @classmethod
    def from_file(
        self,
        image_path: str = None,
        xml_path: str = None,
        pixel_gt_path: str = None,
        patch_size: int = 832,
        resize: tuple = None,
    ):
        """
        Creates the page objects from the given file paths to the required files

        Parameters:
        - image_path (string): path to a file of a page scan
        - xml_path (string): path to the XML file with class segmentation
        - pixel_gt_path (string): path to the image containing pixel precise annotations
        - patch_size (int): The desired size of the patches
        - resize (tuple): The resolution to resize to

        Returns:
        - A Page object
        """
        if image_path and xml_path:
            with open(xml_path, "r") as file:
                xml_data = file.read()

            # Parse the XML using BeautifulSoup with 'lxml-xml' parser
            soup = BeautifulSoup(xml_data, "lxml-xml")

            # Extract Coords points for each matching TextLine
            coordinates_by_group = {"comment": [], "body": [], "decoration": []}
            search = {
                "comment": ["TextLine", "comment"],
                "body": ["TextLine", "textline"],
                "decoration": ["GraphicRegion", "region"],
            }
            for group in coordinates_by_group.keys():
                textlines_with_comment = soup.find_all(
                    search[group][0], {"id": lambda x: x and search[group][1] in x}
                )
                for textline in textlines_with_comment:
                    coords_points = textline.find("Coords")["points"]
                    coordinate_pairs = coords_points.split()
                    coordinates_list = [
                        tuple(map(int, pair.split(","))) for pair in coordinate_pairs
                    ]
                    coordinates_by_group[group].append(coordinates_list)
            image = Image.open(image_path)
            pixel_gt = Image.open(pixel_gt_path)

            # Enumerate each class with from 0 to 3
            colors = {"comment": 1, "body": 2, "decoration": 3}

            # Creates an empty image object with the same dimensions as the one provided to the method
            filled_polygon_img = Image.new("L", image.size)
            draw_filled_polygon = ImageDraw.Draw(filled_polygon_img)

            # Fills the empty images with the values for the respective class
            for group in coordinates_by_group.keys():
                for coords in coordinates_by_group[group]:
                    polygon_coordinates = coords

                    draw_filled_polygon.polygon(
                        polygon_coordinates, outline=None, fill=colors[group]
                    )

            # Converts the image into a numpy array and reformats it into one channel
            gt = np.array(filled_polygon_img)
            gt_precise = np.copy(gt)
            mask = np.array(pixel_gt)
            mask = mask[:, :, 0]  # All information is stored in the red colour channel.

            # Calculates the pixel precise  ground truth by taking the non zero intersection with pixel precise annotations
            non_zero_intersection = (gt != 0) & (mask != 0)
            gt_precise[non_zero_intersection] = 0

            if resize:
                image = image.resize(resize, Image.LANCZOS)
                gt = Image.fromarray(gt).resize(resize, Image.NEAREST)
                gt_precise = Image.fromarray(gt_precise).resize(resize, Image.NEAREST)
            return Page(
                np.array(image),
                np.array(gt),
                np.array(gt_precise),
                image_path[image_path.rfind("/") + 1 : image_path.find(".")],
                patch_size=patch_size,
            )

    def reconstruct_prediction(self):
        """
        Reconstructs the prediction into one images from its patches
        """

        # Gets the dimensions that the output image should have
        height, width = self.img.shape[:2]

        # Initializes an empty array with the specified height and width
        full_image = np.zeros((height, width), dtype=self.grid[0].output.dtype)

        # The patches are places in the empty array based on their coordinated
        for patch in self.grid:
            shape = patch.img.shape[0]
            full_image[
                patch.x * shape : (patch.x + 1) * shape,
                patch.y * shape : (patch.y + 1) * shape,
            ] = patch.output
        self.coarse_segmentation = full_image
        return full_image


def parse_images_from_folder(
    image_dir_path,
    xml_dir_path,
    pixel_dir_path,
    num_pages: int = 2,
    patch_size: int = 832,
    index: int  = -1,
    precise: bool = False,
    resize: tuple = None,
):
    """"
    This function creates a list of pages based on the specified paths to the folders containing the required files

    Args:
        image_files(str): Path to the folder containing the images with page scans
        xml_dir_path(str): Path to the folder containing the XML files with class segmentation
        pixel_dir_path(str): Path to the folder containing images with pixel-precise annotation
        num_pages(int): Sets the number of pages that should be collected the folder
        patch_size (int): The desired size of the patches
        resize (tuple): The resolution to resize to


    Returns:
        A list of Page objects
    """

    #Names of the files inside the folder
    xml_files = os.listdir(xml_dir_path)
    image_files = os.listdir(image_dir_path)
    pixel_files = os.listdir(pixel_dir_path)
    output = []
    prev_i = []

    # Selects "num_pages" of pages from the folders and converts them to a Page object
    i=index
    for _ in range(num_pages):
        if index == -1:
            i = random.randint(0, len(image_files) - 1)
            while i in prev_i:
                i = random.randint(0, len(image_files) - 1)
        prev_i.append(i)
        img_file_path = image_dir_path + "/" + image_files[i]
        xml_file_path = xml_dir_path + "/" + image_files[i][:-3] + "xml"
        pixel_gt_file_path = pixel_dir_path + "/" + image_files[i][:-3] + "png"
        page = Page.from_file(
            img_file_path,
            xml_file_path,
            pixel_gt_file_path,
            patch_size=patch_size,
            resize=resize,
        )
        output.append(page)
        i+=1
    return output


def generate_set(
    num_pages=2,
    manuscripts=["CS18", "CS863", "CB55"],
    patch_size: int = 832,
    resize: tuple = None,
    from_folders=[True, True, False],
    filepath = "",
    indices:int = -1
):
    """
    Generates a complete set with "num_pages" Page objects from each of the specified manuscripts.

    Args:
        num_pages(int): Sets the number of pages that should be collected the folder
        manuscripts(list): A list with the manuscripts to generate pages from
        patch_size(int): The desired size of the patches

    Returns:
        tuple: Containing training, validation and testing datasets.
    """
    data_dict = {}
    val_dict = {}
    test_dict = {}

    # Goes through every manuscripts and parses every required folder
    for manuscript in manuscripts:
        if from_folders[0]:

            data_dict[manuscript] = parse_images_from_folder(
                f"{filepath}/all/img-{manuscript}/img/training",
                f"{filepath}/all/PAGE-gt-{manuscript}/PAGE-gt/training",
                f"{filepath}/all/pixel-level-gt-{manuscript}/pixel-level-gt/training",
                num_pages,
                patch_size=patch_size,
                index=indices,
                resize=resize,
            )
        if from_folders[1]:
            val_dict[manuscript] = parse_images_from_folder(
                f"{filepath}/all/img-{manuscript}/img/validation",
                f"{filepath}/all/PAGE-gt-{manuscript}/PAGE-gt/validation",
                f"{filepath}/all/pixel-level-gt-{manuscript}/pixel-level-gt/validation",
                num_pages,
                patch_size=patch_size,
                index=indices,
                resize=resize,
            )
        if from_folders[2]:
            test_dict[manuscript] = parse_images_from_folder(
                f"{filepath}/all/img-{manuscript}/img/public-test",
                f"{filepath}/all/PAGE-gt-{manuscript}/PAGE-gt/public-test",
                f"{filepath}/all/pixel-level-gt-{manuscript}/pixel-level-gt/public-test",
                num_pages,
                patch_size=patch_size,
                index=indices,
                resize=resize,
            )
    return (
        [i for lst in data_dict.values() for i in lst],
        [i for lst in val_dict.values() for i in lst],
        [i for lst in test_dict.values() for i in lst],
    )

class PatchesDataset(Dataset):
    def get_crops(self):
        for page in self.pages:
            for patch in page.grid:
                self.base_patches.append(patch.img)
                self.base_labels.append(patch.gt)
                self.coords.append((patch.x, patch.y, page.name))

    def __init__(self, pages, transform=None):
        self.pages = pages
        self.transform = transform
        self.base_patches = []
        self.base_labels = []
        self.coords = []
        self.get_crops()

        self.patches = self.base_patches
        self.labels = self.base_labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = Image.fromarray(self.patches[index].astype("uint8"), "RGB")
        label = Image.fromarray(self.labels[index].astype("uint8"), "L")

        patch = self.transform(patch)
        label = pil_to_tensor(label).squeeze().long()
        return patch, label

    def random_patch_generator(self, num_patches):
        """
        Divides an image represented by a numpy array into square patches.

        Parameters:
        - patch_size: integer, the size of each square patch in pixels.

        Returns:
        - A list of Patches, each representing a square patch of the original image.
        """
        images = []
        labels = []

        patch_size = self.base_patches[0].shape[0]
        height, width, _ = self.pages[0].img.shape
        # Generates random  patches bases on the amount specified in num_patches
        for page in self.pages:
            for i in range(num_patches):
                start_x = random.randint(0, width - patch_size)
                start_y = random.randint(0, height - patch_size)
                end_x = start_x + patch_size
                end_y = start_y + patch_size

                content = page.img[start_y:end_y, start_x:end_x, :]
                gt = page.gt[start_y:end_y, start_x:end_x]
                labels.append(gt)
                images.append(content)
        self.patches = self.base_patches + images
        self.labels = self.base_labels + labels
def phansalkar(img, n=5, p=3, q=10, k=0.25, R=0.5):

    """
    An implementation of Phansalkar's method used for thresholding and binarization of an image.

    """


    img = np.asarray(img, dtype=np.float32) / 255.0  # Normalize to 0-1 range
    R = R  # Assuming the image is already normalized

    # Define the window size for local neighborhood
    w = n // 2

    # Pad the image to handle borders
    img_padded = np.pad(img, w, mode="reflect")

    # Compute the local mean and local standard deviation using uniform_filter
    mean = uniform_filter(img_padded, size=n)[w:-w, w:-w]
    mean_sq = uniform_filter(img_padded**2, size=n)[w:-w, w:-w]
    std = np.sqrt(mean_sq - mean**2)

    # Calculate the threshold (Phansalkar formula)
    ph = p * np.exp(-q * mean)
    threshold = mean * (1 + ph + k * ((std / R) - 1))

    # Apply the threshold
    imgN = np.where(img < threshold, 1, 0)

    return imgN * 255  # Convert back to 0-255 range


def refine_image(page: Page, function):
    """
    Args:
        page(Page): A page object that contains the coarse_segmentation to be refined
        function(function): a tresholding function that is used to create the refined image
    """
    base_image = page.img
    coarse_segmentation = page.coarse_segmentation

    base_image_BW = Image.fromarray(base_image).convert("L")
    thresh = function(np.array(base_image_BW))  # 0.05 51

    # Ensures that the values are binary
    binary = base_image_BW < thresh
    binary = binary.astype("uint8")

    # Calculates and stores the refined segmentation
    page.refined_segmentation = coarse_segmentation * binary
