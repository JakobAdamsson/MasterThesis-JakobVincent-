import pandas as pd

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from IPython.display import display
import os
import numpy as np


class Patch:
    def __init__(self, x: int, y: int, image: np.ndarray, ground_truth: np.ndarray):

        self.x = x

        self.y = y
        self.img = image
        self.gt = ground_truth

    def __repr__(self):

        return f"(x={self.x}, y={self.y})"


class Page:
    def __init__(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        name: str,
        square_size: int = 832,
    ):
        self.img = image
        self.gt = ground_truth
        self.name = name
        self.grid = self.divide_image_into_square_patches(square_size)

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
        self, image_path: str = None, xml_path: str = None, pixel_gt_path: str = None
    ):
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
            colors = {"comment": 1, "body": 2, "decoration": 3}
            filled_polygon_img = Image.new("L", image.size)
            draw_filled_polygon = ImageDraw.Draw(filled_polygon_img)
            for group in coordinates_by_group.keys():
                for coords in coordinates_by_group[group]:
                    polygon_coordinates = coords

                    draw_filled_polygon.polygon(
                        polygon_coordinates, outline=None, fill=colors[group]
                    )
            gt = np.array(filled_polygon_img)
            mask = np.array(pixel_gt)
            mask = mask[:, :, 0]  # All information is stored in the red colour channel.
            non_zero_intersection = (gt != 0) & (mask != 0)
            gt[non_zero_intersection] = 0
            return Page(
                np.array(image),
                np.array(gt),
                image_path[image_path.rfind("/") + 1 : image_path.find(".")],
            )

    def __repr__(self):
        return f"Page={self.name}"
