# Goal
Create a few shot pipeline as outlined in De Nardin et al's. paper efficient...

# Sub tasks
## Model training pipeline:

## Patchgeneration
### Create a class which stores patches of images
Purpose: Our model will not process entire images but small parts of an image. To later use the model, we would need some method of rearranging the patches into their original state before outputting them. This class should serve that purpose.

#### Should consist of:
* x: Describes where along the uniform grid the patch was generated from horisontally
* y: Describes where along the uniform grid the patch was generated from vertically
* img_data: Contains the pixel values of a patch
* ID: Tells us from which document and which page the patch is from.
* ground_truth: if applicable, contains the ground truth from the training set for this patch
* prediction: the output of the model for this patch.

### Create a function which creates a grid of patches
This function generates all necessary patches and returns a list of patches. For each page create $grid\_size^2$ amount of subsamples.

Parameters:
* pages: contains the manuscript pages.
* grid_size (int): used to create the grid

### Create a function which randomly generates these patches accross an image
Given a set of pages generate N randomly sampled patches for each page. This is used as part of dataaugmentation to boost number of samples. It does not need ID, x or y.

Parameters:
* pages: -||-
* n: (int) number of patches

Returns: n*pages patches in a list

### Recreate image:
Given a set of patches, create the full page based off patches, including prediction and original.

Parameters:
* patches (list\<abovementioned class\>)

Returns: a list of pages.
