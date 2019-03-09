[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run?template=https://github.com/irhumshafkat/inverse-face-recognition) 
# inverse-face-recognition

Repository containing all code needed to replicate the experiments from this [article](https://blog.floydhub.com/inverting-facial-recognition-models/). Face recognition model from dlib, face generator implemented in PyTorch

### Running on FloydHub
 
Just click the button above, and you should be redirected to open a workspace on FloydHub, with all the default parameters (including the location of the dataset) being inferrred from the floyd.yml present in this repository

### Running locally

Clone this respository, download the required data from [here](https://www.floydhub.com/irhumshafkat/datasets/face-inversion-dataset/4), extract the contents to a folder `floyd/input/data` created inside the repository folder. 
 
## Data 

*Train/Validation/Test*: all images used in these sets were taken from the Labelled Faces in the Wild dataset, found [here](http://vis-www.cs.umass.edu/lfw/), and the splitting method is described in the article

*Visualization*: Images used for visualization purposes in the article were taken from Wikimedia commons, with the source and additional information being available in the URLs in the `sources.txt` file in the `vis` folder of both the 128dim and 1024dim version of the dataset
