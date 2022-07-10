# Using Deep Learning to Improve Detection and Decoding of Barcodes
##
![Barcode Image](https://github.com/cwang16/barcode/blob/main/pipeline.png) <br /> 
<br />
<br />
__The proposed pipline__ takes barcode images as input and performs  barcode detection, rotation and debluring using RDD (Zhong et al., 2020) and deblurGAN-v2 (Kupyn et al., 2019), respectively. The PyTorch implementation of RDD, available at https://github.com/Capino512/pytorch-rotation-decoupled-detector, was used for barcode detection and rotation (and produces horizontally-aligned bounding boxes). The PyTorch implementation of DeblurGAN-v2, available at https://github.com/VITA-Group/DeblurGANv2, was used to deblur the barcode images. The models were trained with approximately 2000 images, including images from sources such as Muenster, ArteLab (w/ AF), ArteLab (w/o AF), OpenFoodFacts, ZXing-ean13 and Skku inyong DB (see below). All images were annotated with  bounding boxes and rotation angles. We make available the datasets used for training and evaluating the models to enable further research on barcode detection and decoding. Specifically, we provide the image name/id and the corresponding source dataset, together with bounding box annotations and ground truth rotation angle, for all the images in our dataset. The specific images can be retrived from the original source datasets based on the image name/id. 
<br />
<br />
## Content of the Repository
__oriented_barcode_detection_dataset.zip__ contains the bounding box annotation, image name and source dataset for traning RDD model <br />
__barcode_image_debluring_dataset.zip__ contains the image name and source dataset for training deblurGANv2 model  <br />
__rotate.py__ is the scirpt to rotate barocde images  <br />

## Source Datasets:
Muenster: https://www.uni-muenster.de/PRIA/forschung/index.shtml and https://github.com/rohrlaf/SlaRle.js<br />
ArteLab (w/ AF) and ArteLab (w/o AF): http://artelab.dista.uninsubria.it/downloads.html<br />
OpenFoodFacts: https://github.com/openfoodfacts/openfoodfacts-ai/issues/15<br />
ZXing-ean13: https://github.com/zxing/zxing/tree/master/core/src/test/resources/blackbox<br />
Skku inyong DB: http://dspl.skku.ac.kr/home_course/data/barcode/skku_inyong_DB.zip<br />

