README

Evyatar Michlis

installation:
1. I create conda env - conda create -n detectron_env python =3.97
2. activate the env : conda activate detectron_env
3. packeges : conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install cython
pip install opencv-python
    
4. clone the detectron2  git -git clone https://github.com/facebookresearch/detectron2.git
5. pip install -e  - inside the detectron2 folder
make sure the detectron2 folder is the same folder as my project


project:

1. detector.py - my project, in the file the class Detector have an __init__ function for initialize the model parametr.
I used detectron2 PanopticSegmentation for this project. (found the model detectron2 documentation)
the class need the pathes for the dataset and for the output file

2. crop_sky function - is my main function, the function use the model to find segmentation for the image, then it search for the class id of the sky (40) in the segmentation the model found, then it mask the sky part and crop all the pixel above the sky line.

3. crop_sky_from_data_set - use crop_sky to crop entrie folder from the sky.


sky-datset.zip - data set before with sky
croped_images - data set without sky
