# Img2Mesh
*3D point cloud estimation from silhouette images*
Developed for the Data Science courseware at University of São Paulo

The project is separated in 3 notebooks:
- Preprocessing [[link](https://github.com/jvaaguiar/Img2Mesh/blob/master/Img2Mesh_Preprocessing.ipynb)]
- Trainning [[link](https://github.com/jvaaguiar/Img2Mesh/blob/master/Img2Mesh_Training.ipynb)]
- Testing [[link](https://github.com/jvaaguiar/Img2Mesh/blob/master/Img2Mesh_Testing.ipynb)]

The notebooks are runnable with **Google Colaboratory**

**ALL NOTEBOOKS SHOULD BE RUN WITH GPU ACCELERATION**
and require **Google Drive** autorization

The data required in in the folder
https://drive.google.com/drive/folders/1fQcbqfRWepWrMfw20UfxqEW3FABt7bt3?usp=sharing
Add this folder to your Google Drive (Shared with me > Img2Mesh)


# 1. Preprocessing
- INPUT  : ModelNet dataset
- OUTPUT : .tar.gz of each class with objects and rendered images as .npy

- In the section **"[!] Preprocessing config (in/out)"** all paths to the data are defined.
- The compressed .tar.gz files go to
**tarsroot = '/content/drive/Shared drives/Img2Mesh/dataset/tars/test_new_tars/'**


# 2. Training
- INPUT  : .tar.gz generated in preprocessing + hidden_size + out_vertices
- OUTPUT : trained model checkpoint

- In the section **"[!] Training config (in/out)"** all paths to the input data are defined,
to the output checkpoints and the **fully connected** layers {hidden_size, out_vertices}

- The convolutionnal feature extraction network gives inputs for 3 fully connected layers:
{features} -> hidden_size -> hidden_size / 2 -> out_verts -> {output}


# 3. Testing
- INPUT  : .tar.gz + checkpoint + bmp files (400x400)
- OUTPUT : scatter plots + .obj of point clouds

- The section **"[!] Testing config (in/out)"** has all the paths to input .tar.gz
and the **testing class**

- The section **"Test"** shows the scatter plots of the predicted object selected with the variable **id**.
- The generated .obj files are automatically downloaded

- The section **"Test with a custom 400x400 bmp"** shows an usage example for .bmp files.
Note that it must have exactly 400x400.

# Notes
An alternative trainning notebook featuring a normal loss optimization for
mesh fitting using faces normals is available [[link](https://github.com/jvaaguiar/Img2Mesh/blob/master/Img2Mesh_Training_(Normal_Loss).ipynb)]
The results from this technique were not satisfatory.

The presentation folder [[link](https://github.com/jvaaguiar/Img2Mesh/tree/master/presentation)] has the presentation further information about the solution design process.