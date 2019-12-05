# Helper

A small abstract and guide for the work done in this repo.

## 0 - Dataset

The dataset was needed for the beginning of the testing phase. Given that there were no available datasets that contained images and their respective 3D models, it was necessary that they were generated using 10 models of fruits and deformations with RIG techniques.

- Synthetic Dataset 
  - Data were generated artificially (Maya + Python)
  - RIG to deform randomly.
    - Renders 6 images per fruit (several angles).
  - Rendered with lights, which makes the frequence spectre well distributed.
  - The dataset ends up being kind of a failure because it is as if it is been applied noise into the base model instead of generating several different objects.
  - The testing overfitted the base models, which is bad.
  - Data were passed into TensorFlow.

## 1 - Fruits + MLP

​	The naive solution would be to apply PCA transformation and then send the information into an MLP. But that didn't work very well because the models were highly non-linear, making the PCA (almost) useless since it didn't reduce the dimension by a significant margin.

​	Another problem is that during the training we needed an error measure. As a beginning, the Minimum Squared Error was used, but that proved itself as a bad alternative, since the vertices needed to be number in the same order in different figures so that the distance was represented correctly. For example, if we had two overlapping triangles but with different numbering of the vertices, they would be said to be very different, but they're not (because they're overlapping).



- Applied PCA
  - Worked well with one or two fruits, after that it was almost useless.
- MLP
  - The distance measure showed itself to be a problem
    - Need to search for a new distance measurement.
- Dataset
  - Bad dataset in the end
  - Necessary to search for another dataset

## 2 - ModelNet + Tensorflow

- Modelnet
  - Real dataset
  - Made for classification and not regression
  - Receives OFF file, which contains the positions of the vertices and polygon faces
    - Faces: List of the indexes of the vertices.
    - Need to convert OFF to OBJ file for compatibility reasons.
    - Using python render (pyrender) to generate 6 images
      - Faces are swapped (because of the conversion).
      - An alternative was to use only silhouettes, not using the faces information.
    - Npy file with
      - List of vertices.
      - List of faces.
      - 6 images (silhouettes).
- TensorFlow with custom optimization
  - The same architecture was not tried again (PCA + MLP)
    - PCA works well on highly linear systems, since the models are not one of those, this approach was skipped.
  - Chamfer loss as a cost function
  - VGG-16 based architecture
    - If trained with only airplane models the result was always the same airplane (overfit happened).
    - With 3 classes the result was just noise for any input.
      - The convolutional network was giving a constant input to the MLP instead of giving features of the image.
    - 9 layers CNN/Maxpool
      - In one of the articles that this work was based the images were well rendered, that means that the potency spectrum was well distributed.
      - The filter was supposed to gather all of the information, but the network was killing the high frequency of the images.
        - CNN + Maxpool works as a low-pass filter.
      - It was needed a filter that allowed medium/high frequencies, so it was used a smaller number of layers.
- Generating a mesh didn't work
  - It was needed a way to connect all of the obtained dots.
  - Convex Hull was a first approach, but the models weren't convex.
    - Concave Hull was the alternative
      - Holes were generated in the models, resulting in bad ones.
  - Another option was to regress pixels and faces.
    - Mapping initial vertices of the exit of the neural network in a icoshedro, since it has a high degree of average node connections
      - Face mapping is half-made in 3d modelling softwares.
    - Normal loss:
      - For each dot, get the normals of the faces and calculate the average.
        - Made using a dot product between the vectors that are being compared. The bigger the more they look alike.
  - Weight was given to Chamfer loss and Normal Loss (Complementing each other)
    - Chamfer loss with larger weight gives better results.
    - Normal loss is not a concave function, that causes spikes during the training and it doesn't give a good final result.
