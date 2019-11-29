# StyleGAN Encoder - Pytorch Implementation
| Reference Image  | Latent Optimization  | Gender Transformation  | Pose Transformation  |
|---|---|---|---|
| <img src="assets/images/test_01/test_01.png" width="256px" height="200px">  | <img src="assets/images/test_01/test_01_optimization.gif" width="256px" height="200px">  | <img src="assets/images/test_01/gender/test_01_w_to_m.gif" width="256px" height="200px">  | <img src="assets/images/test_01/pose/test_01_pose.gif" width="256px" height="200px">  |
|Reference Image | Age Transformation | Gender Transformation | Glasses Transformation |
| <img src="assets/images/test_02/test_02.jpg" width="256px" height="200px">  | <img src="assets/images/test_02/age/test_02_age.gif" width="256px" height="200px">  | <img src="assets/images/test_02/gender/test_02_gender.gif" width="256px" height="200px">  | <img src="assets/images/test_02/glasses/test_02_glasses.gif" width="256px" height="200px">  |

## Usage
Take an image of a face you'd like to modify and align the face by using the align_image.py script.

```bash
python align_image.py image.jpg aligned_image.jpg

```

Then find the latents for the aligned face by using the encode_image.py script.
```bash
python encode_image.py
  aligned_image.jpg
  dlatents.npy # The filepath to save the latents at.
  --save_optimized_image true
```

The script will generate a numpy array file with the latents that can then be passed to the edit.py script located in the InterFaceGAN repo. Edit an image by running the edit.py script.
```bash
python InterFaceGAN/edit.py
  -m stylegan_ffhq
  -o results
  -b InterFaceGAN/boundaries/stylegan_ffhq_pose_boundary.npy # Use any of the boundaries found in the InterFaceGAN repo.
  -i dlatents.npy
  -s WP
  --steps 20
```
The resulting script will modify the latents and correspondingly the aligned face with the boundary that you select (pose in the above example). It will save all of the transformed images in the -o directory (./results in the above example).

## The Image To Latent Model
The process of optimizing the latents with strictly just the features extracted by the Resnet model can be timely and possibly prone to local minima. To combat this problem, we can use another model thats sole goal is to predict the latents of an image. This gives the latent optimizer model a better initilization point to optimize from and helps reduce the amount of time needed for optimization and the likelyhood of getting stuck in a far away minima.

Here you can see the the images generated with the predicted latents from the Image To latent Model.
<img src="assets/images/image_to_latent_predictions.png">

### Usage
The encode_image.py script by default does not use the Image To Latent model, but you can activate it by specifiying the following params when running encode_image.py. Without using an Image To Latent model the encode_image.py script defaults to optimize latents initialized with all zeros.
```bash
python encode_image.py
  aligned_image.jpg
  dlatents.npy
  --use_latent_finder true # Activates model.
  --image_to_latent_path ./image_to_latent.pt # Specifies path to model.
```

### Training
All of the training is located in the train_image_to_latent_model.ipynb notebook. To generated a dataset use the following command.
```bash
python InterFaceGAN/generate_data.py
  -m stylegan_ffhq
  -o dataset_directory
  -n 50000
  -s WP
```
This will populate a directory at ./dataset_directory with 50,000 generated faces and a numpy array file called wp.npy. You can then load these into the notebook to train a new model. Using more than 50,000 will train a better latent predictor.
