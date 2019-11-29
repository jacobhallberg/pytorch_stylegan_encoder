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

  
