import argparse
from tqdm import tqdm
import numpy as np
import torch
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import LatentOptimizer
from models.image_to_latent import ImageToLatent
from models.losses import LatentLoss
from utilities.hooks import GeneratedImageHook
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path

parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
parser.add_argument("image_path", help="Filepath of the image to be encoded.")
parser.add_argument("dlatent_path", help="Filepath to save the dlatent (WP) at.")

parser.add_argument("--save_optimized_image", default=False, help="Whether or not to save the image created with the optimized latents.", type=bool)
parser.add_argument("--optimized_image_path", default="optimized.png", help="The path to save the image created with the optimized latents.", type=str)
parser.add_argument("--video", default=False, help="Whether or not to save a video of the encoding process.", type=bool)
parser.add_argument("--video_path", default="video.avi", help="Where to save the video at.", type=str)
parser.add_argument("--save_frequency", default=10, help="How often to save the images to video. Smaller = Faster.", type=int)
parser.add_argument("--iterations", default=1000, help="Number of optimizations steps.", type=int)
parser.add_argument("--model_type", default="stylegan_ffhq", help="The model to use from InterFaceGAN repo.", type=str)
parser.add_argument("--learning_rate", default=1, help="Learning rate for SGD.", type=int)
parser.add_argument("--vgg_layer", default=12, help="The VGG network layer number to extract features from.", type=int)
parser.add_argument("--use_latent_finder", default=False, help="Whether or not to use a latent finder to find the starting latents to optimize from.", type=bool)
parser.add_argument("--image_to_latent_path", default="image_to_latent.pt", help="The path to the .pt (Pytorch) latent finder model.", type=str)

args, other = parser.parse_known_args()

def optimize_latents():
    print("Optimizing Latents.")
    synthesizer = StyleGANGenerator(args.model_type).model.synthesis
    latent_optimizer = LatentOptimizer(synthesizer, args.vgg_layer)

    # Optimize only the dlatents.
    for param in latent_optimizer.parameters():
        param.requires_grad_(False)
    
    if args.video or args.save_optimized_image:
        # Hook, saves an image during optimization to be used to create video.
        generated_image_hook = GeneratedImageHook(latent_optimizer.post_synthesis_processing, args.save_frequency)

    reference_image = load_images([args.image_path])
    reference_image = torch.from_numpy(reference_image).cuda()
    reference_image = latent_optimizer.vgg_processing(reference_image)
    reference_features = latent_optimizer.vgg16(reference_image).detach()
    reference_image = reference_image.detach()
    
    if args.use_latent_finder:
        image_to_latent = ImageToLatent().cuda()
        image_to_latent.load_state_dict(torch.load(args.image_to_latent_path))
        image_to_latent.eval()

        latents_to_be_optimized = image_to_latent(reference_image)
        latents_to_be_optimized = latents_to_be_optimized.detach().cuda().requires_grad_(True)
    else:
        latents_to_be_optimized = torch.zeros((1,18,512)).cuda().requires_grad_(True)

    criterion = LatentLoss()
    optimizer = torch.optim.SGD([latents_to_be_optimized], lr=args.learning_rate)

    progress_bar = tqdm(range(args.iterations))
    for step in progress_bar:
        optimizer.zero_grad()

        generated_image_features = latent_optimizer(latents_to_be_optimized)
        
        loss = criterion(generated_image_features, reference_features)
        loss.backward()
        loss = loss.item()

        optimizer.step()
        progress_bar.set_description("Step: {}, Loss: {}".format(step, loss))
    
    optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
    np.save(args.dlatent_path, optimized_dlatents)

    if args.video:
        images_to_video(generated_image_hook.get_images(), args.video_path)
    if args.save_optimized_image:
        save_image(generated_image_hook.last_image, args.optimized_image_path)

def main():
    assert(validate_path(args.image_path, "r"))
    assert(validate_path(args.dlatent_path, "w"))
    assert(1 <= args.vgg_layer <= 16)
    if args.video: assert(validate_path(args.video_path, "w"))
    if args.save_optimized_image: assert(validate_path(args.optimized_image_path, "w"))
    if args.use_latent_finder: assert(validate_path(args.image_to_latent_path, "r"))
    
    optimize_latents()

if __name__ == "__main__":
    main()


    
    


