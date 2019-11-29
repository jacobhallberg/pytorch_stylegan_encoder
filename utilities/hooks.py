import numpy as np

class GeneratedImageHook:
    # Pytorch forward pass module hook.

    def __init__(self, module, every_n=10):
        self.generated_images = []
        self.count = 1
        self.every_n = every_n
        self.last_image = None

        self.hook = module.register_forward_hook(self.save_generated_image)

    def save_generated_image(self, module, input, output):
        image = output.detach().cpu().numpy()[0]
        if self.count % self.every_n == 0:
            self.generated_images.append(image)
            self.count = 0

        self.last_image = image
        self.count += 1

    def close(self):
        self.hook.remove()

    def get_images(self):
        return self.generated_images
