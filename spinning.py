import numpy as np
from sim import *

class Spinning():
    def __init__(self):
        # environment configuration
        self.D = 0.6
        self.vertices = [
            -self.D, 0, -self.D,   1.0, 0.0, 0.0,   0.0, 0.0,
            self.D, 0, -self.D,   0.0, 1.0, 0.0,   1.0, 0.0,
            self.D, 0, self.D,   0.0, 0.0, 1.0,   1.0, 1.0,
            -self.D, 0, self.D,   1.0, 1.0, 1.0,   0.0, 1.0,
            ]
        self.indices = [0, 1, 2, 0, 2, 3]
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.indices = np.array(self.indices, dtype=np.uint32)
                
        self.img_path = './SolarCell-defect-detect/data_BS/train/Break/90_cell0019.png'

        # initialise environment
        self.env = sim.Environment(self.vertices, self.indices)
        self.env.set_image(self.img_path)

    def __call__(self):
        self.env.setup()

        # loop
        while not glfw.window_should_close(self.env.window):
            glfw.poll_events()

            # update environment
            self.env.render()

            # save screenshot of viewport
            cam_width = 100
            if self.env.timestamp % 100 == 0:
                pixels = glReadPixels(0, 0, self.env.window_size, self.env.window_size, GL_RGBA, GL_FLOAT)
                pixels = pixels[:,:,0:3]
                pixels = cv2.resize(pixels, (cam_width, cam_width), interpolation=cv2.INTER_LINEAR)
                pixels = np.reshape(pixels, [1, cam_width, cam_width, 3])
                view = Image.fromarray((pixels[0]*255).astype(np.uint8))
                view.save('./SavedImages/Screenshot'+str(self.env.timestamp)+'.png')

            # calculate new pitch, new azi and new image path
            new_pitch = self.env.pitch 
            new_azi = self.env.azi + 0.1
            new_img_path = None
            
            # step environment variables
            self.env.pitch = new_pitch
            self.env.azi = new_azi
            if new_img_path:
                self.env.set_image(new_img_path)

            self.env.timestamp += 1

        glfw.terminate()

if __name__ == "__main__":
    print("Running spinning test")
    test = Spinning()
    test()