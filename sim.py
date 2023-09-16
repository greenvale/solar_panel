import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from PIL import Image
import cv2
import sim
import math

vertex_src = """

# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texture;

uniform mat4 rotation;

out vec3 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = rotation * vec4(a_position, 1.0);
    v_color = a_color;
    v_texture = a_texture;

    v_texture = 1 - a_texture;                        //Flips the texture vertically and horizontally
    //v_texture = vec2(a_texture.s, 1 - a_texture.t);   //Flips the texture vertically
}

"""


fragment_src = """

#version 330

in vec3 v_color;
in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture);  // * vec4(v_color, 1.0f);
}

"""


class Environment:

    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices
        self.pitch = 20
        self.azi = 0
    
    def set_image(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.img_data = image.convert("RGBA").tobytes()
        self.img_width = image.width
        self.img_height = image.height
    
    def setup(self, window_size=500):
        #glfw callback functions
        def window_resize(window, width, height):
            glViewport(0, 0, width, height)

        #initialising glfw library
        if not glfw.init():
            raise Exception("glfw cannot be initialised")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        #creating the window
        self.window = glfw.create_window(window_size, window_size, "Cell Logit Calculator", None, None)

        #Check if window was created
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window cannot be created!")

        glfw.set_window_pos(self.window, 400, 200)
        glfw.set_window_size_callback(self.window, window_resize)
        glfw.make_context_current(self.window)

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        #Vertex array object
        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        IBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, self.vertices.itemsize * 8, ctypes.c_void_p(24))

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glUseProgram(shader)
        glClearColor(0, 0, 0, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.rotation_loc = glGetUniformLocation(shader, "rotation")

    def __call__(self):
        self.setup()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.img_width, self.img_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, self.img_data)
                
            rot_x = pyrr.Matrix44.from_x_rotation(self.pitch * math.pi / 180)
            rot_y = pyrr.Matrix44.from_y_rotation(self.azi * math.pi / 180)
            glUniformMatrix4fv(self.rotation_loc, 1, GL_FALSE, pyrr.matrix44.multiply(rot_y, rot_x))
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
            
            #logits = calculate_logits()
            #logit_table[label, sample_n, (pitch-10)/2, azi/2] = logits

            #print("label {}, sample {}, pitch {}, azi {}".format(label, sample_n, pitch, azi))
            #print("Similarity: {}".format(logits[label] * 100))

            glfw.swap_buffers(self.window)

        glfw.terminate()