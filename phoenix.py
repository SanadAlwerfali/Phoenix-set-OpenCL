#!/usr/bin/env python
import pyopencl as cl
import numpy as np
from PIL import Image
import os

WIDTH = 1024
HEIGHT = 1024
NUM_ITERATIONS = 128

# If the environment variable PYOPENCL_CTX is set, pyopencl will automatically choose a device '0' meaning that we will be using NVIDIA CUDA GPGPU
os.environ["PYOPENCL_CTX"] = "0"


# this kernel calculates in parallel for each point c of a complex
# grid, if the sequence z_{n} = (z_{n-1})^2 + c_real + p*z_{n-2} diverges 
# for n -> infinity. For each point, we store 0 in buffer out if
# if it does not diverge (i.e belongs to the phoenix set) or the
# number of iterations after which we are sure that the sequence diverges.
KernelSource = '''
__kernel void phoenix(const int WIDTH,
                            const int HEIGHT,
                            const int NUM_ITERATIONS,
                            __global int *out,
                            const double c_real,
                            const double c_imag,
                            const double p
                           )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }

    double z_real = (y * 2.4 / (WIDTH - 1)) - 1.2; 
    double z_imag = (x * 3.0 / (HEIGHT - 1)) - 1.5;

    double tmp_z_real, tmp_z_imag;
    double prev_z_real = z_real, prev_z_imag = z_imag;
    double prev_prev_z_real = z_real, prev_prev_z_imag = z_imag;
    double norm;

    int divergence_at = 0;
    for (int i = 2; i <= NUM_ITERATIONS; i++) {
        tmp_z_real = z_real * z_real - z_imag * z_imag + c_real + p * prev_z_real;
        tmp_z_imag = 2 * z_real * z_imag + c_imag + p * prev_z_imag;

        prev_z_real = z_real;
        prev_z_imag = z_imag;

        z_real = tmp_z_real;
        z_imag = tmp_z_imag;

        // if norm > 4.0, we can be sure that the sequence diverges
        norm = z_real * z_real + z_imag * z_imag;
        if (norm > 4.0) {
            divergence_at = i;
            break;
        }
    }

    out[y * WIDTH + x] = divergence_at;
}
'''

def color_map(iterations):
    # Color mapping function based on the number of iterations
    if iterations == 0:
        return (0, 0, 0)  # Background color (Black)
    elif 1 <= iterations <= 32:
        return (255, 0, 0)  # Foreground color 1 (Red)
    elif 33 <= iterations <= 64:
        return (0, 255, 0)  # Foreground color 2 (Green)
    else:
        return (0, 0, 255)  # Foreground color 3 (Blue)

def main():
     # setup open-cl queue and context
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)
    
    # setup an uninitialized buffer on the device
    d_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                      WIDTH * HEIGHT * np.dtype(np.int32).itemsize)

    # compile the Phoenix set kernel
    program = cl.Program(context, KernelSource).build()
    phoenix = program.phoenix
    phoenix.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, np.float64, np.float64, np.float64])

    # Set the values for c and p in the Phoenix set equation
    c_real = 0.56667
    c_imag = 0
    p = -0.5

    # run the Phoenix set kernel
    globalrange = (WIDTH, HEIGHT)
    localrange = None
    phoenix(queue, globalrange, localrange, WIDTH, HEIGHT, NUM_ITERATIONS, d_out, c_real, c_imag, p)
    queue.finish()

    # copy the buffer from the device to the host
    h_out = np.empty((HEIGHT, WIDTH), dtype=np.int32)
    cl.enqueue_copy(queue, h_out, d_out)
    
    #flip the image to make it look like the one on the assignment
    h_out = np.flipud(h_out) 
    # create the image using color mapping
    img = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = img.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            iterations = h_out[y, x]
            color = color_map(iterations)
            pixels[x, y] = color

    # save the image
    img.save("phoenix_image.png")
    img.show()

if __name__ == "__main__":
    main()