# Breadth-first search to grow a cell while limiting to cytoplasm.

from queue import Queue
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)

# Load the image data using your own method
# ---> insert code here <---
img = ti.tools.image.imread('sample_em2d.png')
print(f'the loaded image is {img.shape}')
width, height = 761, 516
start_pixel = (167, 309)
end_pixel = (588, 98)

# Cost to reach each pixel
cost = ti.field(dtype=ti.i32, shape=(width, height))

# The minimum intensity of pixels from start to each pixel.
min_int = ti.field(dtype=ti.u16, shape=(width, height))

# The image to display where we will add data to analyze algorithm
pixels = ti.field(dtype=ti.u8, shape=(width, height, 3))   # for image data
display = ti.field(dtype=ti.u8, shape=(width, height, 3))  # for visualization

# Load the image data into the pixels field
@ti.kernel
def load_pixels(image: ti.types.ndarray()):
    for i, j, k in pixels:
        if k < 3:
            pixels[i, j, k] = image[i, j, k]
            display[i, j, k] = image[i, j, k]
load_pixels(np.ascontiguousarray(img))
print("done loading pixels")

# Calculate the neighbors for any given pixel in the form of a iterator function
def neighborhood(pixel_coord):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i != 0 or j != 0:
                x = pixel_coord[0] + i
                y = pixel_coord[1] + j
                if x >= 0 and x < width and y >= 0 and y < height:
                    neighbors.append((x, y))
    for neighbor in neighbors:
        yield neighbor

# Paint the pixels that have been visited red
@ti.kernel
def paint_visited_green():
    for i, j in min_int:
        if display[i, j, 1] != display[i, j, 0]:
            display[i, j, 1] = ti.uint8(255)

# Display the original image with path inserted between start and end pixels.
gui = ti.GUI('Best Path', res=(width, height))

# Breadth-first search, but we'd want to move toward Dijkstra's algorithm.
# Calculates the cost of through any pixel in the whole image.
frontier = Queue()
frontier.put(start_pixel)
reached = set()
reached.add(start_pixel)

start_x, start_y = start_pixel
min_int[start_pixel] = pixels[start_x, start_y, 0]
print('starting intensity: ', pixels[start_x, start_y, 0])

# This is a manually set threshold where we assume membranes and other
# "walls" are darker than this intensity.  You can modify it and see if
# it's too low, the growing will leak out of the starting cell.
cytoplasm_threshold = 130

checked = 0
painted_visited = False
while gui.running:
    while not frontier.empty():
        current_pixel = frontier.get()
        if current_pixel == end_pixel:
            print('found end pixel!')
            break

        if checked % 100 == 0:
            gui.set_image(display)
            gui.show()

        for next_pixel in neighborhood(current_pixel):
            if next_pixel not in reached:
                checked += 1
                if checked % 10000 == 0:
                    print(f'checked {checked} / {width * height} pixels')

                # get intensity of this pixel
                next_x, next_y = next_pixel
                next_intensity = pixels[next_x, next_y, 0]

                # only grow through cytoplasm (use threshold)
                if next_intensity < cytoplasm_threshold:
                    continue

                frontier.put(next_pixel)
                reached.add(next_pixel)

                # Path cost is the worst (minimum) intensity along path.
                min_int[next_pixel] = min(min_int[current_pixel], next_intensity)
                display[next_x, next_y, 1] = min_int[next_pixel]
    
    # Signal we are done by painting path
    if not painted_visited:
        paint_visited_green()
        painted_visited = True
    gui.set_image(display)
    gui.show()
