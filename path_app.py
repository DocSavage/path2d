from queue import Queue
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)

# Load the image data using your own method
# ---> insert code here <---
img = ti.tools.image.imread('sample_em2d.png')
print(f'the loaded image is {img.shape}')
width, height = 516, 761
start_pixel = (167, 309)
end_pixel = (588, 98)

# Create two fields, one for the path calculation and one for the loaded image.
# You want to use ti.i32 for the cost field because you will be storing the
# cost of the path from the start pixel to each pixel in the cost field.
cost = ti.field(dtype=ti.i32, shape=(width, height))
pixels = ti.field(dtype=ti.u8, shape=(width, height))

# Load the image data into the pixels field
@ti.kernel
def load_pixels(image: ti.types.ndarray()):
    for i, j in pixels:
        pixels[i, j] = image[i, j, 0]  # For grayscale, we only need one channel.
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
                if 0 <= x < width and 0 <= y < height:
                    neighbors.append((x, y))
    for neighbor in neighbors:
        yield neighbor

for next_pixel in neighborhood(start_pixel):
    print(next_pixel)

# Breadth-first search, but we'd want to move toward Dijkstra's algorithm.
# Calculates the cost of through any pixel in the whole image.
frontier = Queue()
frontier.put(start_pixel)
cost[start_pixel] = 0
reached = set()
reached.add(start_pixel)

checked = 0
while not frontier.empty():
    current_pixel = frontier.get()
    for next_pixel in neighborhood(current_pixel):
        if next_pixel not in reached:
            checked += 1
            if checked % 1000 == 0:
                print(f'checked {checked} / {width * height} pixels')
            frontier.put(next_pixel)
            reached.add(next_pixel)
            # Cost of move is the inverse intensity. 0 is barrier, 255 is open.
            move_cost = cost[current_pixel] + 255 - pixels[next_pixel]
            cost[next_pixel] = move_cost


# Use the cost field to figure out optimal path.
path = []

# Display the original image with path inserted between start and end pixels.
gui = ti.GUI('Best Path', res=(width, height))
# load_path(pixels, path, img)

while gui.running:
    gui.set_image(cost)
    gui.show()