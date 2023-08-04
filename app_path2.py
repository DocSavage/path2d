# Try to find better paths by priotizing best paths and
# adding path length to the cost function.

from queue import PriorityQueue
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

# The length of path from start pixel.
path_length = ti.field(dtype=ti.i32, shape=(width, height))

# The minimum intensity of pixels from start to each pixel.
min_int = ti.field(dtype=ti.u8, shape=(width, height))

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

# Display the original image with path inserted between start and end pixels.
gui = ti.GUI('Best Path', res=(width, height))

# Breadth-first search, but we'd want to move toward Dijkstra's algorithm.
# Calculates the cost of through any pixel in the whole image.
frontier = PriorityQueue()

start_x, start_y = start_pixel
start_intensity = pixels[start_x, start_y, 0]
min_int[start_pixel] = start_intensity
print('starting intensity: ', start_intensity)

frontier.put((0, start_pixel))
reached = set()
reached.add(start_pixel)


# This is a manually set threshold where we assume membranes and other
# "walls" are darker than this intensity.  You can modify it and see if
# it's too low, the growing will leak out of the starting cell.
cytoplasm_threshold = 130

checked = 0
abort_when_find_end = True
determined_path = False
found_end = False
while gui.running:
    while not found_end and not frontier.empty():
        score, current_pixel = frontier.get()
        if abort_when_find_end and current_pixel == end_pixel:
            print('found end pixel -- aborting, perhaps prematurely')
            found_end = True
            break

        if checked % 100 == 0:
            gui.set_image(display)
            gui.show()

        for next_pixel in neighborhood(current_pixel):
            checked += 1
            if checked % 10000 == 0:
                print(f'checked {checked} / {width * height} pixels')

            # get intensity of this pixel
            next_x, next_y = next_pixel
            next_intensity = pixels[next_x, next_y, 0]

            # only grow through cytoplasm (use threshold)
            if next_intensity < cytoplasm_threshold:
                continue

            # Path cost is a function of worst (minimum) intensity along path AND path length.
            # We multiply by 255 to make intensity more important than path length.
            cur_path_length = path_length[current_pixel] + 1
            cur_min_intensity = min(min_int[current_pixel], next_intensity)
            cur_path_cost = 255 * (255 - cur_min_intensity) + cur_path_length

            # if this has already been considered, only reconsider if we've found
            # a better (higher intensity) path.
            if next_pixel in reached and cur_path_cost >= cost[next_pixel]:
                continue

            # print(f'Set path to {next_pixel}: cost {cur_path_cost}, min intensity {cur_min_intensity}, length {cur_path_length}')
            min_int[next_pixel] = cur_min_intensity
            path_length[next_pixel] = cur_path_length
            cost[next_pixel] = cur_path_cost

            cost_for_display = int(255 - (cur_path_cost / 255))
            display[next_x, next_y, 1] = cost_for_display if cost_for_display > 0 else 0
            display[next_x, next_y, 0] = 255
            display[next_x, next_y, 2] = 255

            frontier.put((cur_path_cost, next_pixel))
            reached.add(next_pixel)

    # Now that we've computed costs between the start and end point, we can get a good path.
    if not determined_path:
        # reset to original grayscale
        for i, j in reached:
            display[i, j, 0] = pixels[i, j, 0]
            display[i, j, 1] = pixels[i, j, 0]
            display[i, j, 2] = pixels[i, j, 0]

        # paint path
        pixel = end_pixel
        while pixel != start_pixel:
            best_cost = cost[pixel] + 1
            for next_pixel in neighborhood(pixel):
                # make sure we only consider pixels that have been reached
                # since unreached pixels have zero cost
                if next_pixel in reached and cost[next_pixel] < best_cost:
                    best_pixel = next_pixel
                    best_cost = cost[next_pixel]

            pixel = best_pixel
            x, y = pixel
            display[x, y, 0] = 0
            display[x, y, 1] = 0
            display[x, y, 2] = 255
            gui.set_image(display)
            gui.show()
        determined_path = True
        print('done painting path')

    gui.set_image(display)
    gui.show()
