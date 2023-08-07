# path2d
Experiments in path-finding for 2d grayscale image.

Requires installation of taichi language support.

All of the programs show progress by periodic update of an image displayed via
the taichi GUI. At this point, taichi parallel acceleration isn't really used.

app_grow_cell.py -- Simple breadth-first that uses hardcoded "membrane" threshold.
app_path.py  -- Move to PriorityQueue to allow modifying frontier costs by alternative paths.
app_path2.py -- Add path length to cost function and show optimal path at end.
