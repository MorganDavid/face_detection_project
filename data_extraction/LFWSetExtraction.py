import os

_images_path = r"raw datasets/LFW/images"
_labels = r"raw datasets/LFW/annotations"
_fold = "FDDB-fold-01.txt"
_labels_path = os.path.join(_labels, _fold)

# elipse in form <major_axis_radius minor_axis_radius angle center_x center_y 1>.
def elipse_to_rect(elipse):
	x = elipse[1]