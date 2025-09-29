import cv2
import numpy as np

#img = cv2.imread("../basketball-court.ppm") #BGR (y, x, 3)

output_size = (900, 600)

correspondence = [ # (x_i -> (x_i)')
    ([23, 193, 1], [0, 0, 1]),
    ([247, 51, 1], [output_size[0], 0, 1]),
    ([279, 279, 1], [0, output_size[1], 1]),
    ([402, 74, 1], [output_size[0], output_size[1], 1])
]

# ASSUMING: h^i is the ith row of H

def build_system(correspondences):
    rows = 2*len(correspondences)
    columns = 9
    system = np.zeros((rows, columns))
    current_row = 0
    for source, destination in correspondences:
        x, y, w = source
        xp, yp, wp = destination

        system[current_row:current_row+2] = [
            [0, 0, 0, -wp*x, -wp*y, -wp*w, yp*x, yp*y, yp*w],
            [wp*x, wp*y, wp*w, 0, 0, 0, -xp*x, -xp*y, -xp*w]
        ]
        current_row += 2

    return system

def build_homography(correspondences):
    A = build_system(correspondences)
    _, _, vT = np.linalg.svd(A)
    h = vT[-1, :] #last col of vT is last row of v
    return h.reshape((3, 3))

def project(homography, point):
    destination = homography @ point
    scale = destination[-1]
    return destination / scale

H = build_homography(correspondence)
for src, dst in correspondence:
    print(project(H, src), dst)

# import numpy as np
#
# arr = np.zeros((300,400,3))
# arr[100:200, 100] = [255, 0, 0]
# print(arr[100:200, 100])
# cv2.imwrite("../out.png", arr)
