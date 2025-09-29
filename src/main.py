import cv2
import numpy as np

img = cv2.imread("../basketball-court.ppm") #BGR (y, x, 3)
input_size = (img.shape[1], img.shape[0])

output_size = (900, 600)

correspondence = [ # (x_i -> (x_i)')
    ([23, 193, 1], [0, 0, 1]),
    ([247, 51, 1], [output_size[0], 0, 1]),
    ([279, 279, 1], [0, output_size[1], 1]),
    ([402, 74, 1], [output_size[0], output_size[1], 1])
]

def build_normalization(width, height):
    T = np.array([
        [width+height, 0, width/2],
        [0, width+height, height/2],
        [0, 0, 1]
    ])

    return np.linalg.inv(T)

def normalize(correspondences, T1, T2):
    return list(map(lambda pair: (T1@pair[0], T2@pair[1]), correspondences))

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

def build_homography(correspondences, input_size, output_size):
    # generate normalization matrices
    T_norm_in = build_normalization(*input_size)
    T_norm_out = build_normalization(*output_size)

    #normalize coordinates
    normalized_c = normalize(correspondences, T_norm_in, T_norm_out)

    #generate system to solve for H
    A = build_system(normalized_c)
    _, _, vT = np.linalg.svd(A)
    h = vT[-1, :] #last col of vT is last row of v
    H_norm = h.reshape((3, 3))

    #denormalize solution
    return np.linalg.inv(T_norm_out) @ H_norm @ T_norm_in

def project(homography, point):
    destination = homography @ point
    scale = destination[-1]
    return destination / scale

def nearest_neighbor(src, point):
    x, y, _ = np.rint(point).astype(int)
    return src[y, x]

def interpolate_bilinear(src, point):
    i, j, _ = np.floor(point).astype(int)
    a, b, _ = point - np.floor(point)

    return (
        (1-a) * (1-b) * src[j, i] +
        a * (1-b) * src[j, i+1] +
        a * b * src[j+1, i+1] +
        (1-a) * b * src[j+1, i]
    )

def generate_image(src, homography, output_size):
    w, h = output_size
    out = np.zeros((h, w, 3))
    H_inv = np.linalg.inv(homography)

    for x in range(w):
        for y in range(h):
            point = np.array([x, y, 1])
            src_point = project(H_inv, point)
            out[y, x] = interpolate_bilinear(src, src_point)

    cv2.imwrite("../bilinear.png", out)


H = build_homography(correspondence, input_size, output_size)
generate_image(img, H, output_size)
