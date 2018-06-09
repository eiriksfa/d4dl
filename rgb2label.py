import numpy
from skimage import io
import time


def rgb2label(img, color_codes = None, one_hot_encode=False):
    if color_codes is None:
        color_codes = {val:i for i,val in enumerate(set( tuple(v) for m2d in img for v in m2d ))}
    n_labels = len(color_codes)
    result = np.ndarray(shape=img.shape[:2], dtype=int)
    result[:,:] = -1
    for rgb, idx in color_codes.items():
        result[(img==rgb).all(2)] = idx

    if one_hot_encode:
        one_hot_labels = np.zeros((img.shape[0],img.shape[1],n_labels))
        # one-hot encoding
        for c in range(n_labels):
            one_hot_labels[: , : , c ] = (result == c ).astype(int)
        result = one_hot_labels

    return result, color_codes

color2index = {
        (255, 255, 255) : 0,
        (0,     0, 255) : 1,
        (0,   255, 255) : 2,
        (0,   255,   0) : 3,
        (255, 255,   0) : 4,
        (255,   0,   0) : 5,
        (0,0,0) : 6,
        (1,1,1) : 7,
        (1,2,3) : 8,
        (2,13,19) : 9,
        (4,21,31) :10,
        (7,42,62) : 11,
        (19,22,25) : 12
    }

classlib = {
	(0,0,0) : 0,
	(0,0,255) :1,
	(0,255,0): 2
}

def im2index(im):
    """
    turn a 3 channel RGB image to 1 channel index image
    """
    assert len(im.shape) == 3
    height, width, ch = im.shape
    assert ch == 3
    m_lable = np.zeros((height, width, 1), dtype=np.uint8)
    for w in range(width):
        for h in range(height):
            b, g, r = im[h, w, :]
            m_lable[h, w, :] = classlib[(r, g, b)]
    return m_lable

img = io.imread('/home/novian/term2/dl4ad/repo2/d4dl/testimg/317.png')
#print(img)
#img_labels, color_codes = rgb2label(img)
#print(color_codes)

color_map = numpy.ndarray(shape=(256*256*256), dtype='int32')
color_map[:] = -1
for rgb, idx in classlib.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    color_map[rgb] = idx

def img_array_to_single_val(image, color_map):
    image = numpy.dot(image,numpy.array([65536, 256, 1], dtype='int32'))
    return color_map[image]

#imgl = im2index(img)
t = time.time()
imgl = img_array_to_single_val(img,color_map)
print(time.time()-t)
print(imgl)
#print(imgl)