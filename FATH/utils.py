import os
import scipy

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def imsave(images, size, path):
    return scipy.misc.imsave(path, make_image(images, size))

def make_image(images, size):
    h, w = images.shape[1], images.shape[2]

    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))

        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h: j * h + h, i * w:i * w + w, :] = image

        return img
    
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h: j * h + h, i * w:i * w + w] = image[:,:,0]

        return img

    else:
        raise ValueError('must have dimensions 4, 3, 1')

