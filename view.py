import numpy
import nibabel as nib
import matplotlib.pyplot
import PIL.Image

nii_pred = nib.load(
    "/home/vincent/Downloads/st/Task01_BrainTumour/sliced_dataset/out/predictions/xxxx_pred.nii.gz"
)
nii_img = nib.load(
    "/home/vincent/Downloads/st/Task01_BrainTumour/sliced_dataset/out/predictions/xxxx_img.nii.gz"
)
nii_gt = nib.load(
    "/home/vincent/Downloads/st/Task01_BrainTumour/sliced_dataset/out/predictions/xxxx_gt.nii.gz"
)

img_pred = nii_pred.get_fdata()
img_img = nii_img.get_fdata()
img_gt = nii_gt.get_fdata()

img_img = (img_img - img_img.min()) / (img_img.max() - img_img.min())

im1 = numpy.stack((img_img[96, :, :], img_img[96, :, :], img_img[96, :, :]),
                  0).swapaxes(0, 2).swapaxes(0, 1)

I = PIL.Image.fromarray((im1 * 255).astype(numpy.uint8))


def label2pic(a, pos):
    O = PIL.Image.fromarray(numpy.uint8(a[pos, :, :]))
    O = O.convert('RGBA')
    x, y = O.size
    for i in range(x):
        for j in range(y):
            color = O.getpixel((i, j))
            if color[0] == 0:
                O.putpixel((i, j), (0, 0, 0, 0))
            elif color[0] == 1:
                O.putpixel((i, j), (128, 0, 0, 128))
            elif color[0] == 2:
                O.putpixel((i, j), (0, 128, 0, 128))
            elif color[0] == 3:
                O.putpixel((i, j), (0, 0, 128, 128))
    return O


matplotlib.pyplot.figure(figsize=(9, 3))
matplotlib.pyplot.subplot(1, 3, 1)
matplotlib.pyplot.title("Pred")
matplotlib.pyplot.imshow(I)
matplotlib.pyplot.imshow(label2pic(img_pred, 96))

matplotlib.pyplot.subplot(1, 3, 2)
matplotlib.pyplot.title("Img")
matplotlib.pyplot.imshow(I)

matplotlib.pyplot.subplot(1, 3, 3)
matplotlib.pyplot.title("GT")
matplotlib.pyplot.imshow(I)
matplotlib.pyplot.imshow(label2pic(img_gt, 96))

matplotlib.pyplot.show()
