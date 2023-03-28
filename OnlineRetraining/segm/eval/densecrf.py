import numpy as np


def crf_inference_voc12(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    ## voc12
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83 / scale_factor, srgb=5, rgbim=np.ascontiguousarray(np.copy(img_c)), compat=3)
    # d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=13, rgbim=np.copy(img_c), compat=10)
    # d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.ascontiguousarray(np.copy(img_c)), compat=10)
    # d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=32/scale_factor, srgb=13, rgbim=np.copy(img_c), compat=10)
    # d.addPairwiseGaussian(sxy=1 / scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=67 / scale_factor, srgb=3, rgbim=np.copy(img_c), compat=4)

    ## coco
    # d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img_c)), compat=10)

    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def crf_inference_coco(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    ## voc12
    # d.addPairwiseGaussian(sxy=4 / scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=83 / scale_factor, srgb=5, rgbim=np.copy(img_c), compat=3)
    # d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img_c), compat=10)
    # d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    # d.addPairwiseBilateral(sxy=32/scale_factor, srgb=13, rgbim=np.copy(img_c), compat=10)
    ## coco
    d.addPairwiseGaussian(sxy=1 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=67 / scale_factor, srgb=3, rgbim=np.copy(img_c), compat=4)
    # d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.copy(img_c), compat=10)

    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))
