import numpy as np


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7, scale_factor=1):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=4 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=83 / scale_factor, srgb=5, rgbim=np.copy(img), compat=3)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)

def crf_inference_label_coco(img, labels, t=10, n_labels=21, gt_prob=0.7, scale_factor=1):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)