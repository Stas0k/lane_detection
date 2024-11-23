import cv2
import numpy as np


class BaseImgProc(object):
    """
    Image processing tools
        - Edge detection: Canny, Laplace, Sobel X/Y
        - Image smoothing: Gaussian Blur
        - Convert image color to grayscale
        - Thresholding
        - Dilation
        - Normalization
    """

    @classmethod
    def canny(cls, img, threshold_low=None, threshold_high=None):   
        """
        Apply canny edge detection on a given image.
        :param img: input image
        :param threshold_low: low threshold for hysteresis procedure
        :param threshold_high: high threshold for hysteresis procedure
        """  
        threshold_1 = threshold_low if threshold_low else 30
        threshold_2 = threshold_high if threshold_high else 80

        canny = cv2.Canny(img, threshold_1, threshold_2)
        return canny
    
    @classmethod
    def laplacian(cls, img, ddepth=cv2.CV_64F, ksize=5):
        """
        Apply Laplacian operator on the given image for edge detection.
        :param img: input image
        :param ddepth: depth of the output image
        :param ksize: kernel size
        """
        laplacian = cv2.Laplacian(img, ddepth, ksize=ksize)
        return laplacian
    
    @classmethod
    def sobel(cls, img, ddepth=cv2.CV_64F, ksize=5, threshold=50):
        """
        Apply Sobel-based edge detection.
        :param img: input image
        :param ddepth: depth of the output image
        :param ksize: kernel size
        """
        # apply sobel operator
        sobel_x = cv2.Sobel(img, ddepth, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(img, ddepth, 0, 1, ksize=ksize)

        # calculate gradient magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # aply a threshold to identify edges 
        edges = magnitude > threshold

        return edges

    @classmethod
    def cartoon(cls, img, sigma_s=60, sigma_r=0.5):
        """
        Uses cv2.stylization function to artistic effects on image.
        It enhances edges by smooting out low contrast areas and highlitning high contrast features.
        """
        return cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)

    @classmethod
    def gaussian_blur(cls, img, ksize):
        """
        Applies Gaussian blur (convolution of Gaussian kernel) to the image, smoother blur compared to averaging.
        :param img: source image
        :param ksize: Gaussian kernel size. Tuple of two odd integers. E.g. (5, 5)
        """
        return cv2.GaussianBlur(img, ksize, 0)
    
    @classmethod
    def averaging_blur(cls, img, ksize):
        """
        Applies Avearge blur filter to an image.
        Computes the average of all the pixels under the kernel area and replaces the central element with this average.
        :param img: source image
        :param ksize: Average kernel size. Tuple of two odd integers. E.g. (5, 5)
        """
        return cv2.blur(img, ksize)
    
    @classmethod
    def median_blur(cls, img, ksize):
        """
        Applies median blur to the image, by replaces each pixel's value with the median value of the neighboring pixels. 
        Effective for removing salt-and-pepper noise.
        :param img: source image
        :param ksize: Median kernel size - integer. E.g. 5
        """
        return cv2.medianBlur(img, ksize, 0)

    @classmethod
    def convert_color(cls, img, color_space=cv2.COLOR_BGR2GRAY):
        """
        Converts image from one color space to another.
        :param img: source image
        :param color_space: the color space conversion code (default: cv2.COLOR_BGR2GRAY)
        """
        return cv2.cvtColor(img, color_space)

    @classmethod
    def weighted(cls, img_1, img_2, alpha=1., beta=1., gamma=0.):
        """
        Blend two images together using specified weights.
        :param img_1: 1st input image
        :param img_2: 2nd input image
        :param alpha: weight of the 1st image
        :param beta: weight of the 2nd image
        :param gamma: scalar added to each sum
        initial_img * α + img * β + λ
        NOTE: calculation of pixel (x, y) in output image: alpha * img_1(x, y) + beta * img_2(x, y) + gamma
        Returns:
            Blended output image with the same size as img_1 and img_2    
        """
        return cv2.addWeighted(img_1, alpha, img_2, beta, gamma)
    
    @classmethod
    def dilate(cls, img, ksize=None, iterations=1, **kwargs):
        """
        Apply dilation to the image
        :param img: input image
        :param ksize: kernel size for dilation, will be converted to matrix of ones (default: 5x5)
        :param iterations: number of times to apply the dilation
        :param kwargs: optional keyword arguments - anchor, borderType, borderValue
        NOTE: calculation of pixel (x, y) in output image: alpha * img_1(x, y) + beta * img_2(x, y) + gamma
        Returns:
            Returns dilated image     
        """
        ksize = ksize if ksize else (5, 5)
        kernel = np.ones(ksize, np.uint8)
        dilated_img = cv2.dilate(img, kernel, iterations=iterations, **kwargs)
        return dilated_img

    @classmethod
    def threshold(cls, img, threshold=200, max_val=255, threshold_type=cv2.THRESH_BINARY):
        """
        Applies image threshold - segment the image by setting pixel values to a specific level based on the threshold
        :param img: input image
        :param threshold: threshold value. Pixels with values greater than this threshold will be set to the maxval (255 - for white color)
        :param max_val: the maximum value to use with the thresholding. If pixels meets the threshold - it will be set to max_val 
        :param threshold_type: threshold type (default: cv2.THRESH_BINARY)
        Returns:
            Returns thresholded image (kind of masked image)     
        """
        # apply threshold
        _, thresholded_img = cv2.threshold(img, threshold, max_val, threshold_type)
        return thresholded_img
    
    @classmethod
    def normalize(cls, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=-1):
        """
        Normalizes the pixel intensity values of an image in order to improve the contrast and make the image clearer.
        :param img: input image
        :param alpha: lower range boundary for normalization (default: 0)
        :param beta: upper range boundary for normalization (default: 255)
        :param norm_type: type of normalization (default: cv2.NORM_MINMAX)
        NOTE: cv2.NORM_MINMAX - ensures that the minimum pixel value in the original image becomes alpha and the maximum becomes beta.
        :param dtype: output image data type (default: -1 - means output image will have the same type as the input image)
        """
        # normalize image
        # 2nd argument is a destination image, which set to None meaning that cv2.normalize will init it to an empty array with the same shape as the input image
        norm_img = cv2.normalize(img, None, alpha, beta,norm_type, dtype)
        return norm_img