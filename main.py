import cv2 as cv
import numpy as np


if __name__ == "__main__":
    """
    slika = cv.imread("./paprika.jpg")
    slika = cv.resize(slika, (100, 100))
    segmentirana = meanshift(slika, h=30, dimenzija=3)
    cv.imshow("MeanShift segmentacija", segmentirana)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """

    """
    slika = cv.imread("./paprika.jpg")
    segmentirana = kmeans(slika, k=5, iteracije=10, izbira="nakljucno", dimenzija_centra=3, T=20)
    cv.imshow("KMeans segmentacija", segmentirana)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """