import cv2 as cv
import numpy as np

def izracunaj_centre(slika, izbira, dimenzija_centra, T, k):
    h, w, c = slika.shape
    centri = []

    if izbira == "nakljucno":
        while len(centri) < k:
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            barva = slika[y, x]

            if dimenzija_centra == 5:
                kandidat = np.array([barva[0], barva[1], barva[2], x, y])
            else:
                kandidat = np.array([barva[0], barva[1], barva[2]])

            preblizu = False
            for c in centri:
                if np.linalg.norm(c[:3] - kandidat[:3]) < T:
                    preblizu = True
                    break

            if not preblizu:
                centri.append(kandidat)

    elif izbira == "rocno":
        kliknjeni_centri = []

        def klik_mis(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                kliknjeni_centri.append((x, y))
                print(f"Kliknil: ({x},{y})")

        cv.namedWindow("Klikni centre", cv.WINDOW_NORMAL)
        cv.imshow("Klikni centre", slika)
        cv.setMouseCallback("Klikni centre", klik_mis)

        print(f"Klikni {k} centrov z miško...")
        while len(kliknjeni_centri) < k:
            cv.waitKey(1)

        cv.destroyWindow("Klikni centre")

        for (x, y) in kliknjeni_centri:
            barva = slika[y, x]
            if dimenzija_centra == 5:
                kandidat = np.array([barva[0], barva[1], barva[2], x, y])
            else:
                kandidat = np.array([barva[0], barva[1], barva[2]])
            centri.append(kandidat)

    else:
        raise ValueError("Napaka: izbira mora biti 'nakljucno' ali 'rocno'.")

    centri = np.array(centri, dtype=np.float32)
    return centri


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