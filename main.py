import cv2 as cv
import numpy as np

# Funkcija za izračun začetnih centrov (za k-means) - naključno ali ročno z miško
def izracunaj_centre(slika, izbira, dimenzija_centra, T, k):
    h, w, c = slika.shape  # višina, širina in število kanalov slike
    centri = []

    if izbira == "nakljucno":
        # Naključno izberi k centrov, ki so si dovolj različni (glede na prag T)
        while len(centri) < k:
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            barva = slika[y, x]

            # Če je dimenzija 5, vključimo še prostorske koordinate (x, y)
            if dimenzija_centra == 5:
                kandidat = np.array([barva[0], barva[1], barva[2], x, y])
            else:
                kandidat = np.array([barva[0], barva[1], barva[2]])

            # Preveri, če je kandidat preblizu kateremukoli obstoječemu centru
            preblizu = False
            for c in centri:
                if np.linalg.norm(c[:3] - kandidat[:3]) < T:
                    preblizu = True
                    break

            if not preblizu:
                centri.append(kandidat)

    elif izbira == "rocno":
        # Uporabnik z miško izbere k točk na sliki
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
            cv.waitKey(1)  # čaka na uporabnikove klike

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

# Implementacija K-means algoritma za segmentacijo slike
def kmeans(slika, k=3, iteracije=10, izbira="nakljucno", dimenzija_centra=3, T=20):
    h, w, c = slika.shape
    podatki = slika.reshape((-1, 3))  # preoblikuj sliko v seznam barv (vsaka vrstica je piksel)

    if dimenzija_centra == 5:
        # Dodaj prostorske koordinate (x, y) kot dodatne lastnosti
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        xy = np.stack((xx, yy), axis=2).reshape((-1, 2))
        podatki = np.concatenate((podatki, xy), axis=1)

    podatki = np.float32(podatki)

    # Inicializiraj centre
    centri = izracunaj_centre(slika, izbira, dimenzija_centra, T, k)

    for _ in range(iteracije):
        # Izračunaj Manhattan razdaljo do vseh centrov in določi najbližji center
        razdalje = np.sum(np.abs(podatki[:, None, :] - centri[None, :, :]), axis=2)
        labels = np.argmin(razdalje, axis=1)

        # Posodobi centre z aritmetično sredino točk, ki jim pripadajo
        novi_centri = []
        for i in range(k):
            tocke = podatki[labels == i]
            if len(tocke) > 0:
                novi_c = np.mean(tocke, axis=0)
            else:
                novi_c = centri[i]  # če ni točk, ohrani star center
            novi_centri.append(novi_c)
        centri = np.array(novi_centri)

    # Naredi segmentirano sliko z uporabo barv centrov
    segmentacija = centri[labels][:, :3]  # uporabimo le RGB vrednosti
    segmentacija = np.uint8(segmentacija)
    segmentacija = segmentacija.reshape((h, w, 3))
    return segmentacija


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