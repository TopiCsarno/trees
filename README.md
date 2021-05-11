## network effect vizsgálat
RFC_iris_test.ipynb - OK -> RFC_network_iris
    - Iris dataset-en élek számítása 
    - edge importance tesztelése
    - gráffá alakítás

RFC_edge_imp.ipynb - OK -> RFC_network_face
    - face embeddingen
    - 128 pontos gárf alapján edge importance számítás
    - gráf csomópontoknál fokszám számítása alapján
    - top feature-ök kivétele teszt

## redundancy vizsgálat
RFC_iris_redul.py
    - iris dataseten
    - redundancia vizsgálat
    - random forest utakat listázza ki súlyozottan

## adversarial számítás

RFC_advers_iris.py
    - iris dataseten
    - kézzel összerakott adversarial attack
    - próbálgatással, 1 feature módosításával
    - iris-ra tanított RF becsapása sikeresen

RFC_advers_face
    - face embeddingen
    - új algoritmus: egyes fákat hogyan lehet áttéríteni
    - ezzel sikerült több feature módosítással elegendően sok fát áttéríteni

## ez lehet fölös, átvihető a másikba
average_embeddings.py:
    - face embeddingen
    - X_train alapján a 4 race-hez tartozó centroidok kiszámolása

## Todo:
- bemutatni, hogy nem változik sokat módosítás után az embedding
- centroidok PCA plotja esetleg, módosított embedding PCA plot
- redundancy vizsgálat még nem az igazi