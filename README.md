# Logo Similarity

Acest proiect are ca scop gruparea website-urilor pe baza asemanarii logo-urilor lor. Solutia este construita in doua etape principale: generarea embedding-urilor si clusteringul.

## Generarea embedding-urilor

In prima etapa, logo-urile sunt descarcate folosind API-ul logo.dev. Fiecare imagine este preprocesata pentru a elimina fundalul, a uniformiza dimensiunea si a reduce influenta culorilor sau a calitatii imaginii. Pentru fiecare domeniu sunt generate doua embedding-uri folosind modelul DINOv2: unul pentru logo-ul complet si unul pentru icon. Embedding-urile sunt normalizate L2 pentru a putea fi comparate.

## Extragerea iconului

Pentru extragerea iconului este utilizat un algoritm simplu de procesare a imaginilor bazat pe OpenCV. Imaginea este convertita in RGB pe fundal alb, iar apoi o masca de forground este creata folosind pixelii care au o cluare diferita fata de fundal, iar apoi este eliminat zgomotul. Din acesta masca sunt extrase componentele conexe, iar cea mai mare componenta este selectata ca reprezentare a simbolului principal. Procesul este aplicat atat pe imaginea originala, cat si pe cea normalizata, iar rezultatul final este ales automat pe baza cantitatii de foreground detectate.

## Clustering

Clusteringul este realizat in doua etape. Algoritmul DBSCAN cu metrica cosine este folosit pentru obtinerea unor grupuri mai largi si, de asemenea, pentru a reduce numarul de comparatii necesare. In al doilea pas, se construieste un graf in care sunt trasate muchii intre imaginile din aceeasi grupa, doar daca scorul de similaritate dintre acestea depaseste un anumit prag, calculat folosind asemanarea embedding-urilor. Clusterele finale se obtin prin calcularea componentelor conexe ale acestui graf.

## Rezultate

Rezultatele sunt stocate intr-un fisier HTML care arata clusterele si logo-urile grupate, astfel vizualizarea fiind mai simpla.