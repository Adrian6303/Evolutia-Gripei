# Proiect Evoluția Gripei
### Echipa Sănătate pe Sistem
#### Membri Echipă:

* Șchiopu Adrian
* Șerpe Vlad

## Algritmi de Machine Learning folosiți
### 1. Logistic Regression
Regresia logistică este un algoritm de clasificare utilizat frecvent pentru a prezice rezultate binare (de exemplu, răspuns ridicat vs. răspuns scăzut). Acesta calculează probabilitatea ca un punct de date să aparțină unei clase specifice folosind o combinație liniară a caracteristicilor de intrare și aplicând o funcție sigmoidă, care transformă rezultatul într-o probabilitate între 0 și 1. În funcție de această probabilitate, modelul atribuie punctul de date uneia dintre cele două clase.

Pentru antrenarea modelului, regresia logistică folosește o funcție de cost numită entropie încrucișată, care măsoară diferența dintre valorile prezise și cele reale. Printr-o tehnică de optimizare numită gradient descent, modelul ajustează parametrii iterativ pentru a minimiza costul și a îmbunătăți acuratețea.

Regresia logistică este eficientă pentru sarcini de clasificare binară, oferind interpretabilitate clară a importanței caracteristicilor, ceea ce este valoros pentru înțelegerea factorilor ce influențează răspunsul imun la vaccinul antigripal.



### 2. Decision Tree
    
Arborele de decizie este un algoritm de clasificare și regresie care utilizează o structură de tip arbore pentru a lua decizii bazate pe caracteristicile de intrare. Fiecare nod din arbore reprezintă o întrebare sau o condiție asupra unei caracteristici, iar fiecare ramură reprezintă răspunsul (sau decizia) posibilă. Algoritmul împarte datele în subseturi la fiecare nod, până când ajunge la frunze, unde se atribuie o clasă finală sau o valoare de predicție.

Arborii de decizie sunt eficienți și ușor de interpretat, deoarece oferă explicații vizuale pentru clasificări. Aceștia sunt valoroși în proiectul nostru, deoarece permit identificarea factorilor principali care influențează răspunsul la vaccinul antigripal și ajută la dezvoltarea unui sistem de suport decizional în domeniul sănătății.

### 3. Random Forest

Random Forest este un algoritm de învățare automată bazat pe arborele de decizie, utilizat pentru sarcini de clasificare și regresie. Acesta construiește o colecție (sau „pădure”) de arbori de decizie, fiecare fiind antrenat pe un subset aleatoriu al datelor, folosind și un subset aleatoriu de caracteristici. Predicția finală este obținută prin agregarea rezultatelor fiecărui arbore – de exemplu, prin vot majoritar în cazul clasificării sau prin media predicțiilor în cazul regresiei.

Această abordare reduce riscul de supraînvățare (overfitting) și îmbunătățește robustețea și acuratețea modelului. Random Forest este valoros în proiectul nostru deoarece poate analiza relații complexe dintre caracteristici și poate identifica factorii cheie care influențează răspunsul la vaccinul antigripal, contribuind la dezvoltarea unui sistem de sprijin decizional mai precis și fiabil. De asemenea, algoritmul gestionează bine datele incomplete și variabilele irelevante.

## Logistic Regression pe 2 dataset-uri de mici dimensiuni:

Am aplicat Logistic Regression pentru a prezice un outcome ce poate fii low sau high, avănd 75 % train data si 25% test data. Am realizat o acuratete de 0.66 respectiv 0.7 pentru cele 2 dataset-uri.
    
Am încercat de asemenea combinarea celor 2 dataset-uri pentru un număr mai mare de date, dar deoarece nu toți parametrii au fost identici, au fost excluși câțiva parametrii și am ajuns la o acuratete de 0.525.
 
Am afișat informați despre rezultatul obținut în fișier, cum ar fii acuratetea, raportul de clasificare si importanța fiecarui parametru.

## Decision Tree pe 2 dataset-uri de mici dimensiuni:
	
Am aplicat Decision Tree pentru a prezice un outcome ce poate fii low sau high, avănd  aceleasi date folosite anterior. Am realizat o acuratete de 0.8 respectiv 0.7 pentru cele 2 dataset-uri.
     
Am afișat informați despre rezultatul obținut în fișier, cum ar fii acuratetea, raportul de clasificare si importanța fiecarui parametru.

## Logistic Regression si Decision Tree pe dataset-ul mare:

Am aplicat Logistic Regression si apoi Decision Tree pentru a prezice un outcome ce poate fii low sau high, avănd 90 % train data si 10% test data. Am realizat o acuratete de 0.62 respectiv 0.64 pe cei 2 algoritmi. 
Am afișat informați despre rezultatul obținut în fișier, cum ar fii acuratetea, raportul de clasificare si importanța fiecarui parametru.

## Logistic Regression si Decision Tree pe dataset-ul mare preprocesat:
Am prelucrat dataset-ul original pentru a obtine un dataset nou cu fiecare rand reprezentand un pacient. 
Pe dataset-ul rezultat am salvat unul unde raman doar coloanele cu cel mult 75% valori goale, respective 50% valori goale.

Dataset1: 363 pacienti cu  384 de features folositi pentru antrenarea modelului

Dataset2:  363 pacienti cu 102 de features folositi pentru antrenarea modelului

## Concluzie

Dintre toate modelele testate cel mai bun este variant ace foloseste Decision Tree pe un dataset preprocesat, la acesta am obtinut o acuratete de 75%,  al doilea cel muai bun model fiind cel cu Logistic Regression pe acelasi dataset, avand 68% acuratete. 

Pentru procesare am selectat toate analizele unui pacient sa se afle pe o singura linie si am selectat doar parametrii ce au maxim 75% valori nule din numarul total. Am remarcat ca acesta este cel mai efficient dintre cele 3 variante incercate (25% NAN, 50% NAN, 75% NAN). Astfel am ajuns la un dataset cu 363 de pacienti cu 389 de parametri

Aceasta abordare ne permite sa ignoram parametrii care nu influenteaza rezultatul (vaccine_response) si sa ne folosim efficient de datele relevante pentru a prezice cat mai precis un rezultat.