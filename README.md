# SPOT
### SPOT — Système d’appel automatisé par reconnaissance faciale (projet pédagogique)

SPOT est un mini-système de projet de cours visant à simuler **un appel automatique en classe** via **reconnaissance faciale**. Le projet couvre à la fois :

- **la modélisation “métier”** (professeur → cours → classe → liste d’élèves attendus),
- **la constitution d’un dataset**,
- **l’entraînement d’un modèle de reconnaissance**,
- **la détection/reconnaissance en temps réel (webcam)**,
- **la génération d’un rapport de présence** (présent / absent / parti / retard).

Le code principal est dans le notebook `Pipeline_SPOT.ipynb`, conçu pour fonctionner sur **Google Colab**.

---

### Démo


https://github.com/user-attachments/assets/15f06b44-4fe9-4fe4-a9e8-3e1d3a38bda8



### Objectifs pédagogiques

- **Comprendre une pipeline de vision** de bout en bout (capture → détection → prétraitement → entraînement → inférence).
- Faire attention à la différence entre **détection de visage** (localiser un visage) et  **reconnaissance** (identifier _qui_ est la personne).

---

### Architecture logique 

Le dataset a une structure simple :

- **Professeur**
  - **Cours / Classe**
    - **Élèves attendus** (`ELEVES_ATTENDUS`)

Cette liste `ELEVES_ATTENDUS` est la référence utilisée pour :

- afficher le tableau de bord “présence” en live,
- décider “présent / intrus / inconnu”,
- produire le rapport final.

---

### Méthodologie de reconnaissance faciale (vision)

SPOT suit la pipeline suivante :

#### 1) Capture et dataset (WEBCAM)

Le projet privilégie un dataset construit via la **webcam Colab**, car c’est **la même caméra, la même compression et les mêmes conditions** que l’inférence live.

- Un dossier par élève : `/content/dataset_crop/<eleve>/`
- Le dataset est alimenté via des **boutons dynamiques** (un bouton par élève chargé).

#### 2) Détection de visage (Haar Cascade)

La détection utilise un classifieur Haar (`haarcascade_frontalface_default.xml`).

Limite : moins robuste que des détecteurs modernes (DNN/MTCNN/RetinaFace), notamment en multi-visages, angles marqués ou faible lumière.

#### 3) Prétraitement

Le prétraitement standardise les entrées :

- conversion **grayscale**,
- **resize** en `200×200`,
- normalisation de contraste via **CLAHE**.

##### Pourquoi CLAHE ?

**CLAHE** (_Contrast Limited Adaptive Histogram Equalization_) est une version “locale” de l’égalisation d’histogramme :

- Elle améliore le contraste région par région au lieu d’appliquer une correction globale.
- Elle aide beaucoup lorsque la lumière est :
  - inégale (ombre sur le visage),
  - trop faible,
  - variable entre frames.

La version “contrast limited” limite l’amplification du bruit (sinon une égalisation locale peut rendre le grain très visible).

Dans SPOT, CLAHE sert surtout à rendre les textures plus comparables entre images d’un même individu.

#### 4) Modèle de reconnaissance (LBPH)

SPOT utilise **LBPH** (_Local Binary Patterns Histograms_) via `cv2.face.LBPHFaceRecognizer_create`.

Pourquoi LBPH ?

- très accessible pédagogiquement,
- léger (CPU),
- rapide à entraîner,
- simple à déployer en démo.

Limites :

- sensible à la qualité d’image,
- sensible au domaine (caméra),
- performances inférieures aux approches à base d’**embeddings deep** (FaceNet, ArcFace, etc.).

---

### Seuils et calibration

LBPH retourne une **distance** (plus petit = meilleur). Pour décider si une prédiction est acceptée :

- on compare la distance à un **seuil**.

#### Pourquoi calibrer ?

La distance LBPH dépend fortement de la caméra,de la taille du visage dans la frame,du nombre de visages (solo vs 2 personnes dans le cadre) et de la lumière.

SPOT inclut une cellule de **Calibration LIVE** qui mesure les distances réelles (webcam) et propose un seuil du type :

$$\text{seuil} \approx p95 + 10$$

où \(p95\) est le 95e percentile des distances observées.

#### Multi-visages

Quand **2 personnes** sont dans l’image, les visages sont souvent plus petits, ce qui augmente la distance. La pipeline live applique donc :

- un **seuil adaptatif** selon la taille du visage,
- un seuil spécifique **multi-face**,
- un mécanisme anti-flap (validation après plusieurs frames).

---

### Anti-flap (stabilité temporelle)

Pour éviter “présent/inconnu” qui oscille image par image, SPOT demande plusieurs frames consécutives (streak) avant de valider un élève.

Cela améliore :

- la stabilité visuelle,
- la cohérence de la base session (first_seen / last_seen),
- la qualité du rapport final.

---

### Rapport final

En fin de session, SPOT calcule le retard (arrivée), le statut final (présent / absent / parti) et le temps total d’absence (retard + sortie anticipée) et produit un tableau récapitulatif.

---

### Exécution sur Google Colab

#### Pré-requis

Sur Colab, **LBPH** requiert `opencv-contrib-python-headless` (sinon `cv2.face` n’existe pas).

Le notebook inclut une cellule “à exécuter en premier” qui désinstalle les paquets OpenCV conflictuels, réinstalle `opencv-contrib-python-headless` et vérifie la présence de `cv2.face`.

### Étapes recommandées pour tester le projet

- **Redémarrer le runtime** (Colab).
- Exécuter la cellule d’installation OpenCV (tout début du notebook).
- Sélectionner **Prof / Classe** puis cliquer **Charger la classe**.
- Créer le dataset webcam :
  - pour chaque élève, capturer ~20–60 images en variant légèrement pose/angle.
- Lancer l’entraînement LBPH.
- Lancer **Calibration LIVE** (solo puis multi-visages si nécessaire).
- Lancer la reconnaissance live + tableau de bord.
- Stopper la boucle (KeyboardInterrupt) puis générer le rapport final.

---

### Performances et limites (important)

#### Pourquoi le flux webcam peut “saccader” sur Colab ?

Colab fait transiter chaque frame via :

- JS (navigateur) → base64 → Python (`eval_js`)
- traitement CV → rendu widget

Ce “pont” ajoute de la latence. Le notebook limite donc :

- la fréquence de rendu UI,
- le FPS,

pour rendre le flux plus fluide sans nuire à la détection.

Ce projet est **pédagogique** et destiné à une démonstration technique.

---

### Contenu du dépôt

- `Pipeline_SPOT.ipynb`: notebook principal (dataset webcam, entraînement, calibration, live, rapport)
