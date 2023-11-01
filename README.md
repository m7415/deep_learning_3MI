# Correction de Lumière Parasite pour Instruments Optiques de Haute Précision

Ce repo GitHub est dédié à mon rapport de stage au Centre Spatial de Liège, où j'ai eu l'opportunité de travailler sur un projet passionnant visant à améliorer les performances des instruments optiques de haute précision, en mettant un fort accent sur l'instrument d'observation de la Terre, 3MI.

**À propos du Projet :**

L'instrument optique est au cœur des missions spatiales, et sa performance est cruciale pour la qualité des images obtenues. Mon travail a consisté à introduire des techniques d'apprentissage profond (deep learning) pour améliorer la correction de la lumière parasite, un défi majeur dans ce domaine. La lumière parasite peut dégrader la résolution et le rapport signal/bruit des images, ce qui peut avoir un impact significatif sur le succès des missions spatiales.

**Objectifs :**

- Développer une solution de correction des aberrations de lumière parasite.
- Améliorer la généralisation des tâches de correction.
- Augmenter la robustesse pour réduire les corrections manuelles.

Ce repo contient le code source de mes expérimentations lors de ce stage, ainsi que le rapport de stage qui en a résulté. Les données utilisées ne sont pas incluses, car elles sont très volumineuses, et confidentielles. Le code source est brouillon, et mériterait d'être nettoyé. Certains notebook ne sont exécutable que sur google colab.

Je conseille pour commencer de lire mon [rapport de stage](SDIA_PETITPOISSON_Maxime_Rapport_2A.pdf). Le notebook [black and white](black_and_white.ipynb) contient aussi des résultats intéressants qui ne sont pas présentés dans le rapport.
N'hésitez pas à explorer les différentes sections et à poser des questions si vous en avez. Votre intérêt et vos commentaires sont les bienvenus !

## Installation

### create a virtual environment with python 3.6.9
```
python3 -m venv venv
```

### activate the virtual environment (linux)
```
source venv/bin/activate
```

### activate the virtual environment (windows)
```
source venv/Scripts/activate
```

### install the requirements
```
pip install -r requirements.txt
```
