# CSL_3MI_ML

## Description
Experimentations on the use of ML in stray light correction on the case of the 3MI

## TODO

- [ ] change data_wraper to lighten the memory usage
- [ ] do a grid search on the experiment parameters

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

## TODO
* Repasser en linéaire -> map théorique
* prendre le nominal (somme des 4 pixels maximum dans un carré de 2x2)
* normaliser la map au nominal

* mettre le nominal à 0
* faire la somme de la map -> stray light
* on obtient les valeurs de tous les champs
* on plot la valeur de la straylight en fonction de x et y du nominal
* scatter pour avoir une cartographie de l'integrale

* créer un mask de 20 autour du nominal (disque)
* faire la somme de la straylight sauf dans le disque au milieu