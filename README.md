# Atelier 2 – Deep Learning : CNN, Faster R-CNN, Transfer Learning et Vision Transformer sur MNIST
**Module :** Deep Learning et Applications– **Encadrant :** Pr. Lotfi ELAACHAK

## 1. Objectif

Prendre en main PyTorch (en mode GPU) pour construire et comparer plusieurs architectures de vision par ordinateur sur le dataset **MNIST** (60 000 images d'entraînement, 10 000 de test, 28×28, niveaux de gris) :

- **Partie 1** : CNN entraîné de zéro, Faster R-CNN, et fine-tuning de VGG16 et AlexNet.
- **Partie 2** : Vision Transformer (ViT) construit from scratch en suivant le tutoriel de Brian Pulfer, puis comparaison avec la partie 1.

## 2. Environnement

- Google Colab, GPU **Tesla T4**, PyTorch et torchvision
- Mesures : accuracy, F1-score (macro), loss sur le jeu de test, temps d'entraînement (mesuré sur l'entraînement seul)

## 3. Modèles et hyperparamètres

| Modèle | Principe | Paramètres | Hyperparamètres principaux |
|---|---|---|---|
| **CNN** | 2 × (Conv 3×3, stride 1, padding 1 + ReLU + MaxPool 2×2) → Dropout 0,25 → FC 128 → FC 10 | 421 642 | Adam, lr 1e-3, batch 64, 5 epochs |
| **Faster R-CNN** | ResNet50-FPN pré-entraîné COCO, tête de prédiction remplacée (11 classes) ; boîte englobante calculée automatiquement autour de chaque chiffre ; classe de la détection la plus confiante utilisée comme prédiction | – | SGD lr 5e-3, momentum 0,9, weight decay 5e-4, images 128×128, 2 epochs, **10 000 images d'entraînement / 2 000 de test** |
| **AlexNet** (fine-tuning) | Poids ImageNet, convolutions gelées, dernière couche remplacée (10 classes) ; images agrandies à 64×64 et répétées sur 3 canaux | 57 044 810 (54 575 114 entraînables) | Adam, lr 1e-4, 2 epochs |
| **VGG16** (fine-tuning) | Même principe qu'AlexNet | 134 301 514 (119 586 826 entraînables) | Adam, lr 1e-4, 2 epochs |
| **ViT (tuto)** | 49 patchs de 4×4, projection linéaire, token de classe, encodage positionnel sinusoïdal, 2 blocs d'encodeur (LayerNorm, attention multi-têtes, MLP, connexions résiduelles) | 1 642 | `hidden_d=8`, 2 blocs, 2 têtes, Adam lr 5e-3, 5 epochs |
| **ViT (grand)** | Même architecture, plus grande | ≈ 148 000 | `hidden_d=64`, 4 blocs, 4 têtes, Adam lr 1e-3, 8 epochs |

Remarque : la tête de classification du ViT est un simple `Linear` (sans softmax), car `CrossEntropyLoss` applique déjà le softmax. Le tutoriel d'origine ajoute un softmax avant cette loss.

## 4. Résultats (jeu de test)

| Modèle | Accuracy | F1-score | Loss | Temps d'entraînement (s) |
|---|---|---|---|---|
| **CNN** | **0,9918** | **0,9917** | 0,0263 | 76,6 |
| Faster R-CNN | 0,9865 | 0,9866 | 0,0350 (*) | 581,1 |
| ViT (grand) | 0,9820 | 0,9818 | 0,0596 | 254,6 |
| VGG16 (fine-tuning) | 0,9710 | 0,9708 | 0,0909 | 154,0 |
| AlexNet (fine-tuning) | 0,9512 | 0,9506 | 0,1430 | 68,0 |
| ViT (tuto) | 0,8660 | 0,8644 | 0,4174 | 94,3 |

(Voir le graphique de comparaison dans le notebook.)

(*) La loss de Faster R-CNN est la somme de plusieurs pertes de détection (RPN et tête de détection) : elle n'est pas comparable à l'entropie croisée des autres modèles. Seules l'accuracy et le F1-score sont directement comparables.

Résultats issus d'une seule exécution par modèle (pas de moyenne sur plusieurs graines) : des variations d'environ ±0,1 point sont normales.

## 5. Analyse

### Partie 1 : CNN, Faster R-CNN, VGG16 et AlexNet

- **Le CNN entraîné de zéro est le meilleur modèle** : meilleure accuracy (99,18 %), entraînement rapide (≈ 77 s) et seulement 421 642 paramètres.
- **Faster R-CNN** atteint 98,65 % mais demande environ 7,6 fois plus de temps. C'est un détecteur d'objets (localisation et classification) : pour classer un seul chiffre par image, il est surdimensionné. Il a de plus été entraîné sur moins de données (10 000 images, 2 epochs), ce qui rend la comparaison moins équitable.
- **VGG16 (97,10 %) et AlexNet (95,12 %)** restent en retrait. Leurs convolutions, gelées, ont été apprises sur ImageNet (photos couleur), très différent de MNIST ; les images 28×28 ont de plus été agrandies à 64×64, bien en dessous des 224×224 pour lesquels ces réseaux sont conçus (AlexNet y perd davantage de détail spatial). Seulement 2 epochs ont été effectués, et la loss baissait encore.
- **Conclusion** : le transfert d'apprentissage est surtout utile avec peu de données ou des images proches du domaine d'origine. Sur MNIST, un petit CNN dédié est le meilleur compromis précision / coût.

### Partie 2 : Vision Transformer

- **ViT du tutoriel (1 642 paramètres) : 86,60 %.** Le modèle apprend bien (la loss baisse à chaque epoch) mais il est minuscule (vecteurs de dimension 8) et l'accuracy progressait encore à la fin de l'entraînement. Le tutoriel annonce environ 80 % ; l'écart est possiblement lié au retrait du softmax superflu (hypothèse non testée séparément).
- **ViT plus grand (≈ 148 000 paramètres) : 98,20 %.** Passer de 1 642 à environ 148 000 paramètres fait gagner plus de 11 points : la capacité du modèle est déterminante. Il dépasse VGG16 et AlexNet fine-tunés, mais reste en dessous du CNN (99,18 %) et de Faster R-CNN (98,65 %), tout en étant environ 3,3 fois plus lent que le CNN.
- **Pourquoi le CNN reste meilleur ici** : un CNN intègre des a priori adaptés aux images (filtres locaux, partage des poids), alors qu'un ViT doit les apprendre à partir des données avec l'attention. Les ViT sont surtout avantageux avec de grands modèles et beaucoup de données.
- Les temps du ViT sont gonflés par l'implémentation pédagogique du tutoriel (boucle sur les têtes d'attention), peu optimisée pour le GPU.

## 6. Limites

- Une seule exécution par modèle, sans graine fixée.
- Nombre d'epochs et jeux de données différents selon les modèles (Faster R-CNN : sous-ensemble de 10 000 / 2 000 images ; modèles pré-entraînés et ViT (tuto) : 2 ou 5 epochs).
- Optimiseurs et taux d'apprentissage non harmonisés entre les modèles.
- Pas de recherche systématique d'hyperparamètres.

## 7. Synthèse personnelle

Cet atelier m'a permis de prendre en main PyTorch sur GPU : charger les données (`Dataset`, `DataLoader`), définir un réseau (`nn.Module`), écrire la boucle d'entraînement (`zero_grad`, `backward`, `step`) et évaluer les modèles avec l'accuracy, le F1-score, la loss et le temps.

J'ai construit un CNN pour MNIST et vérifié à la main, avec les formules du cours, les dimensions et le nombre de paramètres (421 642). C'est lui qui a obtenu le meilleur résultat : 99,18 %.

J'ai compris que Faster R-CNN est un détecteur d'objets, trop lourd pour classer un seul chiffre (98,65 %, environ 7,6 fois plus lent), et que le transfert d'apprentissage avec VGG16 (97,10 %) et AlexNet (95,12 %) est surtout utile quand les données sont peu nombreuses ou proches d'ImageNet.

En construisant le Vision Transformer étape par étape (patchs, token de classe, encodage positionnel, attention multi-têtes), j'ai vu comment l'attention du cours s'applique aux images. Ses performances dépendent fortement de sa taille : 86,60 % avec 1 642 paramètres, 98,20 % avec environ 148 000.

Enfin, j'ai appris à comparer des modèles honnêtement, en tenant compte des différences de données, d'epochs et de définition de la loss.
