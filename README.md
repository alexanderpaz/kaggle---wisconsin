# wisconsin

## Objetivo

Este es una demo de desarrollo de un modelo predictivo con sus respectivas etapas, utilizando para ello python.
El objetivo es mostrar la versatilidad de diferentes librerías tanto de visualización, ingeniería de datos y de machine learning para afrontar un problema como éste.

## El Dataset

El dataset utilizado es de libre acceso y pertenece a la universidad de Wisconsin. Las características del Dataset se calculan desde imágenes digitalizadas de biopsias de masas detectadas en mama.
Se describen entonces características del núcleo celular presente en la imagen tridimensional obtenida, tal como indica [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

Este dataset se puede obtener también en FTP: ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/ y en el UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

## Atributos

1. ID number
2. Diagnosis (M = malignant, B = benign) 
3-32. Diez atributos del núcleo celular:

a) radius (mean of distances from center to points on the perimeter) 
b) texture (standard deviation of gray-scale values) 
c) perimeter 
d) area 
e) smoothness (local variation in radius lengths) 
f) compactness (perimeter^2 / area - 1.0) 
g) concavity (severity of concave portions of the contour) 
h) concave points (number of concave portions of the contour) 
i) symmetry 
j) fractal dimension ("coastline approximation" - 1)

Para cada una de ellas, se tiene: la media, la desviación standar y el "peor" caso (mas gránde). Lo que resulta en 30 atributos.

Distribución de clases: 357 benignos, 212 malignos
