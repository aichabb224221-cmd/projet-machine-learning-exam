# Projet de clustering — 3 notebooks et Streamlit

Le dossier contient les trois notebooks demandés, leurs CSV et une première application locale Streamlit. Les notebooks Heart Disease et Bank Marketing sont les versions corrigées que tu as exécutées; leurs sorties de clustering sont conservées. Le notebook des graines est ajouté et suit la même normalisation ligne par ligne.

## Résultats lus dans tes notebooks

- Heart Disease (5 groupes) : silhouette K-means **0,302**; CAH **0,258**.
- Bank Marketing (5 groupes) : silhouette K-means **0,561**; CAH **0,497**.
- Graines : trois groupes; la CAH est présélectionnée pour suivre la consigne du professeur. Exécute le notebook pour obtenir son score.

## Lancer l’application localement

1. Installer Python 3.10 ou plus récent.
2. Ouvrir un terminal dans ce dossier.
3. Installer les dépendances : `pip install -r requirements.txt`.
4. Lancer : `streamlit run app.py`.

L’application permet de choisir un jeu, une méthode et le nombre de groupes; elle affiche les répartitions, la silhouette, une projection ACP et les profils moyens. Pour un nouvel exemple en CAH, l’affectation au centre moyen le plus proche est une approximation : la CAH n’a pas de méthode native pour prédire de nouvelles lignes.

On pourra parcourir `app.py` ensemble, étape par étape, puis la lancer et vérifier l’interface avant toute publication en ligne.
