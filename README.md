=== ENGLISH VERSION BELOW ===

Resolution approchée d'équations différentielles fortement oscillantes via l'utilisation du Machine Learning

### Averaging_ML ###

Code Python permettant l'approximation de solutions d'équations différentielles de la forme $$y'(t) = f\left(t/\epsilon,y(t)\right)$$ où $f$ est une fonction périodique en sa première variable. Stratégie: décomposition "slow-fast", composition d'un flot associé à un champ moyenné (dynamique lente) avec un générateur de fortes oscillations (dynamique rapide).

Approximation du champ moyenné: dynamique lente, EDO autonome, analyse rétrograde (problème étudié dans le cas autonome)
Approximation du générateur de fortes oscillations: dynamique rapide, auto-encodeur.

Etapes suivies:

- Création de données
- Entraînement des réseaux de neurones
- Intégration numérique: décomposition "slow-fast" et schéma Micro-Macro avec les champs appris.

Possibilités avec les fonctions du programme:

- Création de données
- Apprentissage via minimisation de Loss
- Intégration numérique slow-fast & Micro-Macro
- Courbes de convergence: erreur numérique (méthode slow-fast ou Micro-Macro)
- Courbes de convergence: test du caractère UA de la méthode Micro-Macro avec les champs appris.
- Courbes de convergence: approximation du champ moyenné modifié ainsi que du générateur de fortes oscillations, erreurs par rapport à $\epsilon$.

### Averaging_Autonomous_ML ###

Code Python permettant l'approximation de solutions d'équations différentielles de la forme $$y' = (1/\epsilon)Ay + g(y)$$, où $A$ est une matrice donc les valeurs prores sont des multiples de $i$. Avec un changement de variable (et donc de fonction), il est possible de se ramener au cas précédent. Mais une nouvelle approche est testée, en regardant ici la solution comme composition de deux flots (un pour la dynamique lente, l'autre pour la dynamique rapide). 

Apprentissage des deux flots via des réseaux de neurones "classiques" (pas d'auto-encodeur cette fois).

Etapes suivies:

- Création de données
- Entraînement des réseaux de neurones
- Intégration numérique: méthode numérique.

Possibilités avec les fonctions du programme:

- Création de données
- Apprentissage via minimisation de Loss
- Intégration numérique via une méthode numérique choisie
- Courbes de convergence: erreur numérique (méthode numérique)

Une comparaison entre les deu programmes peut-être réalisée.

