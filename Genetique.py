#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fonctionnement de l'algo :

- On créé une population de flots.

- On leur attribue un score en fonction de leur cout total
- On sélectionne les flots en se basant sur leur score
- On les reproduit ensemble
- On applique des mutations génétiques sur les enfants
- On recommence à partir de (2) (score...)
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from Reseau import *



class Population:

    def __init__(self, reseau, demographie=200):
        self.demographie = demographie
        self.reseau = reseau
        self.reseau.update_matrices()
        # Creation des flots
        self.flots = [reseau.normaliser([0] * self.reseau.nombre_inconnues)] * self.demographie
        # Création de la table des flots
        self.table = self.creer_table(self.flots)
        # On attribue un score
        self.score = np.array([])
        uns = np.ones((self.reseau.nombre_inconnues, 1))
        self.cout_maximal = (reseau.cout(uns))
        # On garde le meilleur sous le coude
        self.meilleur_flot = self.flots[0]
        self.meilleur_score = 0
        # On stocke les scores qu'on veut grapher
        self.meilleur_scores = []
        self.moyennes = []
        # On attribue les premiers scores
        self.update_scores()

    # ------------------------------------------------------------------------------------------------------------------
    # Outils
    @staticmethod
    def creer_table(flots):
        return np.vstack([deepcopy(flot) for flot in flots])

    # ------------------------------------------------------------------------------------------------------------------
    # Gestion des scores

    def update_scores(self):
        self.score = self.obtenir_scores()
        # Stocke le meilleur
        if self.score.max() > self.meilleur_score:
            indice_meilleur = self.score.argmax()
            self.meilleur_flot = self.flots[indice_meilleur]
            self.meilleur_score = self.score[indice_meilleur]
        # Stocke les scores à grapher
        self.meilleur_scores.append(self.meilleur_score)
        self.moyennes.append(self.score.mean())

    def obtenir_scores(self):
        cout = [self.reseau.cout(self.table[i, :]) for i in range(self.table.shape[0])]
        cout = np.array(cout)
        # Le score est cout_maximal - cout
        score = self.cout_maximal - cout
        self.score = score
        return score

    # ------------------------------------------------------------------------------------------------------------------
    # Gestion de génération
    def prochaine_generation(self):
        enfants = np.zeros(shape=(self.demographie,), dtype='object')
        
        # On choisit les parents en se basant sur le score
        # On se débarasse des pires
        self.score = self.score - self.score.mean()
        print(self.score)
        self.score[self.score < 0] = 0
        # On choisit les parents
        somme_scores = self.score.sum()
        if somme_scores != 0:
            proba = self.score / somme_scores
            print(proba)
        else:
            proba = None
        flots = np.array(self.flots)
        # 25% des enfants sont des parents ayant subis une petite mutation
        indices_heritage = np.random.choice(np.arange(0, len(self.flots)), (int(self.demographie * 25/100),), p=proba)
        flots_heritage = list(flots[indices_heritage])
        # Les autres 75% sont créés par reproduction
        indices_parents = np.random.choice(np.arange(0, len(self.flots)), (2, self.demographie - int(self.demographie * 25/100)), p=proba)
        parents = flots[indices_parents]
        

        # On regroupe tous les enfants avant la mutation
        enfants[:int(self.demographie * 25 / 100)] = flots_heritage
        enfants[int(self.demographie * 25 / 100):] = self.reproduction(parents)

        # Mutation
        enfants = self.mutation(enfants)

        # Création de la nouvelle population
        self.flots = enfants
        self.table = self.creer_table(self.flots)
        self.update_scores()

    # ------------------------------------------------------------------------------------------------------------------
    # Outils génétiques
    def reproduction(self, parents):
        # On fait deux tables avec les flots de tous les parents
        table1 = self.creer_table(parents[0])
        table2 = self.creer_table(parents[1])
        table_bebes = np.zeros(table1.shape)
        # Chaque bébé flot est un mélange aléatoire entre les parents
        rand = np.random.rand(table_bebes.size).reshape(table_bebes.shape)
        table_bebes[rand < 0.5] = table1[rand < 0.5]
        table_bebes[rand >= 0.5] = table2[rand >= 0.5]
        babies = [self.reseau.normaliser(table_bebes[i]) for i in range(parents.shape[1])]
        return babies

    def mutation(self, enfants):
        table_enfants = np.vstack([enfant for enfant in enfants])
        rand = np.random.rand(table_enfants.size).reshape(table_enfants.shape)
        # 50% chance of a slight mutation (mostly between -0.2 and 0.2)
        mask_faible = rand < 0.5
        # Note : the min(max()) thing is here to lower the st.var. based on the difference between the max and min score
        table_enfants[mask_faible] += np.random.normal(scale=min(0.1, max(0.01, self.score.max() - self.score.min())),
                                                       size=table_enfants[mask_faible].size)
        # Be sure it's between 0 and 1
        mask0 = table_enfants < 0
        table_enfants[mask0] = 0
        mask1 = table_enfants > 1
        table_enfants[mask1] = 1
        # 10% chance of getting a new road stream
        mask_nouveau = (rand >= 0.5) * (rand < 0.6)
        table_enfants[mask_nouveau] = np.random.rand(table_enfants[mask_nouveau].size)
        # Reform the child
        child = [self.reseau.normaliser(table_enfants[i]) for i in range(len(enfants))]
        return child

    # ------------------------------------------------------------------------------------------------------------------
    # Evolution
    def evolue(self, nombre_generations=50, afficher_graphique=True):
        """
        Fait évoluer la population
        """
        for i in range(nombre_generations):
            print('\n')
            print('Generation n° : '+str(i+1))
            self.prochaine_generation()
        # Make the graph
        if afficher_graphique :
            meilleurs_scores = self.cout_maximal - np.array(self.meilleur_scores)
            moyennes = self.cout_maximal - np.array(self.moyennes)
            plt.plot(range(len(meilleurs_scores)), meilleurs_scores, label='Meilleurs flots')
            plt.plot(range(len(moyennes)), moyennes, label='Moyenne des flots')
            plt.legend()
            plt.xlabel('Generation')
            plt.ylabel('Cout')


if __name__ == '__main__':
    r = exemples['Savigny_reduit']
    r.resoudre_genetique(dessiner=False)