#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv
import os


def get_variable_associee(nom_variable):
    # Recupère la variable associée à nom_variable
    if nom_variable[0] == "x":
        nom_autre_variable = "L"
    else:
        nom_autre_variable = "x"
    if nom_variable[1] == "'":
        nom_autre_variable += nom_variable[2]
    else:
        nom_autre_variable += "'" + nom_variable[1]
    return nom_autre_variable


class Wolfe:

    def __init__(self, A, B, C, D, stocker=False):
        """
        Le but est de maximiser
        f(x) = xT.A.x + xtB
        où C.x = D
        """
        # Matrices
        self.A = A
        self.B = B.reshape(1, len(B))
        self.C = C
        self.D = D.reshape(len(D), 1)

        """
        Construction du tableau :
        |  X   L'   L    W    W' |
        |------------------------|------|---
        |  2A  I  -C^T   N    0  | -B^T | W
        |  C   0    0    0    N  |   D  | W'
        |------------------------|------|---
        |  S   S    S    0    0  |   S  |

        Où  I est la matrice identité
            N est I où chaque ligne a le signe de la dernière colonne
            S est la somme des lignes au-dessus (le signe de chaque ligne dans la somme est déterminé 
            par l'opposé du signe de la dernière colonne)
        """

        # Autant de x que de lignes/col dans A
        # Les x sont liés aux L'
        self.nombre_de_x = len(self.A)
        # Autant de x' que de valeurs dans d
        # Les x' sont liés aux L
        self.nombre_de_x_prime = len(self.D)

        self.colone_variable = []
        self.colone_variable.extend(["x" + str(i + 1) for i in range(self.nombre_de_x)])
        self.colone_variable.extend(["L'" + str(i + 1) for i in range(self.nombre_de_x)])
        self.colone_variable.extend(["L" + str(i + 1) for i in range(self.nombre_de_x_prime)])
        self.colone_variable.extend(["w" + str(i + 1) for i in range(self.nombre_de_x)])
        self.colone_variable.extend(["w'" + str(i + 1) for i in range(self.nombre_de_x_prime)])

        self.ligne_variable = []
        self.ligne_variable.extend(["w" + str(i + 1) for i in range(self.nombre_de_x)])
        self.ligne_variable.extend(["w'" + str(i + 1) for i in range(self.nombre_de_x_prime)])

        nombre_lignes = self.nombre_de_x + self.nombre_de_x_prime + 1
        nombre_colonnes = self.nombre_de_x * 3 + self.nombre_de_x_prime * 2 + 1
        self.table = np.zeros((nombre_lignes, nombre_colonnes))

        # Construction
        premiere_ligne = slice(self.nombre_de_x)
        seconde_ligne = slice(self.nombre_de_x, self.nombre_de_x + self.nombre_de_x_prime)
        derniere_ligne = [-1]

        premiere_colonne = slice(self.nombre_de_x)
        seconde_colonne = slice(self.nombre_de_x, 2 * self.nombre_de_x)
        troisieme_colonne = slice(2 * self.nombre_de_x, 2 * self.nombre_de_x + self.nombre_de_x_prime)
        quatrieme_colonne = slice(2 * self.nombre_de_x + self.nombre_de_x_prime,
                                  3 * self.nombre_de_x + self.nombre_de_x_prime)
        cinquieme_colonne = slice(3 * self.nombre_de_x + self.nombre_de_x_prime,
                                  3 * self.nombre_de_x + 2 * self.nombre_de_x_prime)
        derniere_colonne = [-1]

        # Première colonne : x
        self.table[premiere_ligne, premiere_colonne] = 2 * self.A
        self.table[seconde_ligne, premiere_colonne] = self.C
        # Seconde colonne : L'
        self.table[premiere_ligne, seconde_colonne] = np.identity(self.nombre_de_x)
        # Troisième colonne : L
        self.table[premiere_ligne, troisieme_colonne] = -self.C.T
        # Dernière colonne
        self.table[premiere_ligne, derniere_colonne] = -self.B.T
        self.table[seconde_ligne, derniere_colonne] = self.D
        # Quatrième et cinquieme colonne : W et W'
        signe = np.sign(self.table[:-1, derniere_colonne]).flatten()
        signe[signe == 0] = 1
        N = signe * np.identity(self.nombre_de_x + self.nombre_de_x_prime)
        self.table[premiere_ligne, quatrieme_colonne] = N[premiere_ligne, :self.nombre_de_x]
        self.table[seconde_ligne, cinquieme_colonne] = N[seconde_ligne,
                                                       self.nombre_de_x: self.nombre_de_x + self.nombre_de_x_prime]
        # Dernière ligne
        somme_lignes = np.sum(self.table[:-1, :] * -signe.reshape(self.nombre_de_x + self.nombre_de_x_prime, 1), axis=0)
        self.table[derniere_ligne, premiere_colonne] = somme_lignes[premiere_colonne]
        self.table[derniere_ligne, seconde_colonne] = somme_lignes[seconde_colonne]
        self.table[derniere_ligne, troisieme_colonne] = somme_lignes[troisieme_colonne]
        self.table[derniere_ligne, derniere_colonne] = somme_lignes[derniere_colonne]

        # Stockage et affichage
        self.stocker = stocker
        if self.stocker:
            self.file_path = os.path.join('.', 'csv_files', 'wolfe.csv')
            path = self.file_path
            i = 0
            while os.path.exists(self.file_path):
                self.file_path = path[:-4] + str(i) + path[-4:]
                i += 1

    # ------------------------------------------------------------------------------------------------------------------
    # Wolfe

    def wolfe_variable_entrante(self):
        """
        Renvoie la liste des variables entrantes valides (= indices des colonnes)
        Il s'agit des x, L, L' qui ont la plus petite valeur strictement négative dans la dernière ligne
        """
        partition = self.table[-1, :-1]
        minimum = np.min(partition)
        if minimum > -10 ** (-10):
            return []
        liste_entrantes = np.argwhere(partition == minimum).flatten()
        return liste_entrantes

    def wolfe_variable_sortante(self, liste_entrantes):
        """
        Retourne une variable sortante valide (= ligne)

        On attibue au passage self.entrante et self.sortante (si valides)
        """
        # Pour déterminer la variable sortante, on prend le ratio entre la dernière colonne et les colonnes entrantes
        derniere_colonne = self.table[:-1, -1]
        derniere_colonne = derniere_colonne.reshape(len(derniere_colonne), 1)
        colonnes_entrantes = self.table[:-1, liste_entrantes]
        ratio = derniere_colonne / colonnes_entrantes
        ratio[colonnes_entrantes == 0] = np.inf
        # ... mais seulement pour les lignes où le ratio est positif (ou nul positif, i.e. 0 / <positif>)
        ratio[ratio < 0] = np.inf
        ratio[(ratio == 0) & (colonnes_entrantes <= 0)] = np.inf

        # En cas d'égalité, on choisit de préférence un w
        minimum = ratio.min()
        if np.isinf(minimum):
            return None
        else:
            indices = np.argwhere(ratio == minimum)
            liste_priorite = []
            for indice in indices[::-1]:
                if self.ligne_variable[indice[0]][0] == 'w':
                    liste_priorite = [indice] + liste_priorite
                else:
                    liste_priorite.append(indice)
            if len(liste_priorite) == 0:
                return None
            choisie = liste_priorite[0]
            self.entrante = liste_entrantes[choisie[1]]  # On confirme la colonne entrante
            self.sortante = choisie[0]  # On fixe la variable sortante
            return self.sortante

    def simplexe(self):
        """
        On effectue le pivot

        colonne entrante : self.entrante
        colonne sortante : self.sortante
        """
        # On divise la ligne du pivot par le pivot
        self.table[self.sortante] = self.table[self.sortante] / self.table[self.sortante, self.entrante]
        # On change les autres lignes
        autres_lignes = list(range(0, self.sortante)) + list(range(self.sortante + 1, self.table.shape[0]))
        colonne = self.table[autres_lignes, self.entrante].reshape(len(autres_lignes), 1)
        ligne = self.table[self.sortante, :].reshape(1, self.table.shape[1])
        self.table[autres_lignes, :] = self.table[autres_lignes, :] - colonne * ligne

        # On fait gaffe que les 0 dans le tableau soient bien des 0.
        self.table[(- 10 ** (-10) <= self.table) & (self.table <= 10 ** (-10))] = 0.0

        # On stocke la variable entrante dans la ligne correspondante
        self.ligne_variable[self.sortante] = self.colone_variable[self.entrante]

    def wolfe_routine(self):
        """
        On récupère une variable entrante, une variable sortante
        On stocke le tableau précédent et les variables sélectionnées dans le csv
        """
        while True:
            # On sélectionne les variables entrantes valides
            liste_entrantes = self.wolfe_variable_entrante()
            if len(liste_entrantes) == 0:
                print('Wolfe : Fin normale !')
                return None

            # On sélectionne une variable sortante valide
            # self.entrante sera l'index entrant valide, self.sortant sera l'index sortant valide
            sortante = self.wolfe_variable_sortante(liste_entrantes)
            if sortante is None:
                self.csv_write()
                print('Wolfe : Fin anormale ! Pas de variable sortante trouvée')
                print(self.A, self.B, self.C, self.D)
                raise ValueError

            # On stocke le tableau et les variables dans le csv
            self.csv_write(pivot=True)

            self.simplexe()

    # ------------------------------------------------------------------------------------------------------------------
    # Dantzig 

    def simplifier(self):
        # Simplifie la table de Wolfe pour faire une table de Dantzig

        # Simplifie les colonnes
        colonnes_W = slice(2 * self.nombre_de_x + self.nombre_de_x_prime, self.table.shape[1] - 1)
        self.table = np.delete(self.table, colonnes_W, axis=1)
        # Supprime aussi les variables
        self.colone_variable = np.array(self.colone_variable)
        self.colone_variable = np.delete(self.colone_variable, colonnes_W)
        self.colone_variable = list(self.colone_variable)

        # Simplifie les lignes
        # Supprime la dernière ligne
        self.table = np.delete(self.table, -1, axis=0)
        # Supprime les lignes en W
        lignes_W = []
        for ligne in range(self.table.shape[0]):
            nom_variable = self.ligne_variable[ligne]
            if nom_variable[0] == 'w':
                lignes_W.append(ligne)
        self.table = np.delete(self.table, lignes_W, axis=0)
        self.ligne_variable = np.array(self.ligne_variable)
        self.ligne_variable = np.delete(self.ligne_variable, lignes_W)
        self.ligne_variable = list(self.ligne_variable)

    def dantzig_ajout_derniere_ligne(self):
        # On ajoute la dernière ligne à la table de dantzig
        """
        Derniere ligne de la forme :

        |    X    L'    L    |
        |--------------------|-----|
        |  -B/2   0   -D^T/2 |  0  |
        """

        premiere_colonne = slice(0, self.nombre_de_x)
        troisieme_colonne = slice(2 * self.nombre_de_x, 2 * self.nombre_de_x + self.nombre_de_x_prime)

        derniere_ligne = np.zeros((1, self.table.shape[1]))
        derniere_ligne[0, premiere_colonne] = - self.B / 2
        derniere_ligne[0, troisieme_colonne] = - self.D.T / 2

        self.table = np.append(self.table, derniere_ligne, axis=0)

    def dantzig_est_standard(self):
        """
        Renvoie True si le tableau est standard, False sinon
        Un tableau est non-standard si une paire (x,L') ou (x',L) est non nulle (i.e. chaque variable de la paire est non-nulle)
        Notons que les x' sont constamment nuls.
        """

        self.standard = True
        # Si le tableau n'est pas standard, on stocke la liste des paires de variables qui ne sont pas dans la base
        self.paires_en_base = []

        # Les x' ne sont jamais en base : ils sont constamment nuls
        # On ne s'intéresse donc pas aux couples (x', L)
        for variable in self.colone_variable[:self.nombre_de_x]:
            # Si une variable est dans la base (= est dans self.ligne_variable) ...
            if variable in self.ligne_variable:
                indice = self.ligne_variable.index(variable)
                valeur_variable = self.table[indice, -1]
                # ... et sa valeur dans la deniere colonne est non-nulle ...
                if valeur_variable != 0:
                    # ... on détermine l'autre variable de la paire
                    autre_variable = get_variable_associee(variable)
                    # Si elle est dans la base, on check sa valeur
                    if autre_variable in self.ligne_variable:
                        autre_indice = self.ligne_variable.index(autre_variable)
                        valeur_autre = self.table[autre_indice, -1]
                        # Si elle est non-nulle, le tableau n'est pas standard
                        if valeur_autre != 0:
                            self.standard = False
                            self.paires_en_base.append((variable, autre_variable))
        return self.standard

    def dantzig_variable_entrante_standard(self):
        # On fait entrer une variable primale (x)
        # Celle dont la duale associée (L') est la plus négative
        derniere_colonne = np.array(self.table[:-1, -1])
        pas_L_prime = []
        for nom_variable in self.ligne_variable:
            pas_L_prime.append(nom_variable[0:2] != "L'")
        derniere_colonne[pas_L_prime] = np.inf
        derniere_colonne[derniere_colonne >= 0] = np.inf
        indice_duale = np.argmin(derniere_colonne)
        if np.isinf(derniere_colonne[indice_duale]):
            self.over = True
            print('ending : Dantzig normal')
            return None
        self.duale = self.ligne_variable[indice_duale]
        primale = get_variable_associee(self.duale)
        self.entrante = self.colone_variable.index(primale)
        return self.entrante

    def dantzig_variable_sortante_standard(self):
        # On fait le ratio entre la dernière colonne et la colonne entrante
        colonne_entrante = self.table[:-1, self.entrante]
        derniere_colonne = self.table[:-1, -1]
        ratio = derniere_colonne / colonne_entrante
        # ... mais seulement pour les lignes où le ratio est positif (ou nul positif, i.e. 0 / <positif>)
        ratio[ratio < 0] = np.inf
        ratio[(ratio == 0) & (colonne_entrante <= 0)] = np.inf
        ratio[np.isnan(ratio)] = np.inf

        # On ne s'intéresse qu'aux ratios des lignes où une variable primale est dans la base
        # (ainsi qu'à la ligne de la duale associée à la primale entrante)
        pas_interessant = []
        for nom_variable in enumerate(self.ligne_variable):
            pas_interessant.append(nom_variable[0] == "L" and nom_variable != self.duale)
        ratio[pas_interessant] = np.inf

        self.sortante = np.argmin(ratio)
        if np.isinf(ratio[self.sortante]):
            self.over = True
            print('ending : Dantzig sortante')
            print('Standard :', self.standard)
            return None
        return self.sortante

    def dantzig_variable_entrante_non_standard(self):
        # On fait entrer la duale d'une paire entièrement hors base
        # On parcourt les duales (L' et L)
        L_prime_et_L = slice(self.nombre_de_x, 2 * self.nombre_de_x + self.nombre_de_x_prime)
        for indice, duale in enumerate(self.colone_variable[L_prime_et_L]):
            # Si la duale n'est pas dans la base...
            if duale not in self.ligne_variable:
                primale = get_variable_associee(duale)
                # ... et que sa primale associée n'y est pas non plus ...
                if primale not in self.ligne_variable:
                    # ... alors la variable entrante est la duale ...
                    entrante = self.colone_variable.index(duale)
                    self.entrante = entrante
                    self.liste_entrantes.append(self.entrante)
                    # Note : Si plusieurs paires sont hors base, on sélectionne la première
        if self.liste_entrantes == []:
            raise AssertionError('Nothing found')

    def dantzig_variable_sortante_non_standard(self):
        self.over = False
        # On fait le ratio entre la dernière colonne et la colonne entrante
        colonne_entrante = self.table[:-1, self.entrante]
        derniere_colonne = self.table[:-1, -1]
        ratio = derniere_colonne / colonne_entrante
        # ... mais seulement pour les lignes où le ratio est positif (ou nul positif, i.e. 0 / <positif>)
        ratio[ratio < 0] = np.inf
        ratio[(ratio == 0) & (colonne_entrante <= 0)] = np.inf
        ratio[np.isnan(ratio)] = np.inf

        # On ne s'intéresse qu'aux ratios des lignes où une variable primale est dans la base
        # (ainsi qu'à la ligne de la duale dont la paire est entièrement dans la base)
        pas_interessant = [False] * len(self.ligne_variable)
        for indice, nom_variable in enumerate(self.ligne_variable):
            # Les L n'ont jamais leur paire dans la base, on ne les prend pas en considération
            if nom_variable[0:2] == "L'":
                for paire_en_base in self.paires_en_base:
                    if nom_variable != paire_en_base[0] and nom_variable != paire_en_base[1]:
                        pas_interessant[indice] = True
                        break

        ratio[pas_interessant] = np.inf
        self.sortante = np.argmin(ratio)
        if np.isinf(ratio[self.sortante]):
            self.over = True
            print('ending : Dantzig sortante')
            print('Standard :', self.standard)
            return None
        return self.sortante

    def dantzig_routine(self):
        self.dantzig_est_standard()
        if self.standard:
            self.dantzig_variable_entrante_standard()
            if self.over:
                return None
            self.dantzig_variable_sortante_standard()
            if self.over:
                return None
        else:
            # Si il y a égalité entre deux variables entrantes :
            # On parcours chacune des variables entrantes
            self.liste_entrantes = []
            self.dantzig_variable_entrante_non_standard()
            if len(self.liste_entrantes) == 0:
                return None
            for entrante in self.liste_entrantes:
                self.entrante = entrante
                self.dantzig_variable_sortante_non_standard()
                if self.over:
                    continue
                else:
                    break
            # Si on a tout parcouru, mais qu'aucune n'est valide, on arrête là
            if self.over:
                return None
        # Affiche si le tableau est standard
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Stand.', str(self.standard)])
        self.csv_write(pivot=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Résoudre
    def resoudre(self):
        self.over = False
        
        # Lancement de Wolfe
        self.wolfe_routine()
        self.csv_write()
        
        # Préparation de Dantzig
        self.simplifier()
        self.csv_write()
        
       
    def extraire(self):
        # Extrait les solutions x1, x2, etc.
        solution = [0] * self.nombre_de_x
        # Pour chaque x ...
        for variable in self.colone_variable[:self.nombre_de_x]:
            # ... on regarde si il correspond à une ligne
            if variable in self.ligne_variable:
                # Si oui, on récupère sa valeur...
                indice = self.ligne_variable.index(variable)
                valeur = self.table[indice, -1]
                # Et on la met dans la case correspondante
                numero = int(variable[1:]) - 1
                solution[numero] = valeur
        # Ceux qui ne correspondent à aucune ligne valent 0
        return solution

    # ------------------------------------------------------------------------------------------------------------------
    # CSV (excel)
    
    def csv_write(self, pivot=False):
        if self.stocker:
            with open(self.file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.colone_variable)
                for ligne, valeurs in enumerate(self.table):
                    valeurs = [round(valeurs[i], 2) for i in range(len(valeurs))]
                    if ligne < len(self.ligne_variable):
                        writer.writerow(list(valeurs) + [self.ligne_variable[ligne]])
                    else:
                        writer.writerow(list(valeurs))
                writer.writerow(('',))

                if pivot:
                    self.csv_write_pivot()
                else:
                    writer.writerow(('',))
                    writer.writerow(('',))

                writer.writerow(('',))

    def csv_write_pivot(self):
        if self.stocker:
            with open(self.file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                nom_entrante = self.colone_variable[self.entrante]
                nom_sortante = self.ligne_variable[self.sortante]
                writer.writerow(('entre', nom_entrante))
                writer.writerow(('sort', nom_sortante))
                