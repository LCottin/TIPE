#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Reseau import *
import numpy as np


def creer_table(reseau):
    """
    Contruction de

     2*A | -C.T
    -----|------
      C  |   0
        
    Les routs sont sur les colonnes, 
    les villes sur les abscisses  
    
    """
    
    reseau.update_matrices()
    taille = reseau.nombre_inconnues + reseau.nombre_contraintes
    table = np.zeros((taille, taille))
    table[:reseau.nombre_inconnues, :reseau.nombre_inconnues] = 2 * reseau.A
    table[reseau.nombre_inconnues:, :reseau.nombre_inconnues] = reseau.C
    table[:reseau.nombre_inconnues, reseau.nombre_inconnues:] = - reseau.C.T
    return table


def resoudre(reseau, egoiste=False, dessiner=True):
    if egoiste:
        reseau_etudie = reseau.egoiste()
    else:
        reseau_etudie = reseau
    table = creer_table(reseau_etudie)
    print(table)
    matrice = np.concatenate((-reseau_etudie.B, reseau_etudie.D))
    print(matrice)
    solution = np.linalg.lstsq(table, matrice)[0][:reseau_etudie.nombre_inconnues]
    print(np.linalg.lstsq(table, matrice))
    if dessiner:
        reseau.dessiner_flot(solution)
    return solution,table,matrice


if __name__ == '__main__':

    reseau = exemples['braess0']        # Fonctionne
#    Reseau.dessiner(reseau)
    sol, table,matrice = resoudre(reseau,dessiner=False)

    # reseau = exemples['braess1']      # Foncionne
    # resoudre(reseau, egoiste=True)

    # reseau = exemples['Savigny']      # Problème : Flots négatifs
    # resoudre(reseau)