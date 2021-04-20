#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Résolution du problème d'optimisation quadratique "à la main"
"""
# Todo : Voir les problèmes avec le ksi (essayer à la main et comparer avec le code)

# ----------------------------------------------------------------------------------------------------------------------
# Imports
from copy import deepcopy
from Reseau import *


# ----------------------------------------------------------------------------------------------------------------------
# Résolution
def resoudre(reseau):
    """ Trouve le flot optimal du réseau """
    epsilon = 0
    # Initialisation
    liste_dependance = liens_de_dependance(reseau)
    nouveau_flot = situation_initiale(reseau, liste_dependance)
    nouveau_cout = reseau.cout(nouveau_flot)
    init = True
    # Optimisation de la situation
    while init or nouveau_cout + epsilon < cout:
        init = False
        # On sauvegarde les anciens
        flot = nouveau_flot
        cout = nouveau_cout
        # On fait une nouvelle vague d'optimisation
        nouveau_flot = optimiser(reseau, liste_dependance, deepcopy(flot))
        nouveau_cout = reseau.cout(nouveau_flot)
    return flot


def optimiser(reseau, liste_dependance, flot):
    """ Optimise le réseau """
    # On récupère les dérivées...
    derivees = get_derivees(reseau, liste_dependance, flot)
    # ... Que l'on trie dans l'ordre décroissant
    indices_derivees = tri_derivees(derivees)
    # On optimise les dérivées successivement
    for indice in indices_derivees:
        liste = liste_dependance[indice]
        ksi = get_ksi(reseau, liste, flot)
        flot = modifier_flot(liste, flot, ksi)
    return flot


# ----------------------------------------------------------------------------------------------------------------------
# Initialisation
def liens_de_dependance(reseau):
    """ Renvoie les liens de dépendances (i.e. les boucles) du réseau """
    liste_dependance = []
    d = dict()
    # On parcourt les villes à la recherche des boucles du graphe
    for index_depart, ville in enumerate(reseau.villes):
        for route in ville.routes_sortantes:
            index_route = reseau.routes.index(route)
            index_arrivee = reseau.villes.index(route.ville_arrivee)
            if index_arrivee in d.keys():
                # Si une route arrive à la même ville qu'une route précédente : on a une boucle !
                chemin_1 = d[index_arrivee]
                if index_depart in d.keys():
                    chemin_2 = d[index_depart]
                else:
                    chemin_2 = []
                chemin_2 = chemin_2 + [str(index_route)]
                chemin_2 = ["-" + index for index in chemin_2]
                liste_dependance.append(chemin_1 + chemin_2)
            else:
                # Sinon, on continue de construire les boucles
                if index_depart in d.keys():
                    # Si un chemin a déjà été créé, on le continue
                    chemin_precedent = d[index_depart]
                else:
                    # Sinon, on en créé un nouveau
                    chemin_precedent = []
                d[index_arrivee] = chemin_precedent + [str(index_route)]
    return liste_dependance


def situation_initiale(reseau, liste_dependance):
    # nombre_villes_ouvertes = 0
    # for ville in reseau:
    #     if ville.ouverte:
    #         nombre_villes_ouvertes += 1
    # flots = [0] * len(nombre_villes_ouvertes)
    flots = [0] * len(reseau.routes)
    for info_route in liste_dependance[-1]:
        negatif = info_route[0] == "-"
        index_route = abs(int(info_route))
        if negatif:
            flots[index_route] = 1
    return flots


# ----------------------------------------------------------------------------------------------------------------------
# Obtention du ksi
def get_ksi(reseau, liste, flot):
    """ Renvoie le ksi valide """
    # On récupère le ksi idéal...
    ksi = get_ksi_ideal(reseau, liste, flot)
    # ...auquel on applique les contraintes de chaque route
    for info_route in liste:
        ksi = get_ksi_contraint(info_route, flot, ksi)
    return ksi


def get_ksi_ideal(reseau, liste, flot):
    """ Renvoie le ksi ideal pour la liste donnée (Le ksi idéal pour optimiser UN cycle) """
    # Fonction à optimiser : f(ksi) = J*ksi + K = 0 donc ksi = - K / J (si un tel ksi est possible)
    K = 0
    J = 0
    for info_route in liste:
        negatif = info_route[0] == "-"
        if negatif:
            signe = -1
        else:
            signe = 1
        index_route = abs(int(info_route))
        route = reseau.routes[index_route]
        a, b = route.coutA, route.coutB
        K += signe * (2 * a * flot[index_route] + b)
        J += 2 * a
    ksi_ideal = -K / J
    return ksi_ideal


def get_ksi_contraint(info_route, flot, ksi):
    """ Renvoie le ksi auquel on a appliqué les contraintes de la route *info_route* """
    negatif = info_route[0] == "-"
    index_route = abs(int(info_route))
    x = flot[index_route]
    if negatif:
        ksi_max = x
        ksi_min = x - 1
    else:
        ksi_max = 1 - x
        ksi_min = -x
    if ksi > ksi_max:
        ksi = ksi_max
    if ksi < ksi_min:
        ksi = ksi_min
    return ksi


# ----------------------------------------------------------------------------------------------------------------------
# Modification du flot
def modifier_flot(liste, flot, ksi):
    """ Modifie le flot en fonction du ksi """
    for info_route in liste:
        negatif = info_route[0] == "-"
        index_route = abs(int(info_route))
        if negatif:
            signe = -1
        else:
            signe = 1
        flot[index_route] += signe * ksi
    return flot


# ----------------------------------------------------------------------------------------------------------------------
# Gestion des dérivées
def get_derivees(reseau, liste_dependance, flot):
    """ Renvoie les dérivées actuelles d'un réseau (pour un certain flot)"""
    derivees = []
    for i, liste in enumerate(liste_dependance):
        d = 0
        for info_route in liste:
            index_route = abs(int(info_route))
            route = reseau.routes[index_route]
            a, b = route.coutA, route.coutB
            negatif = info_route[0] == "-"
            if negatif:
                signe = -1
            else:
                signe = 1
            d += signe * 2 * a * flot[index_route] + b
        derivees.append(d)
    return derivees


def tri_derivees(derivees):
    """ Trie les dérivées par ordre décroissant et renvoie la liste des indices correspondants à ce tri """
    d = deepcopy(derivees)
    indices = []
    while len(d) != 0:
        # Todo : à optimiser
        d_max = max(d)
        indice_max = derivees.index(d_max)
        indices.append(indice_max)
        d.pop(d.index(d_max))
    return indices


# ----------------------------------------------------------------------------------------------------------------------
# Tests
if __name__ == '__main__':
    ex1 = Reseau.charger([7,
                          [0, 1, 1, 1],
                          [0, 2, 1, 1],
                          [0, 2, 1, 1],
                          [0, 3, 1, 1],
                          [1, 6, 1, 1],
                          [2, 4, 1, 1],
                          [2, 5, 1, 1],
                          [3, 5, 1, 1],
                          [4, 6, 1, 1],
                          [5, 6, 1, 1]],
                         ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    # r = ex1
    # l = liens_de_dependance(r)
    # print(l)
    # flot = situation_initiale(r, l)
    # print(flot)
    # r.dessiner('flot', flot)

    # r = exemples['simple']

    r = exemples['Savigny2']

    # print(liens_de_dependance(r))

#    flot = resoudre(r.egoiste())
#    r.dessiner('flot', flot)
#    time.sleep(3)
#    r.resoudre_qp(egoiste=True)
#    time.sleep(3)
#    r.dessiner()