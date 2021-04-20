#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import argv
from Carte import Carte

def stats_afficher(carte):
    total_distance = 0.0
    connectes = [[False] * len(carte.nodes)] * len(carte.nodes)
    total_sens_unique = 0

    for way in carte.ways:
        if way["oneway"]:
            total_sens_unique += 1
        for j, node in enumerate(way["nodes"][:-1]):
            # Le dernier nœud n'a pas de prochain nœud.
            total_distance += carte.distance(node, way["nodes"][j + 1])
            connectes[node][way["nodes"][j + 1]] = True

    print("Nombre de routes : " + str(len(carte.ways)))
    print("Longueur moyenne : " + str(total_distance / float(len(carte.ways))))

    total_connectes = 0
    for _, ligne in enumerate(connectes):
        for _, connexion in enumerate(ligne):
            if connexion:
                total_connectes += 1

    # La densité vaut 1 si tous les nœuds sont connectés entre eux.
    print("Densité moyenne  : " +
          str(total_connectes / float(len(carte.nodes)**2)))


if __name__ == "__main__":
    filename = "test.xml"
    if len(argv) > 1:
        filename = argv[1]

    carte = Carte(filename)
    stats_afficher(carte)
