#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Chargement de carte OpenStreetMap """

import xml.dom.minidom as dom
from sys import argv

import matplotlib.pyplot as plt


def get_map_bounds(tree):
    """ Récupère les dimensions de la carte.

    :tree: Arbre XML à analyser. """
    bounds = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    for child in tree.documentElement.childNodes:
        if isinstance(child, dom.Text):  # sinon tagName renvoie une erreur
            continue

        if child.tagName == "bounds":
            info = child.attributes
            bounds["x"] = float(info["minlat"].value)
            bounds["y"] = float(info["minlon"].value)
            bounds["w"] = float(info["maxlat"].value) - bounds["x"]
            bounds["h"] = float(info["maxlon"].value) - bounds["y"]
            break

    return bounds


def parse_node(element, bounds, node_ids, nodes):
    """ Récupère les informations d'un nœud.

    :element: Nœud XML à analyser.
    :bounds: On normalise les coordonées dans [0, 1] avec les dimensions de la carte.

    :node_ids: (modifié) Dictionnaire qui associe au nom du nœud son indice dans la liste :nodes:.
    :nodes: (modifié) On ajoute le nœud à la carte. """
    info = element.attributes
    nid = info["id"].value
    lat, lon = float(info["lat"].value), float(info["lon"].value)
    node_ids[nid] = len(nodes)
    nodes.append(((lon - bounds["x"]) / bounds["w"],
                  (lat - bounds["y"]) / bounds["h"]))


def parse_way(element, node_ids, ways):
    """ Récupère les informations d'un chemin.

    :element: Nœud XML à analyser.
    :node_ids: (modifié) Dictionnaire qui associe au nom du nœud son indice dans la liste :nodes:.
    :nodes: Liste des nœuds de la carte.

    :ways: (modifié) On ajoute le chemin à la carte. """
    way_nodes = []
    lanes = 1
    for node in element.childNodes:
        if isinstance(node, dom.Text):
            continue
        if node.tagName == "nd":
            ref = node.attributes["ref"].value
            way_nodes.append(
                node_ids[ref])  # on remplace le nom du nœud par un numéro.
        # Nombre de voies
        elif node.tagName == "tag" and node.attributes["k"].value == "lanes":
            lanes = int(node.attributes["v"].value)
    if way_nodes[0] != way_nodes[-1]:  # on n'ajoute pas les boucles.
        ways.append((way_nodes, lanes))


def parse_map(filename):
    """ Récupère les nœuds et les chemins d'une carte OpenStreetMap. """
    tree = dom.parse(filename)

    # 1ère étape : on lit les dimensions de la carte.
    bounds = get_map_bounds(tree)

    node_ids = {}
    nodes, ways = [], []

    # 2ème étape : on parcourt les noeuds et les chemins.
    for child in tree.documentElement.childNodes:
        if isinstance(child, dom.Text):
            continue
        elif child.tagName == "node":
            parse_node(child, bounds, node_ids, nodes)
        elif child.tagName == "way":
            parse_way(child, node_ids, ways)

    return nodes, ways


def affiche_carte(nodes, ways, ax):
    """ Affichage de la carte. 
    :nodes: Liste des nœuds de la carte.
    :ways: Liste des chemins de la carte.

    :ax: (modifié) Dessin Matplotlib. """
    colors = ['blue', 'green', 'red']
    #bleu = 1 voie
    #vert = 2 voies
    #rouge =  3 voies
    for i, way in enumerate(ways):
        node_x, node_y = [], []
        for node in way[0]:
            node_x.append(nodes[node][0])
            node_y.append(nodes[node][1])
        ax.plot(node_x, node_y, color=colors[way[1] - 1])
    plt.show()


def main():
    """ Point d'entrée du programme. """

    filename = "test-rond-point.xml"
    if len(argv) > 1:
        filename = argv[1]

    nodes, ways = parse_map(filename)

    fig, ax = plt.subplots()

    affiche_carte(nodes, ways, ax)

    fig.show()


if __name__ == "__main__":
    main()