#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Route:

    def __init__(route, ville_depart, ville_arrivee, coutA, coutB=None):
        """
        Creation d'une nouvelle route !
        Le coût de la route vaut a*x + b
        """
        route.ville_depart = ville_depart
        route.ville_depart.routes_sortantes.append(route)
        route.ville_arrivee = ville_arrivee
        route.ville_arrivee.routes_entrantes.append(route)
        if type(coutA) is tuple:
            route._coutA, route._coutB = coutA
        else:
            route._coutA = coutA
            route._coutB = coutB
        route.update_texte()
        route._ouverte = True
        route.ouverte = route._ouverte

    @property
    def coutA(route):
        return route._coutA

    @coutA.setter
    def coutA(route, coef):
        route._coutA = coef
        route.update_texte()

    @property
    def coutB(route):
        return route._coutB

    @coutB.setter
    def coutB(route, coef):
        route._coutB = coef
        route.update_texte()

    def update_texte(route):
        texte = ''
        if route.coutA != 0:
            if route.coutA != 1:
                texte += str(route.coutA)
            texte += 'x'
        if route.coutB == 0 and route.coutA != 0:
            pass
        else:
            if route.coutB > 0:
                if route.coutA != 0:
                    texte += '+'
            elif route.coutB < 0:
                texte += '-'
            texte += str(route.coutB)
        route.texte = texte

    @property
    def ouverte(route):
        return route._ouverte

    @ouverte.setter
    def ouverte(route, booleen):
        if route.ouverte != booleen:
            route._ouverte = booleen
            route.ville_depart.update_ville()
            route.ville_arrivee.update_ville()

    def ouvrir(route):
        route.ouverte = True

    def fermer(route):
        route.ouverte = False

    # ------------------------------------------------------------------------------------------------------------------
    # Outils
    def __repr__(route):
        description = '<Route de {} à {}, cout : {}>'.format(route.ville_depart.nom, route.ville_arrivee.nom, route.texte)
        return description