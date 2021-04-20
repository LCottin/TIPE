#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Ville:

    def __init__(ville, numero, nom=None):
        """
        Créé une nouvelle ville !
        """
        ville.numero = numero
        ville.routes_entrantes = []
        ville.routes_sortantes = []
        ville._ouverte = True
        if nom is None:
            ville.nom = ville.numero
        else:
            ville.nom = nom

    @property
    def ouverte(ville):
        return ville._ouverte

    @ouverte.setter
    def ouverte(ville, booleen):
        if ville.ouverte != booleen:
            ville._ouverte = booleen
            ville.update_routes()

    # ------------------------------------------------------------------------------------------------------------------
    # Gestion de la ville

    def update_routes(ville):
        # Si on ouvre / ferme la ville, on ouvre / ferme toutes les routes associées
        for route in ville.routes_entrantes:
            route.ouverte = ville.ouverte
        for route in ville.routes_sortantes:
            route.ouverte = ville.ouverte

    def update_ville(ville):
        # Si toutes les routes entrantes (ou sortantes) sont femées, on ferme la ville
        # Sinon, on l'ouvre
        etat_routes_entrantes = [route.ouverte for route in ville.routes_entrantes]
        etat_routes_sortantes = [route.ouverte for route in ville.routes_sortantes]
        if (len(ville.routes_entrantes) != 0 and not (True in etat_routes_entrantes)) \
                or (len(ville.routes_sortantes) != 0 and not (True in etat_routes_sortantes)):
            ville.ouverte = False
        else:
            ville.ouverte = True

    def ouvrir(ville):
        ville.ouverte = True

    def fermer(ville):
        ville.ouverte = False

    # ------------------------------------------------------------------------------------------------------------------
    # Outils
    def __repr__(ville):
        """ Défini comment la ville est décrite dans la console """
        if ville.ouverte:
            etat = 'ouverte'
        else:
            etat = 'fermee'
        description = '<Ville {}, {}>'.format(ville.nom, etat)
        return description

    def __int__(ville):
        return ville.numero

    def __eq__(ville1, ville2):
        """ ville1 == ville2 """
        return ville1.numero == ville2.numero

    def __gt__(ville1, ville2):
        """ ville1 > ville2"""
        return