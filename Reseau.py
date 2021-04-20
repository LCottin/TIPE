#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Ville import Ville
from Route import Route
from Wolfe import Wolfe
from copy import deepcopy
import numpy as np
import quadprog as qp
import time
import resolution 
from Genetique import Population
from graphviz import Digraph

class Reseau:

    def __init__(self, nombre_villes=0):
        """ Créé un réseau vierge """
        self.villes = [Ville(i) for i in range(nombre_villes)]
        self.routes = []

    @classmethod
    def aleatoire(cls, nombre_villes=4, nombre_routes=1):
        """
        Créé un réseau aléatoire

        nombre_villes : Nombre de villes
        nombre_routes : Nombre minimal de routes
        """
        # Config. minimale : 2 villes, 1 route
        nombre_villes = max(2, nombre_villes)
        nombre_routes = max(1, nombre_routes)
        # Créer un réseau vierge
        self = cls(nombre_villes)
        # On garde un oeil sur les villes qui ont été reliées
        liaisons = [0] * nombre_villes
        # La source et le puit sont reliés par défaut
        liaisons[0] = liaisons[-1] = 1

        # Créer des routes
        while self.routes.__len__() < nombre_routes or (0 in liaisons):
            # On choisit deux ville (celles les moins reliées sont choisies d'abord)
            proba = [1 / (nombre_liaisons + 1) for nombre_liaisons in liaisons]
            total = sum(proba)
            proba = [proba[i] / total for i in range(nombre_villes)]
            ville_depart, ville_arrivee = np.random.choice(self.villes, size=2, p=proba, replace=False)
            if ville_depart.numero > ville_arrivee.numero:
                ville_depart, ville_arrivee = ville_arrivee, ville_depart
            liaisons[ville_depart.numero] += 1
            liaisons[ville_arrivee.numero] += 1
            # Créer la route
            self.nouvelle_route(ville_depart, ville_arrivee)

        # On s'assure que la source soit une source et le puits un puits
        source = self.villes[0]
        puits = self.villes[-1]
        for ville in self.villes:
            if ville.routes_entrantes.__len__() == 0 and ville != source:
                self.nouvelle_route(source, ville)
            if ville.routes_sortantes.__len__() == 0 and ville != puits:
                self.nouvelle_route(ville, puits)

        return self

    # ------------------------------------------------------------------------------------------------------------------
    # Gestion des routes

    def nouvelle_route(self, ville_depart, ville_arrivee, cout=None):
        """ Créer une nouvelle route """
        if cout is None:
            # Cout aléatoire
            coutA = np.random.randint(1, 10)
            coutB = np.random.randint(0, 8)
        else:
            coutA, coutB = cout
        # Récupérer les villes
        ville_depart = self.villes[int(ville_depart)]
        ville_arrivee = self.villes[int(ville_arrivee)]
        # Créer la route
        route = Route(ville_depart, ville_arrivee, coutA, coutB)
        self.routes.append(route)

    def choisir_route(self, ville_depart=None, ville_arrivee=None, coutA=None, coutB=None):
        """ Renvoie la route sélectionnée """
        # Récupérer les villes
        ville_depart = self.choisir_ville(ville_depart)
        ville_arrivee = self.choisir_ville(ville_arrivee)
        # Trouver la route
        for route in self.routes:
            if route.ville_depart == ville_depart:
                if route.ville_arrivee == ville_arrivee:
                    if type(coutA) is tuple:
                        # Permet de faire reseau.choisir_route(1, 2, (2, 0))
                        if (route.coutA, route.coutB) == coutA:
                            return route
                    elif coutA is None or route.coutA == coutA:
                        if coutB is None or route.coutB == coutB:
                            return route
        return None

    def supprimer_route(self, ville_depart=None, ville_arrivee=None, coutA=None, coutB=None):
        route = self.choisir_route(ville_depart, ville_arrivee, coutA, coutB)
        if route is None:
            return 'Not found'
        else:
            self.routes.remove(route)
            return 'Done'

    def ouvrir_route(self, ville_depart=None, ville_arrivee=None, coutA=None, coutB=None):
        route = self.choisir_route(ville_depart, ville_arrivee, coutA, coutB)
        if route is None:
            return 'Not found'
        else:
            route.ouvrir()
            return 'Done'

    def fermer_route(self, ville_depart=None, ville_arrivee=None, coutA=None, coutB=None):
        route = self.choisir_route(ville_depart, ville_arrivee, coutA, coutB)
        if route is None:
            return 'Not found'
        else:
            route.fermer()
            return 'Done'

    # ------------------------------------------------------------------------------------------------------------------
    # Gestion des villes

    def choisir_ville(self, nom=None):
        """ Choisit une ville """
        if nom is None:
            # Ville aléatoire
            return np.random.choice(self.villes)
        elif type(nom) is str:
            # Ville basée sur le nom
            for ville in self.villes:
                if ville.nom == nom:
                    return ville
            return None
        elif type(nom) is int:
            # Ville basée sur le numéro
            try:
                return self.villes[nom]
            except IndexError:
                return None

    def ouvrir_ville(self, nom=None):
        ville = self.choisir_ville(nom)
        if ville is None:
            return 'Not found'
        else:
            ville.ouvrir()
            return 'Done'

    def fermer_ville(self, nom=None):
        ville = self.choisir_ville(nom)
        if ville is None:
            return 'Not found'
        else:
            ville.fermer()

    # ------------------------------------------------------------------------------------------------------------------
    # Outils

    def __repr__(self):
        desc = '<Reseau de {} villes et {} routes>'.format(self.villes.__len__(), self.routes.__len__())
        return desc

    def sauvergarder(self):
        """
        Renvoie une ligne qui peut être utilisée pour recréer un réseau avec Reseau.charger()
        """
        # Nombre de villes
        save = [self.villes.__len__()]
        # Nom des villes
        noms = [ville.nom for ville in self.villes]
        # Sauvegarder les routes
        for route in self.routes:
            save.append([int(route.ville_depart), int(route.ville_arrivee), route.coutA, route.coutB])
        return save, noms

    @classmethod
    def charger(cls, save, noms=None):
        """ Charge un réseau à partir d'une sauvegarde """
        nombre_villes = save[0]
        self = cls(nombre_villes)
        # Recréer les routes
        routes = save[1:]
        for route in routes:
            self.nouvelle_route(route[0], route[1], route[2:])
        # Nom des villes
        if noms is not None:
            for i in range(nombre_villes):
                self.villes[i].nom = noms[i]
        return self

    def randomiser(self):
        """ Randomise les couts des routes """
        for route in self.routes:
            # Cout aléatoire
            coutA = np.random.randint(1, 10)
            coutB = np.random.randint(0, 8)
            route.coutA = coutA
            route.coutB = coutB
        return self


    def copy(self):
        sauvegarde = self.sauvergarder()
        clone = Reseau.charger(*sauvegarde)
        for i in range(len(clone.routes)):
            if not self.routes[i].ouverte:
                clone.routes[i].fermer()
        return clone

    # ------------------------------------------------------------------------------------------------------------------
    # Calculs
    def cout(self, flot):
        """ Cout total du flot """
        flot = np.array(flot)
        flot = flot.reshape(len(flot), 1)
        self.update_matrices()
        if len(flot) != len(self.A):
            flot_sans_fermes = []
            for i, route in enumerate(self.routes):
                if route.ouverte:
                    flot_sans_fermes.append(flot[i])
            flot = np.array(flot_sans_fermes)
        cout = flot.T.dot(self.A).dot(flot) + self.B.T.dot(flot)
        return float(cout)

    def update_matrices(self):
        """
        Met à jour les matrices
        On a:
        f(x) = x.A.x + B.x
        où C.x = D
        """
        # Met à jour le nombre de routes et villes ouvertes
        self.nombre_routes_ouvertes = 0
        for route in self.routes:
            if route.ouverte:
                self.nombre_routes_ouvertes += 1
        self.nombre_villes_ouvertes = 0
        for ville in self.villes:
            if ville.ouverte:
                self.nombre_villes_ouvertes += 1
        self.nombre_inconnues = self.nombre_routes_ouvertes
        self.nombre_contraintes = self.nombre_villes_ouvertes

        # Créer les matrices
        A = np.zeros((self.nombre_inconnues, self.nombre_inconnues))
        B = np.zeros((self.nombre_inconnues,))
        C = np.zeros((self.nombre_contraintes, self.nombre_inconnues))
        i = -1
        for route in self.routes:
            if route.ouverte:
                i += 1
                A[i, i] = route.coutA
                B[i] = route.coutB
                j = -1
                for ville in self.villes:
                    if ville.ouverte:
                        j += 1
                        if ville is route.ville_depart:
                            C[j, i] = -1
                        if ville is route.ville_arrivee:
                            C[j, i] = 1
        D = np.zeros((self.nombre_contraintes,))
        D[0] = -1
        D[-1] = 1
        self.A, self.B, self.C, self.D = A, B, C, D
        return A, B, C, D

    def verifier_contraintes(self, flot):
        return (self.C.dot(flot) - self.D < 10**(-10)).all()

    def normaliser(self, flot):
        """ Change le flot pour qu'il respecte la loi des noeuds (utile en génétique) """
        # Pour chaque ville
        for ville in self.villes[:-1]:
            # On calcule le flot entrant et le flot sortant
            flot_entrant = int(int(ville) == 0)
            flot_sortant = 0
            indices_routes_sortantes = []
            j = -1
            for route in self.routes:
                if route.ouverte:
                    j += 1
                    # La route doit être ouverte
                    if int(ville) != 0 and route.ville_arrivee == ville:
                        flot_entrant += flot[j]
                    if route.ville_depart == ville:
                        flot_sortant += flot[j]
                        indices_routes_sortantes.append(j)
            # On normalise les flots sortants
            if flot_sortant != 0:
                # On remplace les flots sortants par flot / flot_sortant * flot_entrant (une proportion du flot entrant)
                for i in indices_routes_sortantes:
                    flot[i] = flot[i] / flot_sortant * flot_entrant
            else:
                # Si tous les flots valent 0, on fait une répartition uniforme
                nombre_sortant = len(indices_routes_sortantes)
                for i in indices_routes_sortantes:
                    flot[i] = flot_entrant / nombre_sortant
        return flot

    # ------------------------------------------------------------------------------------------------------------------
    # Etude temporelle
    def comparaison_temps(self):
        # Algo génétique
        t = time.clock()
        Reseau.resoudre_genetique(self, dessiner=False)
        t1 = time.clock() - t

        t = time.clock()
        Reseau.resoudre_genetique(self, egoiste=True, dessiner=False)
        t2 = time.clock() - t

        # Quadprog
        t = time.clock()
        Reseau.resoudre_qp(self, dessiner=False)
        t3 = time.clock() - t

        t = time.clock()
        Reseau.resoudre_qp(self, egoiste=True, dessiner=False)
        t4 = time.clock() - t

        # Wolfe
        t = time.clock()
        Reseau.resoudre_wolfe(self, dessiner=False)
        t5 = time.clock() - t
    
        t = time.clock()
        Reseau.resoudre_wolfe(self, egoiste=True, dessiner=False)
        t6 = time.clock() - t

        # Descente
        t = time.clock()
        #Reseau.resoudre_descente(self, dessiner=False)
        t7 = time.clock() - t

        t = time.clock()
        #Reseau.resoudre_descente(self, egoiste=True, dessiner=False)
        t8 = time.clock() - t

        print(f"\nTemps d'execution de l'algorithme genetique : {round(t1, 4)}s")
        print(f"\nTemps d'execution de l'algorithme genetique egoiste : {round(t2, 4)}s")
        print(f"\nTemps d'execution de l'algorithme de Quadprog : {round(t3, 4)}s")
        print(f"\nTemps d'execution de l'algorithme de Quadprog egoiste : {round(t4, 4)}s")
        print(f"\nTemps d'execution de l'algorithme de Wolfe : {round(t5, 4)}s")
        print(f"\nTemps d'execution de l'algorithme de Wolfe egoiste : {round(t6, 4)}s")
        print(f"\nTemps d'execution de l'algorithme de descente de gradient : {round(t7, 4)}s")
        print(f"\nTemps d'execution de l'algorithme de descente de gradient egoiste : {round(t8, 4)}s")
        print("\n.")

    # ------------------------------------------------------------------------------------------------------------------
    # Egoiste
    def egoiste(self):
        reseau_egoiste = deepcopy(self)
        for route in reseau_egoiste.routes:
            route.coutA = route.coutA / 2
        return reseau_egoiste

    # ------------------------------------------------------------------------------------------------------------------
    # Trouver des réseaux
    @staticmethod
    def trouver_reseaux(flot_epsilon=0.1, cout_epsilon=0.1, nbville=8):
        compteur = 0
        liste_ouverte = [True]
        while all(liste_ouverte):

            r = Reseau.aleatoire(nbville, 18)
            try:
                r2 = r.optimiser(flot_epsilon=flot_epsilon, cout_epsilon=cout_epsilon)
            except:
                continue
            compteur += 1
            print(compteur)
            liste_ouverte = [route.ouverte for route in r2.routes]
        return r, r2, compteur

    def optimiser(self, flot_epsilon=0.10, cout_epsilon=0.1, m=None):
        # Si il n'y a pas de meilleur, self est le meilleur
        if m is None:
            m = self
        routes_fermees = [not route.ouverte for route in self.routes]
        if not all(routes_fermees):
            # On n'étudie le réseau que si il reste des routes ouvertes
            flot_soc = self.resoudre_qp(dessiner=False)
            flot_ego = self.resoudre_qp(egoiste=True, dessiner=False)
            flot_ego_m = m.resoudre_qp(egoiste=True, dessiner=False)
            if self.cout(flot_ego) + cout_epsilon < m.cout(flot_ego_m):
                # On compare ce réseau au meilleur
                m = self
            j = -1     # Compteur pour les routes ouvertes
            for i in range(len(self.routes)):
                # On parcourt les routes...
                if self.routes[i].ouverte:
                    j += 1
                    if flot_soc[j] + flot_epsilon < flot_ego[j]:
                        # ... pour voir si certaines peuvent être améliorées
                        # print(j)
                        clone = self.copy()
                        clone.routes[i].fermer()
                        # On effectue une récursion
                        m2 = clone.optimiser(flot_epsilon=flot_epsilon, cout_epsilon=cout_epsilon, m=m)
                        # On pense à mettre à jour le meilleur
                        flot_ego_m = m.resoudre_qp(egoiste=True, dessiner=False)
                        flot_ego_m2 = m2.resoudre_qp(egoiste=True, dessiner=False)
                        if m2.cout(flot_ego_m2) + cout_epsilon < m.cout(flot_ego_m):
                            m = m2
        return m


    @staticmethod
    def stats(nb_reseaux=10000):
        """ Fait des stats sur les réseaux optimisables """
        nb_optimisable = 0
        nb_total = 0
        while nb_total < nb_reseaux:
            r1, r2, compteur = Reseau.trouver_reseaux(nbville=8)
            nb_optimisable += 1
            nb_total += compteur
            print(nb_total)
        return nb_optimisable,  nb_total, nb_optimisable / nb_total

    # ------------------------------------------------------------------------------------------------------------------
    # Algo génétique
    def resoudre_genetique(self, egoiste=False, afficher_graphique=True, dessiner=True):
        # Choix du réseau
        if egoiste:
            reseau = self.egoiste()
        else:
            reseau = self
        # Résoudre
        p = Population(reseau, demographie=5)
        p.evolue(nombre_generations=5, afficher_graphique=afficher_graphique)
        solution = p.meilleur_flot
        if dessiner:
            self.dessiner_flot(solution)
        return solution

    # ------------------------------------------------------------------------------------------------------------------
    # Wolfe
    def resoudre_wolfe(self, egoiste=False, dessiner=True, stocker=False):
        """
        L'objectif est de minimiser
        f(x) = xT.A.x + BT.x
        où C.x = D

        i.e. maximiser
        -f(x) = xT.-A.x + -BT.x
        où C.x = D
        """
        # Choix du réseau
        if egoiste:
            reseau = self.egoiste()
        else:
            reseau = self
        # Résoudre
        A, B, C, D = reseau.update_matrices()
        A = -A
        B = -B
        w = Wolfe(A, B, C, D, stocker=stocker)
        w.resoudre()
        # Extraction des soltions
        solution = w.extraire()
        # Dessiner
        if dessiner:
            self.dessiner_flot(solution)
        return solution

    # ------------------------------------------------------------------------------------------------------------------
    # Quadprog
    def resoudre_qp(self, egoiste=False, dessiner=True):
        """ Ne fonctionne que si aucun coef A = 0 """
        # Choix du réseau
        if egoiste:
            reseau = self.egoiste()
        else:
            reseau = self
        A, B, C, D = reseau.update_matrices()
        A = 2*A
        B = -B
        # Contraintes d'inégalité
        # -x <= 0 i.e. x >= 0
        G = -np.identity(reseau.nombre_inconnues)
        h = np.zeros((reseau.nombre_inconnues,))
        # Regrouper les contraintes
        qp_C = -np.vstack([C[:-1, :], G]).T
        qp_D = -np.hstack([D[:-1], h])
        nombre_contraintes_egalite = reseau.nombre_contraintes
        
        solution = qp.solve_qp(A, B, qp_C, qp_D, nombre_contraintes_egalite, factorized=False)[0]
        
        if dessiner:
            self.dessiner_flot(solution)
        return solution

    # ------------------------------------------------------------------------------------------------------------------
    # Résolution dérivées partielles
    def resoudre_descente(self, egoiste=False, dessiner=True):
        # Choix du réseau
        if egoiste:
            reseau = self.egoiste()
        else:
            reseau = self
        solution = resolution.resoudre(reseau, histoire=dessiner)
        if dessiner:
            self.dessiner_flot(solution)
        return solution

    # ------------------------------------------------------------------------------------------------------------------
    # Dessin
    def dessiner(self, mode='cout', flot=None):
        """ Dessine le réseau """
        input('Appuie sur entree : ')
        graph = Digraph('Network', format='pdf')

        # Gestion des villes
        for ville in self.villes:
            if ville.ouverte:
                color = ''
            else:
                color = '0 1 0.8'
            graph.node(str(ville.nom), color=color, tooltip=str(ville.numero))

        compteur_routes_fermees = 0
        for i, route in enumerate(self.routes):
            # Ce qu'on affiche sur les routes
            edgetooltip = ''
            label = ''
            color = ''
            if flot is not None:
                if not route.ouverte:
                    color = '0 1 0.8'
                    compteur_routes_fermees += 1
                    valeur = 0
                else:
                    valeur = flot[i - compteur_routes_fermees]
                    valeur = round(valeur, 3)
                    if valeur == -0.0:
                        valeur = 0
                    # Pour avoir in nombre entre 0.33 et 0.1 : 0.33 - 0.23 * <nb entre 0 et 1>
                    color = '{} 1 0.8'.format(0.33 - 0.23 * valeur)
                if mode == 'cout':
                    edgetooltip = str(valeur)
                elif mode == 'flot':
                    label = str(valeur)
                    edgetooltip = route.texte
            else:
                if not route.ouverte:
                    color = '0 1 0.8'
            if mode == 'cout':
                #route.update_texte(i)
                label = route.texte
            # Trace la route
            graph.edge(str(route.ville_depart.nom), str(route.ville_arrivee.nom),
                       label=label, edgetooltip=edgetooltip, color=color)

        # Gestion du cout total
        if flot is not None:
            #graph.node("Cout : " + str(11.453))
            graph.node("Cout : " + str(round(self.cout(flot), 3)))

        # Afficher le graph
        graph.render(view=True)

    # Pour dessiner un flot
    def dessiner_flot(self, flot):
        self.dessiner(mode='flot', flot=flot)


# ----------------------------------------------------------------------------------------------------------------------
# Exemples
exemples = {'simple': Reseau.charger([3, [0, 1, 1, 2], [0, 2, 2, 1], [1, 2, 1, 0]]),
            'braess0': Reseau.charger([4, [0, 1, 1, 0], [0, 2, 0, 1], [1, 3, 0, 1], [2, 3, 1, 0]]),
            'braess1': Reseau.charger([4, [0, 1, 1, 0], [0, 2, 0, 1], [1, 3, 0, 1], [2, 3, 1, 0], [1, 2, 0, 0]]),
            'Nanterre': Reseau.charger([8,
                                        [0, 1, 4, 4],
                                        [0, 2, 3, 9],
                                        [1, 3, 6, 6],
                                        [1, 5, 6, 6],
                                        [2, 3, 4, 3],
                                        [3, 6, 6, 6],
                                        [2, 6, 3, 8],
                                        [4, 5, 5, 7],
                                        [4, 7, 5, 7],
                                        [6, 7, 6, 3],
                                        [5, 7, 6, 6],
                                        [1, 4, 5, 7],
                                        [3, 4, 5, 7]],
                                       ['Nanterre',
                                        'Porte Maillot',
                                        'Stade de France',
                                        'Porte Chapelle',
                                        'Notre-Dame-de-Paris',
                                        'Porte Orléans',
                                        'Porte Bagnolet',
                                        'Porte vincennes']),
            'Triomphe': Reseau.charger([7,
                                        [0, 1, 3, 3],
                                        [0, 2, 2, 3],
                                        [1, 2, 1, 1],
                                        [1, 3, 5, 5],
                                        [2, 4, 4, 6],
                                        [3, 4, 2, 3],
                                        [3, 5, 5, 4],
                                        [4, 6, 4, 5],
                                        [1, 5, 4, 8],
                                        [5, 6, 4, 3]],
                                       ['Arc de Triomphe',
                                        'Place Concorde',
                                        'Place Madeleine',
                                        'Sainte Chapelle',
                                        'République',
                                        'Pont Sully',
                                        'Bastille']),
                                        
             'Savigny': Reseau.charger([10,
                                       [0, 1, 2, 1],
                                       [0, 2, 1, 2],
                                       [1, 2, 2, 1],
                                       [0, 4, 3, 4],
                                       [1, 3, 4, 2],
                                       [2, 3, 2, 3],
                                       [3, 4, 1, 1],
                                       [3, 7, 2, 2],
                                       [3, 8, 4, 4],
                                       [4, 5, 1, 2],
                                       [4, 6, 3, 2],
                                       [5, 6, 1, 1],
                                       [5, 8, 2, 1],
                                       [6, 9, 2, 1],
                                       [7, 8, 3, 1],
                                       [8, 9, 3, 1]],
                                      ['Epinay-sur-Orge',
                                       'Lycee',
                                       'Gare',
                                       'Rond-point',
                                       'Parking lycee',
                                       'Résidence',
                                       'Grande Rue',
                                       'Début N7',
                                       'So Square',
                                       "Restaurant"]),
                                       
            'Savigny_reduit' : Reseau.charger([4,
                                               [0, 1, 2, 1],
                                               [0, 2, 1, 2],
                                               [1, 2, 2, 0],
                                               [2, 3, 2, 3],
                                               [1, 3, 4, 2]],
                                               ['Epinay-sur-Orge',
                                                'Lycee',
                                                'Gare',
                                                'Rond-point']),
            'Savigny2': Reseau.charger([10,
                                       [0, 1, 2, 1],
                                       [0, 2, 1, 0],
                                       [1, 2, 2, 1],
                                       [1, 4, 1, 3],
                                       [1, 3, 4, 2],
                                       [2, 3, 2, 3],
                                       [3, 4, 2, 1],
                                       [3, 7, 3, 3],
                                       [3, 8, 2, 3],
                                       [4, 5, 1, 2],
                                       [4, 6, 3, 2],
                                       [5, 6, 1, 1],
                                       [5, 8, 2, 1],
                                       [6, 9, 2, 1],
                                       [7, 8, 3, 1],
                                       [8, 9, 3, 1]],
                                      ['Epinay-sur-Orge',
                                       'Lycee',
                                       'Gare',
                                       'Rond-point',
                                       'Parking Corot',
                                       'Résidence',
                                       'Grande Rue',
                                       'Début N7',
                                       'So Square',
                                       "Restaurant"]),
            'opti1': Reseau.charger(([9,
                                      [3, 5, 1, 0],
                                      [3, 4, 4, 2],
                                      [0, 1, 1, 1],
                                      [1, 5, 1, 2],
                                      [0, 1, 4, 0],
                                      [3, 8, 1, 2],
                                      [2, 7, 3, 2],
                                      [7, 8, 3, 2],
                                      [2, 3, 3, 0],
                                      [6, 7, 1, 0],
                                      [0, 2, 4, 0],
                                      [4, 8, 3, 0],
                                      [5, 8, 3, 1],
                                      [0, 6, 3, 1]]))}

if __name__ == '__main__':
    # Exemple d'optimisation : opti1 avec r.optimiser
#    r = exemples['opti1']
#    r2 = r.optimiser()
#    r.resoudre_qp()
#    time.sleep(2)
#    r.resoudre_qp(egoiste=True)
#    time.sleep(2)
#    r2.resoudre_qp(egoiste=True)
#    time.sleep(2)
#
#    # opti1 à la main
#    r_clone = r.copy()
#    r_clone.fermer_route(0, 1, 1, 1)
#    r_clone.resoudre_qp(egoiste=True)
#
#    # # Trouver des réseaux à optimiser
#    # r, r2 = Reseau.trouver_reseaux()
#    # r.resoudre_qp()
#    # time.sleep(2)
#    # r.resoudre_qp(egoiste=True)
#    # time.sleep(2)
    # r2.resoudre_qp(egoiste=True)


    pass