#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cerwer tool est un programme qui permet de produire un rapport
contenant le Word Error Rate (WER), le Character Error Rate (CER) et
le nombre d'insertions, substitutions et suppressions afin de
comparer une transcription Ground Truth (reference_file) et une transcription
issue d'un modèle de transcription (hypothesis_file).

[OPTIONS]

- édition d'un graphique d'étapes

Author : Lucas Terriel
Date : 30/04/2020

"""
import sys
from datetime import datetime
import time

import re

import numpy as np
import matplotlib.pyplot as plt
from kraken.lib.dataset import _fast_levenshtein

# Fonction de tokenisation des mots et des lettres
# Possibilité d'ajouter dans les regex les espaces


def tokenize(phrase):
    """
    tokenise() découpe la phrase en mots et en caracatères afin
    d'obtenir le maximum de précision dans les calculs
    :param phrase: reference_sentence, hypothesis_sentence
    :return: str
    """
    sentence_tokenize_words = re.findall(
        r"[\wÀ-ÿ]+|[\]\[!\"#$%&'()*+,.\/:;<=>?@\^_`{|}~-]", phrase)
    sentence_tokenize_characters = re.findall(
        r"[\wÀ-ÿ]|[\]\[!\"#$%&'()*+,.\/:;<=>?@\^_`{|}~-]", phrase)
    return sentence_tokenize_words, sentence_tokenize_characters

# Fonction de calcul de la distance de Levenshtein (ou distance d'édition)


def lev_distance(reference, hypothesis):
    """
    Cette fonction calcule la distance d'édition
    c'est-à-dire quelle mesure la différence entre deux chaines de caractères,
    en l'occurence une phrase de référence et une phrase hypothétique.
    L'algorithme utilisé correspond à celui utilisé en programmation
    dynamique qui utilise Numpy.
    Cf. WER-in-python(Github @zszyellow) / https://en.wikibooks.org/wiki/Algorithm_Implementation
    /Strings/Levenshtein_distance

    :param r: phrase de référence tokénisée au mot ou à la lettre
    :type r: list
    :param h: phrase cible et hypothétique construite
    par le modèle tokénisée au mot ou à la lettre
    :type h: list
    :return: d soit la distance d'édition sous la
    forme d'une liste imbriquée (matrix)
    (matrice de dimension [r + 1 x h + 1])
    """
    distance = np.zeros((len(reference) + 1) * (len(hypothesis) + 1), dtype=np.uint8).reshape\
        ((len(reference) + 1, len(hypothesis) + 1))
    for i in range(len(reference) + 1):
        distance[i][0] = i
    for j in range(len(hypothesis) + 1):
        distance[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                substitute = distance[i - 1][j - 1] + 1
                insert = distance[i][j - 1] + 1
                delete = distance[i - 1][j] + 1
                distance[i][j] = min(substitute, insert, delete)
    return distance

# getStepList() calcule et stocke les étapes
# d'insertions, de suppressions et de substitutions de mots
# ou de lettres qui ont pu subvenir durant le calcul de la
# distance d'édition


def get_step_list(reference, hypothesis, distance):
    """Cette fonction exprime les étapes
    pour passer de la phrase de référence à la phrase cible
    :param r: phrase de référence tokénisée au mot ou à la lettre
    :type r: list
    :param h: phrase cible et hypothétique construite par
    le modèle tokénisée au mot ou à la lettre
    :type h: list
    :param d: distance d'édition
    :type d: matrix
    :return: impression des résultats sous la forme "Ins : , Subs :  , Dels : "
    Cf. WER-in-python(Github @zszyellow)
    """

    r_x = len(reference)
    r_y = len(hypothesis)
    liste = []
    liste2 = []
    while True:
        if r_x == 0 and r_y == 0:
            break
        if r_x >= 1 and r_y >= 1 and distance[r_x][r_y] == distance[r_x - 1][r_y - 1] and \
                reference[r_x - 1] == hypothesis[r_y - 1]:
            liste.append("e")
            r_x = r_x - 1
            r_y = r_y - 1
        elif r_y >= 1 and distance[r_x][r_y] == distance[r_x][r_y - 1] + 1:
            liste.append("i")
            r_x = r_x
            r_y = r_y - 1
        elif r_x >= 1 and r_y >= 1 and distance[r_x][r_y] == distance[r_x - 1][r_y - 1] + 1:
            liste.append("s")
            r_x = r_x - 1
            r_y = r_y - 1
        else:
            liste.append("d")
            r_x = r_x - 1
            r_y = r_y
    exact_match = liste.count("e")
    liste2.append(exact_match)
    ins = liste.count("i")
    liste2.append(ins)
    subs = liste.count("s")
    liste2.append(subs)
    dels = liste.count("d")
    liste2.append(dels)
    return "Exact : %d, Ins : %d, Subs : %d, Dels : %d" \
           % (liste.count("e"), liste.count("i"), liste.count("s"), liste.count("d")), liste2


# Fonctions de calcul WER et CER


def resultwer(reference_sentence, distance):
    """
    calcul du WER = Total Word Errors (Edit distance) / Total Words
    of reference sentence * 100 (en %)
    :param reference_sentence: phrase de référence tokénisée
    :type str
    :param hypothesis_sentence: phrase de hypothétique tokénisée
    :type str
    :param distance: editditance
    :type matrix
    :return: resultat WER avec et sans %
    """
    result_wer = float(distance/len(reference_sentence)) * 100
    result_wer_withoutp = float((distance)/len(reference_sentence))
    result_wer = str("%.2f" % result_wer) + "%"
    result_wer_withoutp = str("%.3f" % result_wer_withoutp)
    return result_wer, result_wer_withoutp



def resultcer(reference_sentence, distance):
    """
    calcul du CER = Total Char Errors (Edit distance) / Total Char of
     reference sentence * 100 (en %)
    :param reference_sentence: phrase de référence tokénisée
    :type str
    :param hypothesis_sentence: phrase de hypothétique tokénisée
    :type str
    :param distance: editditance
    :type int
    :return: resultat CER avec et sans %
    """
    result_cer = float(distance/len(reference_sentence))*100
    result_cer_withoutp = float(distance/len(reference_sentence))
    result_cer = str("%.2f" % result_cer) + "%"
    result_cer_withoutp = str("%.3f" % result_cer_withoutp)
    return result_cer, result_cer_withoutp

# fonction principale d'execution du programme

def cerwer_tool(reference_file, hypothesis_file):
    """
    Fonction d'execution du programme

    :param reference_file: phrase de référence
    :type str
    :param hypothesis_file: phrase issu du modèle
    :type str
    :return: affichage et graph
    """

    name = input("Entrer votre prénom et votre nom : ")

    # On initialise le chrono

    start_time = time.time()

    reference = reference_file
    hypothesis = hypothesis_file

    # STEP 1 : Tokenisation

    r_w, r_l = tokenize(reference)
    h_w, h_l = tokenize(hypothesis)

    # STEP 2 : calcul de la distance de Levenshtein (ou distance d'édition) :

    # On utilise ici l'algo numpy pour le calcul d'étapes

    lev_words = lev_distance(r_w, h_w)
    lev_characters = lev_distance(r_l, h_l)

    # On utilise ici fast_levenstein() du module Kraken pour le calcul du CER
    # du WER

    w_lev = _fast_levenshtein(r_w, h_w)
    l_lev = _fast_levenshtein(r_l, h_l)

    # STEP 3 : Calcul d'étapes : insertions (Ins), subtitutions (Subs)
    # et suppressions (Dels)

    step_words, y_pred_w = get_step_list(r_w, h_w, lev_words)
    step_characters, y_pred_c = get_step_list(r_l, h_l, lev_characters)

    # STEP 4a : calcul du WER (Word Error Rate)

    wer_p, wer_sp = resultwer(r_w, w_lev)

    # STEP 4b : calcul du CER (Character Error Rate)

    cer_p, cer_sp = resultcer(r_l, l_lev)

    # STEP 5 : Réalisation du graphique (OPTIONNEL)

    option1 = input("Désirez-vous éditer un graphique ? [O/n] : ")

    if option1 == "O":

        # Constantes pour les graphiques : Calcul de la
        # distance d'édition et de la liste d'étapes idéal

        lev_words_true = lev_distance(r_w, r_w)
        step_words_true, y_true = get_step_list(r_w, r_w, lev_words_true)

        # Graphique d'étapes

        # on initialise les barres

        bars1 = y_true
        bars2 = y_pred_w

        # On définit la largeur des barres du graphique

        bar_width = 0.1

        # On initialise la position des barres sur l'axe des ordonnées

        r_1 = np.arange(len(bars1))
        r_2 = [x + bar_width for x in r_1]

        fig, axis_x = plt.subplots()
        rects1 = axis_x.bar(r_1, bars1, bar_width, label='Reference_sentence')
        rects2 = axis_x.bar(r_2, bars2, bar_width, label='Predicted_sentence')

        # On ajoute les titres les labels les légendes éventuels et on défini leurs places

        axis_x.set_ylabel('nombre_de_mots')
        axis_x.set_title('Evaluation du modèle de transcription')
        plt.xticks(
            [r + bar_width for r in range(len(bars1))],
            ['mots reconnus', 'insertions', 'substitutions', 'suppresions']
            )
        plt.legend(bbox_to_anchor=(1.6, 1), loc='upper right', prop={'size': 10})

        # Fonction qui permet d'attacher un nombre à la barre

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                axis_x.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

    # STEP 6 : Output => affichage des résultats

    rapport = print("*"*40, \
                " \n", \
                "---- RAPPORT ----", \
                " \n", "* Effectué par : ", name,\
                "\n",\
                "* Date - Heure : ", datetime.now(), "\n",\
                "* Temps d'execution : %f secondes ---" % (time.time() - start_time), "\n",\
                " \n",\
                "Phrase de référence           : \n", reference, "\n",\
                "Phrase du modèle              : \n", hypothesis, "\n",\
                " \n"\
                "Phrase de référence tokenisée : \n", r_w,\
                "Longeur                       : ", len(r_w), "token(s)", "\n",\
                " \n",\
                "Phrase du modèle tokénisée    : \n", h_w, "\n",\
                "Longeur                       : ", len(h_w), "token(s)", "\n",\
                " \n",\
                "Mots :", step_words, "\n",\
                " \n",\
                "Lettres :", step_characters, "\n",\
                " \n",\
                "Résultats du WER (en %)       : ", wer_p, "\n",\
                "Résultats du WER              : ", wer_sp, "\n",\
                " \n",\
                "Résultats du CER (en %)       : ", cer_p, "\n",\
                "Résultats du CER              : ", cer_sp, "\n",\
                " \n",\
                "*" * 40)


    return rapport, plt.show()

if __name__ == '__main__':
    FILENAME1 = sys.argv[1]
    FILENAME2 = sys.argv[2]
    with open(FILENAME1, 'r', encoding="utf8") as ref:
        REFERENCE_FILE = ref.read()
    with open(FILENAME2, 'r', encoding="utf8") as hyp:
        HYPOTHESIS_FILE = hyp.read()
    cerwer_tool(REFERENCE_FILE, HYPOTHESIS_FILE)
