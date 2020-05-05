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
- édition d'un rapport .txt

Author : Lucas Terriel
Date : 05/05/2020

"""

from datetime import datetime
import time
import argparse


import re
import pyfiglet

import numpy as np
import matplotlib.pyplot as plt

from kraken.lib.dataset import _fast_levenshtein

# FONCTIONS UTILES :

# Fonction de tokenisation des mots et des lettres
# Possibilité d'ajouter dans les regex les espaces


def tokenize(phrase):
    """
    Segmentation de la phrase en mots et en caractères afin
    d'obtenir le maximum de précision dans les calculs

    :param phrase: chaine de caractères (phrase de référence ou phrase d'hypothèse)
    :type phrase : str
    :return: retourne une liste de mots et de caractères qui correpond à la phrase
    segmentée
    :type return : list
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

    Détails:

    L'algorithme utilisé correspond à celui utilisé en programmation
    dynamique qui utilise Numpy.
    Pour le code :
    Cf. WER-in-python(Github @zszyellow)
    Pour l'algorithme :
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance

    :param reference: phrase de référence tokénisée au mot ou à la lettre
    :type reference: list
    :param hypothesis: phrase cible et hypothétique construite
    par le modèle tokénisée au mot ou à la lettre
    :type hypothesis: list
    :return: d soit la distance d'édition
    :type return : matrix (matrice de dimension [r + 1 x h + 1])
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

# liste_etapes() calcule et stocke les étapes
# d'insertions, de suppressions et de substitutions de mots
# ou de lettres qui ont pu subvenir durant le calcul de la
# distance d'édition


def liste_etapes(reference, hypothesis, distance):
    """Cette fonction exprime les étapes
    pour passer de la phrase de référence à la phrase cible

    Pour le code :
    Cf. WER-in-python(Github @zszyellow)

    :param reference: phrase de référence tokénisée au mot ou à la lettre
    :type reference: list
    :param hypothesis: phrase cible tokénisée au mot ou à la lettre
    :type hypothesis: list
    :param distance: distance d'édition
    :type distance: matrix
    :return: impression des résultats sous la forme "Exact: Ins : , Subs :  , Dels : " et une
    liste contenant le nombre d'exactitudes, d'insertions, de substitutions et de suppressions
    :type returns : str and list
    """

    longueur_reference = len(reference)

    longueur_hypothese = len(hypothesis)

    liste_etapes = []

    liste_etapes_stats = []

    while True:
        if longueur_reference == 0 and longueur_hypothese == 0:
            break

        if longueur_reference >= 1 and longueur_hypothese >= 1 and \
                distance[longueur_reference][longueur_hypothese] == \
                distance[longueur_reference - 1][longueur_hypothese - 1] and \
                reference[longueur_reference - 1] == hypothesis[longueur_hypothese - 1]:
            liste_etapes.append("e")
            longueur_reference = longueur_reference - 1
            longueur_hypothese = longueur_hypothese - 1

        elif longueur_hypothese >= 1 and distance[longueur_reference][longueur_hypothese] == \
                distance[longueur_reference][longueur_hypothese - 1] + 1:
            liste_etapes.append("i")
            longueur_reference = longueur_reference
            longueur_hypothese = longueur_hypothese - 1

        elif longueur_reference >= 1 and longueur_hypothese >= 1 and \
                distance[longueur_reference][longueur_hypothese] ==\
                distance[longueur_reference - 1][longueur_hypothese - 1] + 1:
            liste_etapes.append("s")
            longueur_reference = longueur_reference - 1
            longueur_hypothese = longueur_hypothese - 1

        else:
            liste_etapes.append("d")
            longueur_reference = longueur_reference - 1
            longueur_hypothese = longueur_hypothese

    exact_match = liste_etapes.count("e")
    liste_etapes_stats.append(exact_match)

    ins = liste_etapes.count("i")
    liste_etapes_stats.append(ins)

    subs = liste_etapes.count("s")
    liste_etapes_stats.append(subs)

    dels = liste_etapes.count("d")
    liste_etapes_stats.append(dels)

    return f"Exact : {liste_etapes.count('e')}, " \
               f"Ins : {liste_etapes.count('i')}, Subs : {liste_etapes.count('s')}" \
               f", Dels : {liste_etapes.count('d')}", liste_etapes_stats


# Fonctions de calcul WER et CER


def resultwer(reference_sentence, distance):
    """calcul du WER = Total Word Errors (Edit distance) / Total Words
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
    result_wer_withoutpercentage = float((distance)/len(reference_sentence))
    result_wer = str("%.2f" % result_wer) + "%"
    result_wer_withoutp = str("%.3f" % result_wer_withoutpercentage)
    return result_wer, result_wer_withoutp


def resultcer(reference_sentence, distance):
    """calcul du CER = Total Char Errors (Edit distance) / Total Char of
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
    result_cer_withoutpercentage = float(distance/len(reference_sentence))
    result_cer = str("%.2f" % result_cer) + "%"
    result_cer_withoutp = str("%.3f" % result_cer_withoutpercentage)
    return result_cer, result_cer_withoutp

# Fonction pour générer un graphique

def edit_graphique_barres(reference_tokenize_words, y_pred_words):
    """Fonction qui permet d'éditer un diagramme en barres pour
    évaluer le nombre d'insertions, de substitutions ou de
    suppressions entre une phrase de référence et une phrase
    hypothétique par rapport aux nombres de mots totals

    :param reference_tokenize_words: phrase de référence tokénisée au mot
    :type:list
    :param y_pred_words: liste contenant le nombre total de mots
    exacts, supprimés, substitués et ajoutés de la phrase cible par rapport
    à la phrase de référence
    :type:list
    :return: graphique -> matplotlib.object
    """

    # Constantes pour les graphiques : Calcul de la
    # distance d'édition et de la liste d'étapes idéal

    lev_words_true = lev_distance(reference_tokenize_words,
                                  reference_tokenize_words)

    step_words_true, y_true = liste_etapes(reference_tokenize_words,
                                           reference_tokenize_words,
                                           lev_words_true)

    # Graphique d'étapes

    # on initialise les barres

    bars1 = y_true
    bars2 = y_pred_words

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

# Edition des résultats


def edit_resultats(name, start_time, reference, hypothesis,
                   reference_tokenize_words, hypothesis_tokenize_words,
                   steps_words, steps_letters, wer_percent, wer_without_percentage,
                   cer_percent, cer_without_percentage):

    """Fonction qui édite un rapport à la fin de l'éxécution
    du programme

    :param name: nom de l'utilisateur
    :type: str
    :param start_time: initialisation du compteur
    :type: str
    :param reference: phrase de référence
    :type: str
    :param hypothesis: phrase cible
    :type: str
    :param reference_tokenize_words: phrase de référence tokénisée (mots)
    :type: list char
    :param hypothesis_tokenize_words: phrase cible tokénisée (mots)
    :type: list char
    :param steps_words: liste d'étapes pour les mots
    :type: list int
    :param steps_letters: liste d'étapes pour les lettres
    :type: list int
    :param wer_percent: résultat du WER (%)
    :type: int
    :param wer_without_percentage: résultat du WER
    :type: int
    :param cer_percent: résultat du CER (%)
    :type: int
    :param cer_without_percentage: résultat du CER
    :type: int
    :return: rapport
    :type return : format str
    """
    rapport = f"""

{"*"*40}\n

----RAPPORT----

- Effectué par : {name}
- Date - Heure : {datetime.now()}
- Temps d'execution : {time.time() - start_time} secondes ---

* Phrase de référence           : {reference}\n

* Phrase du modèle              : {hypothesis}\n         

* Phrase de référence tokenisée : {reference_tokenize_words}\n
            * Longeur                       : {len(reference_tokenize_words)}\n

* Phrase du modèle tokénisée    : {hypothesis_tokenize_words}\n
            * Longeur                       : {len(hypothesis_tokenize_words)}\n

* Mots : {steps_words}\n
* Lettres : {steps_letters}\n

* Résultats du WER (en %)       : {wer_percent}
* Résultats du WER              : {wer_without_percentage}\n

* Résultats du CER (en %)       : {cer_percent}
* Résultats du CER              : {cer_without_percentage}

{"*"*40}\n

            """
    return rapport


# fonction principale d'execution du programme

def main(reference_file, hypothesis_file):
    """
    Fonction principale d'execution du programme

    :param reference_file: phrase de référence
    :type str
    :param hypothesis_file: phrase issu du modèle
    :type str
    :return: rapport et graphique (optionnel)
    :type: str & graphique -> matplotlib.object
    """

    name = input("Entrer votre prénom et votre nom : ")

    # On initialise le chrono

    start_time = time.time()

    reference = reference_file
    hypothesis = hypothesis_file

    # STEP 1 : Tokenisation

    reference_tokenize_words, reference_tokenize_letters = tokenize(reference)
    hypothesis_tokenize_words, hypothesis_tokenize_letters = tokenize(hypothesis)

    print("\n", "*"*10, "Tokenisation Done", "*"*10, "\n")

    # STEP 2 : calcul de la distance de Levenshtein (ou distance d'édition) :

    # On utilise ici l'algo numpy de la distance d'édition pour le calcul ultérieur d'étapes

    editdistance_token_words_tosteps = lev_distance(reference_tokenize_words,
                                                    hypothesis_tokenize_words)

    editdistance_token_letters_tosteps = lev_distance(reference_tokenize_letters,
                                                      hypothesis_tokenize_letters)

    # On utilise ici fast_levenstein() du module Kraken pour le calcul ultérieur du CER et
    # du WER

    levenshtein_token_words_tocerwer = _fast_levenshtein(reference_tokenize_words,
                                                         hypothesis_tokenize_words)

    levenshtein_token_letters_tocerwer = _fast_levenshtein(reference_tokenize_letters,
                                                           hypothesis_tokenize_letters)

    print("*" * 10, "Edit distance Done", "*" * 10, "\n")

    # STEP 3 : Calcul d'étapes : insertions (Ins), subtitutions (Subs)
    # et suppressions (Dels)

    steps_words, y_prediction_words = liste_etapes(reference_tokenize_words,
                                                   hypothesis_tokenize_words,
                                                   editdistance_token_words_tosteps)

    steps_letters, y_prediction_letters = liste_etapes(reference_tokenize_letters,
                                                       hypothesis_tokenize_letters,
                                                       editdistance_token_letters_tosteps)

    print("*" * 10, "Steps between reference and hypothesis Done", "*" * 10, "\n")

    # STEP 4a : calcul du WER (Word Error Rate)

    wer_percent, wer_without_percentage = resultwer(reference_tokenize_words,
                                                    levenshtein_token_words_tocerwer)

    # STEP 4b : calcul du CER (Character Error Rate)

    cer_percent, cer_without_percentage = resultcer(reference_tokenize_letters,
                                                    levenshtein_token_letters_tocerwer)

    print("*" * 10, "WER and CER Done", "*" * 10, "\n")

    # STEP 5 : Réalisation du graphique (OPTIONNEL)

    if OPTION_GRAPHIQUE:
        edit_graphique_barres(reference_tokenize_words,
                              y_prediction_words)

    # STEP 6 : Edition des résultats

    rapport = edit_resultats(
        name,
        start_time,
        reference,
        hypothesis,
        reference_tokenize_words,
        hypothesis_tokenize_words,
        steps_words, steps_letters,
        wer_percent,
        wer_without_percentage,
        cer_percent,
        cer_without_percentage)

    # STEP 7 : Ecriture dans un fichier du rapport (Optionnel)

    if OPTION_LOG:
        with open('rapportCERWERTool.txt', 'w') as file:
            file.write(rapport)

    # STEP 8 : Output => affichage du rapport et du graphique (optionnel)

    return print(rapport), plt.show()

# Let's start ! Appel des arguments :

# Pour l'affichage du logo

RESULT = pyfiglet.figlet_format("CERWER Tool", font="bubble")

PARSER = argparse.ArgumentParser(description="*|C|E|R|W|E|R| |T|O|O|L|*"
                                             " A little program to evaluate a Character "
                                             "Error Rate and a Word Error rate "
                                             "between a reference sentence and a hypothesis "
                                             "sentence and edit a graphic")
PARSER.add_argument('--input', '-i', nargs=2, required=True,
                    help='reference file and hypothesis file')
PARSER.add_argument('--graphique', '-g', action='store_true',
                    help='edit a graphic')
PARSER.add_argument('--log', '-l', action='store_true',
                    help='edit a report in txt file')
ARGS = PARSER.parse_args()

# OPTIONS :

OPTION_GRAPHIQUE = vars(ARGS)['graphique']
OPTION_LOG = vars(ARGS)['log']

# Initialisation du programme

print("#"*15, "Welcome to", "#"*15)
print(RESULT)
time.sleep(5)

FILENAME1, FILENAME2 = ARGS.input
with open(str(FILENAME1), 'r', encoding="utf8") as ref:
    REFERENCE_FILE = ref.read()
with open(str(FILENAME2), 'r', encoding="utf8") as hyp:
    HYPOTHESIS_FILE = hyp.read()
main(REFERENCE_FILE, HYPOTHESIS_FILE)
