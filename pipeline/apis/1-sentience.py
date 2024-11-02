#!/usr/bin/env python3
"""
    Return the list of names of the home
    planets of all sentient species
"""

import requests


def sentientPlanets():
    """
    Returns the list of name of home planets
    of all sentient sepcies
    """
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    speciesList = []
    while url:
        response = requests.get(url)
        data = response.json()
        speciesList += data["results"]
        url = data["next"]
    homePlanets = []
    for species in speciesList:
        if species["designation"] == "sentient" or\
           species["classification"] == "sentient":
            homePlanet = requests.get(species["homeworld"]).json()
            homePlanets.append(homePlanet["name"])
    return homePlanets
