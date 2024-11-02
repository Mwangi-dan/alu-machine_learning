#!/usr/bin/env python3
"""
Displays number of launches per rocket
"""
import requests


def rocket_frequency():
    """
    Returns the number of launches per rocket
    """
    url = 'https://api.spacexdata.com/v4/rockets'
    response = requests.get(url)
    rockets = response.json()

    # Create a dictionary to store the number of launches per rocket
    rocket_frequency = {}
    for rocket in rockets:
        rocket_id = rocket.get('id')
        rocket_name = rocket.get('name')
        launches_url = 'https://api.spacexdata.com/v4/launches'
        response = requests.get(launches_url)
        launches = response.json()
        launches_per_rocket = 0
        for launch in launches:
            if launch.get('rocket') == rocket_id:
                launches_per_rocket += 1
        rocket_frequency[rocket_name] = launches_per_rocket
    return rocket_frequency


if __name__ == "__main__":
    rocket_launch_count = rocket_frequency()
    for rocket, count in rocket_launch_count.items():
        print("{}: {}".format(rocket, count))
