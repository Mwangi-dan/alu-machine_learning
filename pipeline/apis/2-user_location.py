#!/usr/bin/env python3
"""
Prints ths location of a specific user
"""
import requests
import time
from datetime import datetime


def userLocation(url):
    """
    url: User id passed as first argument of the script

    Returns: the location of a specified user
    """
    response = requests.get(url)

    if response.status_code == 404:
        print("Not Found")
    elif response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        current_time = int(time.time())
        reset_in_minutes = int((reset_time - current_time) / 60)
        print("Reset in {} min".format(reset_in_minutes))
    else:
        print(response.json()["location"])


if __name__ == "__main__":
    import sys

    userLocation(sys.argv[1])
