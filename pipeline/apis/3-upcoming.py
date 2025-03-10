#!/usr/bin/env python3
"""
Displays the upcoming launches for SpaceX
"""

import requests
from datetime import datetime, timezone, timedelta


def upcoming_launch():
    """
    Returns upcoming launches:
    - Name of launch
    - Date
    - Rocket name
    - Name of lauchpad
    """
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    response = requests.get(url)
    launches = response.json()

    # Sorting
    # Sort launches by date_unix to find the soonest launch
    launches.sort(key=lambda x: x["date_unix"])
    upcoming_launch = launches[0]

    launch_name = upcoming_launch.get('name')
    date_unix = upcoming_launch.get('date_unix')
    rocket_id = upcoming_launch.get('rocket')
    launchpad_id = upcoming_launch.get('launchpad')

    # Formating date
    launch_date_utc = datetime.fromtimestamp(date_unix, tz=timezone.utc)
    launch_date_local = launch_date_utc.astimezone(
        timezone(timedelta(hours=-4)))
    launch_date_str = launch_date_local.strftime('%Y-%m-%dT%H:%M:%S%z')
    launch_date_str = "{}:{}".format(
        launch_date_str[:-2], launch_date_str[-2:]
    )

    # Rocket name
    rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    rocket_response = requests.get(rocket_url)
    rocket_name = rocket_response.json()["name"]

    # Launchpad details
    launchpad_url = "https://api.spacexdata.com/v4/launchpads/{}".format(
        launchpad_id
    )
    launchpad_response = requests.get(launchpad_url)
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data["name"]
    launchpad_locality = launchpad_data["locality"]

    output = "{} ({}) {} - {} ({})".format(
        launch_name,
        launch_date_str,
        rocket_name,
        launchpad_name,
        launchpad_locality
    )

    return output


if __name__ == "__main__":
    print(upcoming_launch())
