#!/usr/bin/python
#
# convert data into format more suitable for reading into dict
# by merging rows by an index key.

import pandas as pd

def generate_event_apps_dict_file():
    """generates file with mapping event_id -> [app_ids]"""
    app_events = pd.read_csv("../data/app_events.csv")
    event_apps_dict = {}
    for row in app_events.itertuples():
        event_id = row[1]
        app_id = row[2]
        if event_id in event_apps_dict:
            event_apps_dict[event_id].append(app_id)
        else:
            event_apps_dict[event_id] = [app_id]
    with open("../processed_data/event_apps.txt", "w") as f:
        for k,v in event_apps_dict.items():
            print(k, ":", ','.join(str(x) for x in v), sep='', file=f)
    print("done")

def generate_app_labels_dict_file():
    """generates file with mapping app_id -> [label_ids]"""
    app_labels = pd.read_csv("../data/app_labels.csv")
    app_labels_dict = {}
    for row in app_labels.itertuples():
        app_id = row[1]
        label_id = row[2]
        if app_id in app_labels_dict:
            app_labels_dict[app_id].append(label_id)
        else:
            app_labels_dict[app_id] = [label_id]
    with open("../processed_data/app_labels.txt", "w") as f:
        for k,v in app_labels_dict.items():
            print(k, ":", ','.join(str(x) for x in v), sep='', file=f)
    print("done")
