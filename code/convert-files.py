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
    write_dict_to_file(event_apps_dict, "../processed_data/event_apps.txt")
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
    write_dict_to_file(app_labels_dict, "../processed_data/app_labels.txt")
    print("done")

def generate_device_events_dict_file():
    """generates file with mapping device_id -> [event_ids]"""
    events = pd.read_csv("../data/events.csv")
    device_events_dict = {}
    for row in events.itertuples():
        event_id = row[1]
        device_id = row[2]
        if device_id in device_events_dict:
            device_events_dict[device_id].append(event_id)
        else:
            device_events_dict[device_id] = [event_id]
    write_dict_to_file(device_events_dict, "../processed_data/device_events.txt")
    print("done")

def write_dict_to_file(d, fpath):
    """write dict d to file given by path fpath"""
    with open(fpath, "w") as f:
        for k,v in d.items():
            print(k, ":", ','.join(str(x) for x in v), sep='', file=f)

generate_device_events_dict_file()
