#!/usr/bin/python
#
# generates training data containing:
# device_id, gender, age, group, phone_brand, device_model, [event_ids], [app_ids], [label_ids]
# and writes dataframe to pickle.

import numpy as np
import os
import pandas as pd
import pickle
import time

PICKLE_OUTPUT_FILE = "../processed_data/pickle/data.pickle"

def read_dict_file(fpath):
    d = {}
    with open(fpath, "r") as f:
        for line in f:
            key, vals = line.rstrip().split(":")
            vals_a = [int(s) for s in vals.split(",")]
            d[int(key)] = vals_a
    return d


def get_event_ids_optimized(device_id, device_events_dict):
    """gets list of event_ids matching given devide_id"""
    if device_id in device_events_dict:
        return device_events_dict[device_id]
    else:
        return []

def get_event_ids(device_id, events):
    """gets list of event_ids matching given devide_id"""
    return events[events["device_id"] == device_id]["event_id"].tolist()

def get_app_ids(event_ids, event_apps_dict):
    """get list of unique installed app ids based on event_apps_dict."""
    app_ids = set()
    for event_id in event_ids:
        if event_id in event_apps_dict:
            for app_id in event_apps_dict[event_id]:
                app_ids.add(app_id)
    return list(app_ids)

def get_label_ids(app_ids, app_labels_dict):
    """get list of unique category label ids based on installed app ids and dict app_id -> app_label."""
    label_ids = set()
    for app_id in app_ids:
        if app_id in app_labels_dict:
            for label_id in app_labels_dict[app_id]:
                label_ids.add(label_id)
    return list(label_ids)


# Read Data
event_apps_dict = read_dict_file("../processed_data/event_apps.txt")
app_labels_dict = read_dict_file("../processed_data/app_labels.txt")
device_events_dict = read_dict_file("../processed_data/device_events.txt")
events = pd.read_csv("../data/events.csv")
train = pd.read_csv("../data/gender_age_train.csv")
phone_device = pd.read_csv("../data/phone_brand_device_model.csv")
brand_dict = pd.read_csv("../processed_data/phone_brands.csv", index_col=0).to_dict()['phone_brand_id']
model_dict = pd.read_csv("../processed_data/device_models.csv", index_col=0).to_dict()['device_model_id']
print("done reading data")

# Join with phone device data
df = pd.merge(train, phone_device, on='device_id')
print(df.shape)
print(len(df))
df["phone_brand_id"] = df["phone_brand"].map(lambda s: brand_dict[s])
df["device_model_id"] = df["device_model"].map(lambda s: model_dict[s])
df.drop(["device_model", "phone_brand"], inplace=True, axis=1)

# Join with event_ids, app_ids and app_labels
print("start map")
print(time.time())
#df["event_ids"] = df["device_id"].map(lambda _id: get_event_ids_optimized(_id, device_events_dict))
df["event_ids"] = df["device_id"].map(lambda _id: get_event_ids(_id, events))
print("done map 1")
print(time.time())
df["app_ids"] = df["event_ids"].map(lambda event_ids: get_app_ids(event_ids, event_apps_dict))
print("done map 2")
print(time.time())
df["label_ids"] = df["app_ids"].map(lambda app_ids: get_label_ids(app_ids, app_labels_dict))
print(time.time())
print("start pickling")
pickle.dump(df, open(PICKLE_OUTPUT_FILE, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

print("done")
