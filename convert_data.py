import os
import pickle
import argparse
import numpy as np
import pandas as pd


def convert_csv(path):
    old = pd.read_csv(path)
    columns = [c for c in old.columns]
    features = [f for f in columns if f.startswith('feature ')]
    demographics = [d for d in columns if d.startswith('demo ')]
    new = {'available_demographics': demographics, 'data': {}}
    for i, row in old.iterrows():
        new['data'][i] = {demo: row[demo] for demo in demographics}
        new['data'][i]['learner_id'] = i
        new['data'][i]['target'] = row['label']
        new['data'][i]['features'] = np.hstack([row[f] for f in features])

    return new

def save(settings, new):
    if settings['name'] == '':
        name = 'personal_data'
    else:
        name = settings['name']
    folder = 'data/{}/'.format(name)
    os.makedirs(folder, exist_ok=True)
    with open('data/{}/data_dictionary.pkl'.format(name), 'wb') as fp:
        pickle.dump(new, fp)


def main(settings):
    new_data = convert_csv(settings['path'])
    save(settings, new_data)

if __name__ == '__main__': 
    settings = {}
    parser = argparse.ArgumentParser(description='Plot the results')
    parser.add_argument('--path', dest='path', default='', action='store')
    parser.add_argument('--name', dest='name', default='', action='store')
    settings.update(vars(parser.parse_args()))

    main(settings)