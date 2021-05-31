import h5py
import click

from tqdm import tqdm
import json


@click.command()
@click.option('--source', default='test.h5', help='.hdf5')
@click.option('--output', default='test.json', help='.json')
def start(source, output):
    f = h5py.File(source, 'r')
    data = {'original': load_h5(f, 'original')}
    data['original'] = [[o] if isinstance(o, str) else o for o in data['original']]
    try: # Test Labels
        data['label'] = load_h5(f, 'labels')
        data['label'] = [[o] if isinstance(o[0], int) else o for o in data['label']]
    except Exception:
        pass
    f.close()
   


    with open(output, 'w') as f:
        # assert len(data['label']) == len(data['original'])
        json.dump(data, f)
        


def load_h5(f, label):
    size = len(f.get(label))
    result = [f.get(label).get(str(i))[()].squeeze().tolist() for i in tqdm(range(size))]
    return result

# ['data', 'original', 'products', 'scodes', 'w2v']
if __name__ == '__main__':
    start()