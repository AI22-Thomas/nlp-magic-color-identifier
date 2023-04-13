import glob
import json
import requests
import re

# get first file starting with oracle-cards
file = glob.glob('data/oracle-cards*')

# if no file found, download it
if not file:
    print('Downloading oracle-cards.json...')
    # get json from https://api.scryfall.com/bulk-data
    r = requests.get('https://api.scryfall.com/bulk-data')
    data = r.json()

    # get oracle-cards url
    url = [d['download_uri'] for d in data['data'] if d['type'] == 'oracle_cards'][0]

    # get file name from url
    file = re.search(r'oracle-cards.*\.json', url).group(0)

    # remove folder from file name
    file = 'data/' + file.split('/')[-1]

    # download file
    r = requests.get(url)
    with open(file, 'wb') as f:
        f.write(r.content)

file = glob.glob('data/oracle-cards*')[0]

# read json data from file utf-8 encoded
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

special_symbol_replace = {
    '0': '0',
    '1': '1',
    '2': '2',
    '2/B': '2 black',
    '2/G': '2 green',
    '2/R': '2 red',
    '2/U': '2 blue',
    '2/W': '2 white',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '10': '10',
    '12': '12',
    '20': '20',
    'B': 'black',
    'B/G': 'black green',
    'B/P': 'black phyrexian',
    'B/R': 'black red',
    'C': 'colorless',
    'E': 'energy',
    'G': 'green',
    'G/P': 'green phyrexian',
    'G/U': 'green blue',
    'G/U/P': 'green blue phyrexian',
    'G/W': 'green white',
    'G/W/P': 'green white phyrexian',
    'P': 'phyrexian',
    'Q': 'untap',
    'R': 'red',
    'R/G': 'red green',
    'R/G/P': 'red green phyrexian',
    'R/P': 'red phyrexian',
    'R/W': 'red white',
    'R/W/P': 'red white phyrexian',
    'S': 'snow',
    'T': 'tap',
    'TK': 'ticket',
    'U': 'blue',
    'U/B': 'blue black',
    'U/P': 'blue phyrexian',
    'U/R': 'blue red',
    'W': 'white',
    'W/B': 'white black',
    'W/P': 'white phyrexian',
    'W/U': 'white blue',
    'X': 'X',
}

cards = []

not_allowed_types = ['Land', 'Dungeon', 'Plane', 'Attraction']


def extract_card(data, cards, set):
    # if type_line is not in not_allowed_types, continue
    for not_allowed_type in not_allowed_types:
        if not_allowed_type in data['type_line']:
            return

    oracle_text = data['oracle_text']

    # replace special symbols
    for key, value in special_symbol_replace.items():
        oracle_text = oracle_text.replace('{' + key + '}', ' ' + value)

    # replace all mentions of cardname with CARDNAME
    oracle_text = oracle_text.replace(data['name'], 'CARDNAME')

    colors = data['colors'] if 'colors' in data else []
    color_identity = data['color_identity'] if 'color_identity' in data else []

    # transform arrays to numbers, with 0 being white, 1 being blue, 2 being black, 3 being red, 4 being green, 5 being colorless
    # 1 means the color is in the color, 0 means it is not
    colors = [1 if color in colors else 0 for color in ['W', 'U', 'B', 'R', 'G']]
    colors.append(1 if sum(colors) == 0 else 0)

    # same as above, but for color identity
    color_identity = [1 if color in color_identity else 0 for color in ['W', 'U', 'B', 'R', 'G']]
    color_identity.append(1 if sum(color_identity) == 0 else 0)

    cards.append({
        'name': data['name'],
        'rules_text': oracle_text,
        'colors': colors,
        'color_identity': color_identity,
        'flavour_text': 'flavour_text' in data and data['flavour_text'] or None,
        'type_line': data['type_line'],
        'power': 'power' in data and data['power'] or None,
        'toughness': 'toughness' in data and data['toughness'] or None,
        'set': set,
    })


allowed_layout = ['normal',
                  'split',
                  'flip',
                  'transform',
                  'meld',
                  'leveler',
                  'saga',
                  'class',
                  'adventure',
                  'battle',
                  'meld']

print('Extracting cards...')
for card in data:
    # if layout is not in allowed_layout, continue
    if card['layout'] not in allowed_layout:
        continue
    # if set type funny, continue
    if card['set_type'] == 'funny':
        continue
    set = card['set']
    # if card has card_faces, extract each card face
    if 'card_faces' in card:
        for face in card['card_faces']:
            extract_card(face, cards, set)
    else:
        extract_card(card, cards, set)

print('Saving cards...')
# save as json file
with open('data/cards.json', 'w', encoding='utf-8') as f:
    json.dump(cards, f, ensure_ascii=False)
