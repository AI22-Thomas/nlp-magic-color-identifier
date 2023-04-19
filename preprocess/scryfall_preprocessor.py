import glob
import json
import re

import requests
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


# Preprocessor for Scryfall data
# https://scryfall.com/docs/api/bulk-data
class ScryfallPreprocessor:
    def __init__(self, file_glob):
        self.file_glob = file_glob
        # Special symbols that need to be replaced, e.g. {T} -> tap
        # Note: Color words are not added to the replacement output, as that would allow the model to "cheat"
        self.special_symbol_replace = {
            '0': '0',
            '1': '1',
            '2': '2',
            '2/B': '2',
            '2/G': '2',
            '2/R': '2',
            '2/U': '2',
            '2/W': '2',
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
            'B': '',
            'B/G': '',
            'B/P': 'phyrexian',
            'B/R': '',
            'C': 'colorless',
            'E': 'energy',
            'G': '',
            'G/P': 'phyrexian',
            'G/U': '',
            'G/U/P': 'phyrexian',
            'G/W': '',
            'G/W/P': 'phyrexian',
            'P': 'phyrexian',
            'Q': 'untap',
            'R': '',
            'R/G': '',
            'R/G/P': 'phyrexian',
            'R/P': 'phyrexian',
            'R/W': '',
            'R/W/P': 'phyrexian',
            'S': 'snow',
            'T': 'tap',
            'TK': 'ticket',
            'U': '',
            'U/B': '',
            'U/P': 'phyrexian',
            'U/R': '',
            'W': '',
            'W/B': '',
            'W/P': 'phyrexian',
            'W/U': '',
            'X': 'X',
        }
        # Card types that will not be used from the data
        # These are cards that do not have a color, or are not part of the normal gameplay
        self.not_allowed_types = ['Land', 'Dungeon', 'Plane', 'Attraction']
        # These are all the card "layouts" that are allowed in the game.
        # The layouts are defined here: https://scryfall.com/docs/api/layouts
        # Certain layouts are excluded, as they are not part of the normal gameplay
        self.allowed_layout = ['normal',
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

    def load_data(self):
        # get first file starting with oracle-cards
        file = glob.glob(self.file_glob)

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
            file = '../data/' + file.split('/')[-1]

            # download file
            r = requests.get(url)
            with open(file, 'wb') as f:
                f.write(r.content)

        file = glob.glob(self.file_glob)[0]

        # read json data from file utf-8 encoded
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    # Extract a card from the data
    def extract_card(self, data, cards, set):
        # if type_line is not in not_allowed_types, continue
        for not_allowed_type in self.not_allowed_types:
            # split and check each for exact match
            if not_allowed_type in data['type_line'].split(' '):
                return

        oracle_text = data['oracle_text']

        # convert all <number>/<number> to special tokens in rules_text
        # e.g. 2/2 -> <2/2>, */* -> <*/*>, +1/+1 -> <+1/+1>
        # this is done to prevent the tokenizer from splitting the numbers
        oracle_text = re.sub(r'([+-]?[\d\*XY]+)/([+-]?[\d\*XY]+)', r'<\1/\2>', oracle_text)

        # replace special symbols
        for key, value in self.special_symbol_replace.items():
            oracle_text = oracle_text.replace('{' + key + '}', ' ' + value)

        # replace all mentions of the cardname with "CARDNAME"
        # this is done to "anyonymize" the card, so the model does not learn the name of the card
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

    # Extract all cards from the bulk data
    def extract_cards(self, data):
        cards = []
        print('Extracting cards...')
        for card in data:
            # if layout is not in allowed_layout, continue
            if card['layout'] not in self.allowed_layout:
                continue
            # if set type funny, continue
            # "funny" are joke sets, like Unglued, Unhinged, etc.
            # They are intentionally designed to be rule breaking, and are thus not included
            if card['set_type'] == 'funny':
                continue
            set = card['set']
            # if card has card_faces, extract each card face
            if 'card_faces' in card:
                for face in card['card_faces']:
                    self.extract_card(face, cards, set)
            else:
                self.extract_card(card, cards, set)

        return cards

    # Save the cards to a json file
    def save_cards(self, cards, path):
        print('Saving cards...')
        # save as json file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cards, f, ensure_ascii=False)

    # Stop word removal, and special character removal
    # This destroys special tokens like <2/2>, but they are not needed for training which removes stop words
    def stopword_preprocessing(self, text):
        quoteRemoval = text.replace('"', '')
        spaceRemoval = re.sub("\s\s+", " ", quoteRemoval)
        stringRemoval = spaceRemoval.strip()
        specialChar = re.sub(r"[^a-zA-Z]+", ' ', stringRemoval)
        stop = set(stopwords.words('english'))  # to remove stop words like is,the,that etc
        stpwords = ' '.join([i for i in specialChar.lower().split() if i not in stop])
        return stpwords

    # helper function to return all the relevant rules text
    def train_text(self, card):
        input_text = card['type_line']
        if card['rules_text'] is not None:
            input_text += '\n' + card['rules_text']
        if card['power'] is not None:
            input_text += '\n<' + card['power'] + '/' + card['toughness'] + '>'
        return input_text

    # helper function to return a card from scryfall
    def get_card(self, card_name):
        # get card from scryfall, fuzzy search
        req = requests.get('https://api.scryfall.com/cards/named?fuzzy=' + card_name)
        if req.status_code == 200:
            return req.json()
        else:
            return None
