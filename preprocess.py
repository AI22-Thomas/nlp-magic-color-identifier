from scryfall_preprocessor import ScryfallPreprocessor


sf_preprocessor = ScryfallPreprocessor('data/oracle-cards*.json')
data = sf_preprocessor.load_data()
cards = sf_preprocessor.extract_cards(data)
sf_preprocessor.save_cards(cards, 'data/cards.json')