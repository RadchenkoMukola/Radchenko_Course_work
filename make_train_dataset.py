import os
from loguru import logger
from typing import Optional

current_dir = os.path.dirname(os.path.realpath(__file__))

positive_folder = os.path.join(current_dir, 'data', 'restaurant_reviews_pos_and_neg', 'positive')
negative_folder = os.path.join(current_dir, 'data', 'restaurant_reviews_pos_and_neg', 'negative')

category_words = {
    'kitchen': ['tasty', 'delicious', 'fresh', 'good', 'great', 'nice', 'yummy', 'flavorful', 'spicy',
                'savory', 'terrible', 'sweet', 'sour', 'bitter', 'umami', 'hot', 'cold', 'warm', 'cool', 'food',
                'meal', 'dish', 'cuisine', 'menu', 'recipe', 'ingredient', 'seasoning', 'sauce', 'drink',
                'beverage', 'appetizer', 'dessert', 'snack', 'breakfast', 'lunch', 'dinner', 'supper',
                'pizza', 'burger', 'salad', 'sushi', 'steak', 'fries', 'chicken', 'fish', 'vegan', 'caesar',
                'vegetarian', 'gluten-free', 'dairy-free', 'spicy', 'sweet', 'sour', 'bitter', 'salty'],
    'service': ['wait', 'service', 'time', 'fast', 'slow', 'quick', 'long', 'short', 'minute', 'hour',
                'friendly', 'rude', 'polite', 'courteous', 'smile', 'frown', 'greet', 'serve', 'server',
                'waiter', 'waitress', 'staff', 'employee', 'crew', 'team', 'help', 'assist', 'customer',
                'client', 'patron', 'tip', 'bill', 'check', 'cash', 'credit', 'pay', 'charge', 'price'],
    'ambience': ['atmosphere', 'ambience', 'mood', 'vibe', 'feeling', 'setting', 'decor', 'furniture',
                 'light', 'music', 'sound', 'noise', 'quiet', 'loud', 'crowd', 'busy', 'empty', 'full',
                 'clean', 'dirty', 'neat', 'tidy', 'messy', 'smell', 'scent', 'aroma', 'fragrance',
                 'odor', 'stink', 'air', 'conditioning', 'temperature', 'ventilation', 'comfort', 'space'],
    'location': ['location', 'place', 'spot', 'area', 'neighborhood', 'district', 'region', 'local',
                 'near', 'far', 'close', 'distance', 'drive', 'parking', 'lot', 'street', 'road', 'highway',
                 'route', 'direction', 'map', 'gps', 'landmark', 'sign', 'view', 'scene', 'scenery',
                 'sight', 'landscape', 'building', 'architecture', 'design', 'style', 'structure']
}


def make_dataset(tag: str) -> Optional[None]:
    if tag == 'positive':
        logger.info('Making positive dataset')
        input_folder = positive_folder
        output_folder = os.path.join(current_dir, 'dataset', 'positive_with_categories')
        overall = 1
    elif tag == 'negative':
        logger.info('Making negative dataset')
        input_folder = negative_folder
        output_folder = os.path.join(current_dir, 'dataset', 'negative_with_categories')
        overall = 1
    else:
        logger.error('Invalid tag')
        return None

    for file in os.listdir(input_folder):

        is_kitchen = False
        is_service = False
        is_ambience = False
        is_location = False

        with open(os.path.join(input_folder, file), 'r') as f:
            review = f.readline().lower()
            review_1 = review[:len(review) // 2]
            review_2 = review[len(review) // 2:]
            if review_1 == review_2:
                review = review_1
            for category in category_words:
                for word in category_words[category]:
                    if word in review:
                        if category == 'kitchen':
                            is_kitchen = True
                        elif category == 'service':
                            is_service = True
                        elif category == 'ambience':
                            is_ambience = True
                        elif category == 'location':
                            is_location = True
                        break

            if tag == 'positive':
                res = f'"{review}",{1 if is_service else 0},0,0,{1 if is_kitchen else 0},' \
                      f'0,0,{1 if is_ambience else 0},0,0,' \
                      f'{1 if is_location else 0},0,0,1,0,0'

            elif tag == 'negative':
                res = f'"{review}",0,0,{1 if is_service else 0},0,0,{1 if is_kitchen else 0},' \
                        f'0,0,{1 if is_ambience else 0},0,0,' \
                        f'{1 if is_location else 0},0,0,1'

            with open(os.path.join(output_folder, file), 'w') as write_file:
                write_file.write(res + '\n')


def main():
    logger.info("Start...")
    make_dataset('positive')
    make_dataset('negative')
    logger.info("Separated datasets done")
    logger.info("Making final dataset")

    with open(os.path.join(current_dir, 'dataset', 'dataset.csv'), 'w') as dataset:

        dataset.write(
            '"review_text","service_good","service_neutral","service_bad","kitchen_good","kitchen_neutral",'
            '"kitchen_bad","ambience_good","ambience_neutral","ambience_bad","location_good","location_neutral",'
            '"location_bad","overall_good","overall_neutral","overall_bad"\n')

        for file in os.listdir(os.path.join(current_dir, 'dataset', 'positive_with_categories')):
            with open(os.path.join(current_dir, 'dataset', 'positive_with_categories', file), 'r') as f:
                dataset.write(f.readline())
        for file in os.listdir(os.path.join(current_dir, 'dataset', 'negative_with_categories')):
            with open(os.path.join(current_dir, 'dataset', 'negative_with_categories', file), 'r') as f:
                dataset.write(f.readline())

    logger.info("Final dataset done")


if __name__ == '__main__':
    main()
