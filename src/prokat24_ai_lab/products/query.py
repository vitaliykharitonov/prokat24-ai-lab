from collections.abc import Sequence

from prokat24_ai_lab.text_normalization import normalize_text

RENTAL_PERIOD_ALIASES = {
    "2 тижні": [
        "2 тижні",
        "два тижні",
    ],
    "Тиждень": [
        "на тиждень",
        "тиждень",
    ],
    "1-й день": [
        "1 день",
        "один день",
    ],
}

PRODUCT_QUERY_STOP_WORDS = {
    "скільки",
    "коштує",
    "вартість",
    "ціна",
    "за",
    "що",
    "чи",
    "є",
    "на",
    "в",
    "у",
    "вас",
    "оренда",
    "прокат",
    "тиждень",
    "тижні",
    "застава",
    "заставу",
    "який",
    "яка",
    "яку",
    "яке",
    "які",
    "якого",
    "якої",
    "наявності",
    "наявність",
    "маєте",
    "характеристики",
    "опис",
    "комплектація",
    "комплект",
    "комплекті",
    "входить",
    "розмір",
    "розміру",
    "розміри",
    "вага",
    "ваги",
    "важить",
}

PRODUCT_QUERY_STOP_WORD_SEQUENCES = (("буде", "коштувати"),)

DEPOSIT_INTENT_WORDS = {
    "застава",
    "заставу",
}

AVAILABILITY_INTENT_WORDS = {
    "наявності",
    "наявність",
    "маєте",
}

AVAILABILITY_INTENT_WORD_SEQUENCES = (
    ("чи", "є"),
    ("у", "вас", "є"),
)

DETAILS_INTENT_WORDS = {
    "характеристики",
    "опис",
    "комплектація",
    "комплект",
    "входить",
    "розмір",
    "вага",
}

ATTRIBUTE_ALIASES = {
    "Комплектація": {
        "комплектація",
        "комплект",
        "комплекті",
    },
    "Розмір": {
        "розмір",
        "розміру",
        "розміри",
    },
    "Вага": {
        "вага",
        "ваги",
        "важить",
    },
}

RENTAL_DAY_WORDS = {
    "день",
    "дні",
    "дня",
    "днів",
}

PRICE_AND_DEPOSIT_WORDS = {
    "ціна",
    "вартість",
    "коштує",
    "коштувати",
    "застава",
    "заставу",
}

UKRAINIAN_NUMBER_WORDS = {
    "один": 1,
    "два": 2,
    "три": 3,
    "чотири": 4,
    "пʼять": 5,
    "п'ять": 5,
    "шість": 6,
}


def remove_word_sequence(words: list[str], sequence: Sequence[str]) -> list[str]:
    sequence_words = list(sequence)
    sequence_length = len(sequence_words)

    for start_index in range(len(words)):
        end_index = start_index + sequence_length

        if words[start_index:end_index] == sequence_words:
            return words[:start_index] + words[end_index:]

    return words


def contains_word_sequence(words: list[str], sequence: Sequence[str]) -> bool:
    sequence_words = list(sequence)

    if not sequence_words:
        return False

    sequence_length = len(sequence_words)

    for start_index in range(len(words) - sequence_length + 1):
        end_index = start_index + sequence_length

        if words[start_index:end_index] == sequence_words:
            return True

    return False


def build_product_search_query(query: str) -> str:
    words = normalize_text(query).split()
    period = detect_rental_period(query)
    day_count = detect_rental_day_count(query)

    words = remove_rental_period_words(words, period)
    words = remove_rental_day_count_words(words, day_count)

    for sequence in PRODUCT_QUERY_STOP_WORD_SEQUENCES:
        words = remove_word_sequence(words, sequence)

    product_words = []

    for word in words:
        if word in PRODUCT_QUERY_STOP_WORDS:
            continue

        product_words.append(word)

    return " ".join(product_words)


def detect_rental_period(query: str) -> str | None:
    normalized_query = normalize_text(query)

    for period, aliases in RENTAL_PERIOD_ALIASES.items():
        for alias in aliases:
            if normalize_text(alias) in normalized_query:
                return period

    return None


def detect_rental_day_count(query: str) -> int | None:
    query_words = normalize_text(query).split()

    for index in range(len(query_words) - 1):
        current_word = query_words[index]
        next_word = query_words[index + 1]

        if next_word not in RENTAL_DAY_WORDS:
            continue

        if current_word.isdigit():
            return int(current_word)

        if current_word in UKRAINIAN_NUMBER_WORDS:
            return UKRAINIAN_NUMBER_WORDS[current_word]

    return None


def remove_rental_period_words(words: list[str], period: str | None) -> list[str]:
    if period is None:
        return words

    aliases = RENTAL_PERIOD_ALIASES[period]

    for alias in aliases:
        alias_words = normalize_text(alias).split()
        updated_words = remove_word_sequence(words, alias_words)

        if updated_words != words:
            return updated_words

    return words


def remove_rental_day_count_words(words: list[str], day_count: int | None) -> list[str]:
    if day_count is None:
        return words

    for index in range(len(words) - 1):
        current_word = words[index]
        next_word = words[index + 1]

        if next_word not in RENTAL_DAY_WORDS:
            continue

        if current_word.isdigit() and int(current_word) == day_count:
            return remove_word_sequence(words, [current_word, next_word])

        if UKRAINIAN_NUMBER_WORDS.get(current_word) == day_count:
            return remove_word_sequence(words, [current_word, next_word])

    return words


def detect_deposit_intent(query: str) -> bool:
    query_words = set(normalize_text(query).split())

    return bool(query_words & DEPOSIT_INTENT_WORDS)


def detect_availability_intent(query: str) -> bool:
    words = normalize_text(query).split()
    query_words = set(words)

    if query_words & PRICE_AND_DEPOSIT_WORDS:
        return False

    if query_words & AVAILABILITY_INTENT_WORDS:
        return True

    for sequence in AVAILABILITY_INTENT_WORD_SEQUENCES:
        if contains_word_sequence(words, sequence):
            return True

    return bool(words and words[0] == "є")


def detect_product_details_intent(query: str) -> bool:
    query_words = set(normalize_text(query).split())

    return bool(query_words & DETAILS_INTENT_WORDS)


def detect_requested_attribute_label(query: str) -> str | None:
    query_words = set(normalize_text(query).split())

    for attribute, aliases in ATTRIBUTE_ALIASES.items():
        if query_words & aliases:
            return attribute

    return None
