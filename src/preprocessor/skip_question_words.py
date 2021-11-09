from typing import List


def skip_question_words() -> List[str]:
    filter_elems = [
        "எத்தனை",  # how many
        "எங்கு",  # where
        "யார்",  # who
        "எவ்வாறு",  # what (?)
        "உள்ள",  # anything within (?)
        "எது",  # which
        "எப்பொழுது",  # when
        "எப்போது",  # when
        "ஆகும்",  # is
        "என்ன",  # what
        "எவ்வளவு",  # how much
        "எந்த",  # No
        "எப்படி",  # how
        "யாரால்",  # by whom
        "எங்கே",  # where
        "किसने",  # who
        "क्या",  # what
        "कहाँ",  # where
        "कितने",  # how many
        "किस",  # what
        "कीहै",  # has (means what w/ ?)
        "कितनी",  # how many
        "कब",  # when
        "कौन",  # who
        "कहा",  # said  (where)
        "कितना",  # how much
        "का",  # NS (?)
        "किन",  # why
    ]
    return filter_elems
