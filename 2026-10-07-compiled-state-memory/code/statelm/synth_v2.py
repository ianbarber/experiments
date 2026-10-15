"""Sealed synthetic test items for the Phase-1 gate (v2). Disjoint from the P0/P0b dev items
(synth.ATTRS, synth_prefs.PREFS). Do not tune anything on these.

FACTS2: (key, template "The user ... {v}.", question, lead-in, X, Y)
  The colleague version is template.replace("The user", "The user's colleague", 1).
PREFS2: (key, thing, question, continuation that embraces the thing, continuation that avoids it)
  fact form:        X "The user loves {thing}."                  Y "The user can't stand {thing}."
  instruction form: X "Always favour {thing} in suggestions for the user."
                    Y "Always avoid {thing} in suggestions for the user."
  Both memories mention {thing}, so lexical priming of {thing} cancels in the counterfactual shift.

FILLER2: neutral filler with no preference-like or attribute-like entries (fixes the v1 filler leak).
"""
import random

FACTS2 = [
    ("best_friend", "The user's best friend is called {v}.", "What is my best friend's name?", "Your best friend's name is", "Tobias", "Marcus"),
    ("bank", "The user banks with {v}.", "Which bank do I use?", "You bank with", "Monzo", "Barclays"),
    ("football", "The user supports {v} football club.", "Which football club do I support?", "You support", "Arsenal", "Liverpool"),
    ("car_colour", "The user's car is {v}.", "What colour is my car?", "Your car is", "red", "silver"),
    ("birth_month", "The user was born in {v}.", "Which month was I born in?", "You were born in", "March", "October"),
    ("partner", "The user's partner is called {v}.", "What is my partner's name?", "Your partner's name is", "Elena", "Priya"),
    ("street", "The user lives on {v} Street.", "Which street do I live on?", "You live on", "Baker", "Mill"),
    ("phone", "The user uses a {v} phone.", "What brand of phone do I use?", "You use a", "Samsung", "Nokia"),
    ("degree", "The user studied {v} at university.", "What did I study at university?", "You studied", "chemistry", "history"),
    ("colour", "The user's favourite colour is {v}.", "What is my favourite colour?", "Your favourite colour is", "teal", "orange"),
    ("language", "The user is learning {v}.", "Which language am I learning?", "You are learning", "Japanese", "Portuguese"),
    ("gym_day", "The user goes to the gym on {v}s.", "Which day do I go to the gym?", "You go to the gym on", "Tuesday", "Friday"),
    ("son", "The user's son is called {v}.", "What is my son's name?", "Your son's name is", "Oliver", "Felix"),
    ("daughter", "The user's daughter is called {v}.", "What is my daughter's name?", "Your daughter's name is", "Maya", "Chloe"),
    ("brother", "The user's brother is called {v}.", "What is my brother's name?", "Your brother's name is", "Daniel", "Samuel"),
    ("airline", "The user usually flies with {v}.", "Which airline do I usually fly with?", "You usually fly with", "Lufthansa", "Emirates"),
    ("topping", "The user's favourite pizza topping is {v}.", "What is my favourite pizza topping?", "Your favourite pizza topping is", "pineapple", "mushroom"),
    ("shoe_size", "The user's shoe size is {v}.", "What is my shoe size?", "Your shoe size is", "nine", "eleven"),
    ("dentist", "The user's dentist is Dr {v}.", "What is my dentist's name?", "Your dentist is Dr", "Patel", "Morgan"),
    ("wifi", "The user's Wi-Fi network is called {v}.", "What is my Wi-Fi network called?", "Your Wi-Fi network is called", "Nebula", "Starling"),
    ("editor", "The user writes code in {v}.", "Which text editor do I use?", "You use", "Vim", "Emacs"),
    ("prog_lang", "The user's main programming language is {v}.", "What is my main programming language?", "Your main programming language is", "Rust", "Kotlin"),
    ("tea_extra", "The user takes {v} in their tea.", "What do I take in my tea?", "You take", "honey", "lemon"),
    ("desk_plant", "The user keeps a {v} on their desk.", "What plant is on my desk?", "You keep a", "cactus", "fern"),
    ("bike", "The user rides a {v} bike.", "What brand of bike do I ride?", "You ride a", "Trek", "Brompton"),
    ("board_game", "The user's favourite board game is {v}.", "What is my favourite board game?", "Your favourite board game is", "Catan", "Scrabble"),
    ("tv_show", "The user's favourite TV show is {v}.", "What is my favourite TV show?", "Your favourite TV show is", "Succession", "Seinfeld"),
    ("podcast", "The user's favourite podcast is {v}.", "What is my favourite podcast?", "Your favourite podcast is", "Radiolab", "Serial"),
    ("painter", "The user's favourite painter is {v}.", "Who is my favourite painter?", "Your favourite painter is", "Monet", "Rothko"),
    ("composer", "The user's favourite composer is {v}.", "Who is my favourite composer?", "Your favourite composer is", "Chopin", "Mahler"),
    ("fruit", "The user's favourite fruit is {v}.", "What is my favourite fruit?", "Your favourite fruit is", "mango", "cherry"),
    ("flower", "The user's favourite flower is the {v}.", "What is my favourite flower?", "Your favourite flower is the", "tulip", "orchid"),
    ("season", "The user's favourite season is {v}.", "What is my favourite season?", "Your favourite season is", "autumn", "spring"),
    ("lucky_number", "The user's lucky number is {v}.", "What is my lucky number?", "Your lucky number is", "seven", "thirteen"),
    ("middle_name", "The user's middle name is {v}.", "What is my middle name?", "Your middle name is", "James", "Edward"),
    ("grandmother", "The user's grandmother lives in {v}.", "Where does my grandmother live?", "Your grandmother lives in", "Seville", "Krakow"),
    ("honeymoon", "The user went to {v} on their honeymoon.", "Where did I go on my honeymoon?", "You went to", "Bali", "Iceland"),
    ("first_job", "The user's first job was as a {v}.", "What was my first job?", "Your first job was as a", "lifeguard", "waiter"),
    ("job", "The user works as a {v}.", "What is my job?", "You work as a", "pharmacist", "architect"),
    ("manager", "The user's manager is called {v}.", "What is my manager's name?", "Your manager's name is", "Grace", "Victor"),
    ("landlord", "The user's landlord is called {v}.", "What is my landlord's name?", "Your landlord's name is", "Rupert", "Gareth"),
    ("ice_cream", "The user's favourite ice cream flavour is {v}.", "What is my favourite ice cream flavour?", "Your favourite ice cream flavour is", "pistachio", "vanilla"),
    ("lunch", "The user's usual lunch is a {v} sandwich.", "What is my usual lunch?", "Your usual lunch is a", "tuna", "falafel"),
    ("watch", "The user wears a {v} watch.", "What brand of watch do I wear?", "You wear a", "Casio", "Seiko"),
    ("superhero", "The user's favourite superhero is {v}.", "Who is my favourite superhero?", "Your favourite superhero is", "Batman", "Superman"),
    ("trip", "The user is planning a trip to {v} next year.", "Where am I planning to travel next year?", "You are planning a trip to", "Peru", "Vietnam"),
    ("anniversary", "The user's wedding anniversary is in {v}.", "Which month is my wedding anniversary?", "Your wedding anniversary is in", "June", "September"),
    ("neighbour", "The user's next-door neighbour is called {v}.", "What is my next-door neighbour's name?", "Your next-door neighbour's name is", "Harold", "Simon"),
]

PREFS2 = [
    ("spicy", "spicy food", "Suggest a curry for me to order tonight.", "Try a fiery vindaloo.", "Try a mild korma."),
    ("jazz", "jazz", "Recommend some music for my evening.", "Put on some classic jazz.", "Put on some indie pop."),
    ("horror", "horror films", "What film should I watch tonight?", "Watch a scary horror film.", "Watch a light romantic comedy."),
    ("cold", "cold weather", "Where should I go on holiday in January?", "Head north to Lapland.", "Head south to the Canary Islands."),
    ("crowds", "crowds", "What should I do this Saturday?", "Go to the big street festival in town.", "Take a quiet walk in the countryside."),
    ("long_novels", "long novels", "Recommend a book for my holiday.", "Try War and Peace.", "Try a short story collection."),
    ("early", "early mornings", "When should I schedule my workout?", "Do it at 6am before work.", "Do it in the evening after work."),
    ("seafood", "seafood", "What should I cook for dinner?", "Cook some garlic prawns.", "Cook a mushroom risotto."),
    ("coasters", "rollercoasters", "Suggest a day out for the weekend.", "Visit a theme park with big rollercoasters.", "Visit a botanical garden."),
    ("hiking", "hiking", "What should I do on my day off?", "Go for a long hike in the hills.", "Visit a museum in town."),
    ("chocolate", "chocolate", "Suggest a dessert for my birthday.", "Have a rich chocolate cake.", "Have a lemon tart."),
    ("cats", "cats", "Should I get a pet?", "Adopt a cat.", "Get a goldfish."),
    ("camping", "camping", "How should I spend the long weekend?", "Go camping in the forest.", "Book a nice hotel in the city."),
    ("opera", "opera", "Suggest an evening out.", "See an opera at the theatre.", "Go to a comedy club."),
    ("running", "running", "Suggest a way to get fit.", "Start running three times a week.", "Try a cycling class."),
    ("garlic", "garlic", "Suggest a pasta sauce.", "Make a garlic and chilli sauce.", "Make a simple butter sauce."),
    ("video_games", "video games", "What should I do to relax tonight?", "Play a video game.", "Read a novel."),
    ("beaches", "beaches", "Where should I go for a summer holiday?", "Go to a beach resort in Greece.", "Go to a mountain lodge in Austria."),
    ("cooking", "cooking", "What should I do about dinner tonight?", "Cook something new from scratch.", "Order a takeaway."),
    ("speaking", "public speaking", "Should I volunteer to give the team presentation?", "Yes, volunteer to give it.", "No, let someone else give it."),
    ("dogs", "dogs", "Should I go to the dog show this weekend?", "Yes, go to the dog show.", "No, skip the dog show."),
    ("coffee", "coffee", "What should I drink at the cafe?", "Order a flat white.", "Order a herbal tea."),
    ("sushi", "sushi", "Where should we go for lunch?", "Go to the sushi bar.", "Go to the sandwich shop."),
    ("museums", "museums", "What should I do in Paris?", "Spend a day at the Louvre.", "Go shopping on the Champs-Elysees."),
    ("rain", "rainy weather", "Which city should I move to?", "Move to Seattle.", "Move to Phoenix."),
    ("parties", "parties", "How should I celebrate my birthday?", "Throw a big party.", "Have a quiet dinner at home."),
    ("gardening", "gardening", "What hobby should I take up?", "Start a vegetable garden.", "Learn to play chess."),
    ("flying", "flying", "How should I travel to Edinburgh?", "Take a short flight.", "Take the train."),
    ("musicals", "musicals", "What should I see in the West End?", "See a big musical.", "See a serious play."),
    ("spreadsheets", "spreadsheets", "How should I plan my budget?", "Build a detailed spreadsheet.", "Use a simple budgeting app."),
    ("yoga", "yoga", "Suggest a class at the gym.", "Join the yoga class.", "Join the boxing class."),
    ("sweet_drinks", "sweet drinks", "What should I drink with lunch?", "Have a milkshake.", "Have a sparkling water."),
    ("small_talk", "small talk", "How should I approach the networking event?", "Chat with as many people as possible.", "Find one person for a deeper conversation."),
    ("road_trips", "road trips", "How should I travel around Scotland?", "Rent a car and drive.", "Take the trains."),
    ("olives", "olives", "Suggest a pizza topping.", "Add black olives.", "Add sweetcorn."),
    ("board_games", "board games", "What should we do on family night?", "Play a board game.", "Watch a film."),
    ("swimming", "swimming", "Suggest some exercise for this afternoon.", "Go for a swim.", "Go for a walk."),
    ("classical", "classical music", "Suggest some background music for studying.", "Play some Bach.", "Play some lo-fi beats."),
    ("spiders", "spiders", "Should I visit the insect house at the zoo?", "Yes, visit the insect house.", "No, skip the insect house."),
    ("big_cities", "big cities", "Where should I spend a long weekend?", "Spend it in New York.", "Spend it in a small seaside village."),
    ("mint", "mint", "Which ice cream flavour should I get?", "Get mint chocolate chip.", "Get strawberry."),
    ("documentaries", "documentaries", "What should I watch tonight?", "Watch a nature documentary.", "Watch a comedy series."),
    ("winter_sports", "winter sports", "Plan a February holiday for me.", "Book a ski trip to the Alps.", "Book a city break in Rome."),
    ("baking", "baking", "What should I do on a rainy Sunday?", "Bake a loaf of bread.", "Do a jigsaw puzzle."),
    ("cycling", "cycling", "How should I commute to work?", "Cycle to work.", "Take the bus."),
    ("poetry", "poetry", "Suggest a gift for my reading group.", "Give them a poetry anthology.", "Give them a crime novel."),
    ("heights", "heights", "Should I book the glass-floor observation deck?", "Yes, book the observation deck.", "No, skip the observation deck."),
    ("tea", "tea", "What should I drink to wake up?", "A strong cup of breakfast tea.", "A glass of orange juice."),
]

PREF_FORMS = {
    "fact": ("The user loves {t}.", "The user can't stand {t}."),
    "instruction": ("Always favour {t} in suggestions for the user.", "Always avoid {t} in suggestions for the user."),
}

FILLER2 = [
    "The user mentioned a {a} trip planned for {m}.",
    "The user asked for advice about {c}.",
    "The user said their {d} has been {e} lately.",
    "The user finished a course on {c}.",
    "The user asked how to fix a problem with their {x}.",
    "The user shared an article about {c}.",
    "The user had a meeting about {c} on {m}.",
    "The user asked for a summary of a report on {c}.",
]
FILL2_VOCAB = dict(
    a=["work", "family", "weekend", "conference"],
    m=["Monday", "the 14th", "next week", "the end of the month", "Thursday"],
    c=["tax returns", "home networking", "spreadsheet formulas", "insurance renewals", "bird watching",
       "retirement savings", "time management", "local history", "printer drivers", "recycling rules"],
    d=["commute", "sleep", "workload", "neighbourhood", "inbox", "schedule"],
    e=["better", "worse", "unpredictable", "quieter", "busier", "fine"],
    x=["laptop", "printer", "router", "calendar app", "email client", "smart speaker"],
)


def _date(rng):
    return f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"


def make_memory2(sentences_at: dict[int, str], n_entries: int, seed: int) -> str:
    """Neutral-filler memory of n_entries lines; sentences_at maps entry index -> sentence.
    Dates are drawn and sorted for all lines (inserted lines take the date of their slot)."""
    rng = random.Random(seed)
    lines = []
    for _ in range(n_entries):
        t = rng.choice(FILLER2)
        lines.append(t.format(**{k: rng.choice(v) for k, v in FILL2_VOCAB.items()}))
    for i, s in sentences_at.items():
        lines[i] = s
    dates = sorted(_date(rng) for _ in lines)
    return "\n".join(f"- [{d}] {e}" for d, e in zip(dates, lines))


def colleague(sentence: str) -> str:
    return sentence.replace("The user", "The user's colleague", 1)
