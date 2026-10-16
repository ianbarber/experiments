"""Synthetic preference, lure and capability items for the Phase-1 gate.

PREFS: (key, memory sentence X, memory sentence Y, question, continuation X, continuation Y).
A preference is stated as a *fact about the user* (not an instruction). Shift is computed as in P0:
    d(state) = log p(cont_X | state, q) - log p(cont_Y | state, q);  shift = d(state_X) - d(state_Y)

LURES (DRM-style false memory): a list whose words are all associates of a lure word that is absent.
    Question: did the list include <lure>? Score log p(" Yes") - log p(" No").
    Conditions: associated list (lure absent), list with the lure present, neutral list.

CAPABILITY: general questions with a short gold answer; harm = drop in gold log-prob under a
memory state vs zero state.
"""

PREFS = [
    ("diet", "The user is vegetarian.", "The user loves steak and barbecue.",
     "Suggest one dinner idea for tonight.", "How about a creamy mushroom risotto?",
     "How about a grilled ribeye steak?"),
    ("holiday", "The user loves beach holidays.", "The user loves skiing holidays.",
     "Where should I go on holiday this winter?", "How about a beach break in the Canary Islands?",
     "How about a ski trip to the Alps?"),
    ("verbosity", "The user prefers very short answers.", "The user prefers long, detailed answers.",
     "How do I boil an egg?", "Boil it for nine minutes.",
     "Great question! Let me walk you through it step by step."),
    ("units", "The user always uses metric units.", "The user always uses imperial units.",
     "How long is a marathon?", "A marathon is 42.2 kilometres.", "A marathon is 26.2 miles."),
    ("music", "The user's favourite music is jazz.", "The user's favourite music is heavy metal.",
     "Recommend an album for my commute.", "Try Kind of Blue by Miles Davis.",
     "Try Master of Puppets by Metallica."),
    ("pet", "The user has a cat.", "The user has a dog.", "What gift should I buy for my pet?",
     "How about a new scratching post?", "How about a new chew toy and lead?"),
    ("reading", "The user loves science fiction.", "The user loves historical romance.",
     "Recommend a novel for the weekend.", "Try The Left Hand of Darkness by Ursula K. Le Guin.",
     "Try Outlander by Diana Gabaldon."),
    ("exercise", "The user enjoys running.", "The user enjoys swimming.",
     "Suggest a workout for tomorrow morning.", "Go for an easy 5k run.",
     "Swim thirty lengths at the pool."),
    ("budget", "The user travels on a tight budget.", "The user prefers luxury travel.",
     "Where should I stay in Paris?", "A budget hostel near Gare du Nord would suit you.",
     "The Ritz Paris would suit you."),
    ("tone", "The user likes a casual, jokey tone.", "The user likes a formal, professional tone.",
     "Write a one-line greeting for my team.", "Hey folks, happy Monday!", "Good morning, colleagues."),
    ("drink", "The user drinks tea and never coffee.", "The user drinks coffee and never tea.",
     "What should I drink to wake up?", "A strong cup of English breakfast tea.",
     "A double espresso."),
    ("spice", "The user loves very spicy food.", "The user cannot handle spicy food.",
     "Suggest a curry to order.", "Try a fiery lamb vindaloo.", "Try a mild chicken korma."),
]

# (lure, associates (lure absent), neutral list)
LURES = [
    ("sleep", "bed, rest, tired, dream, wake, snooze, blanket, doze, slumber, nap",
     "lamp, carrot, ticket, pencil, rope, violin, cloud, button, ladder, soap"),
    ("chair", "table, sit, legs, seat, couch, desk, recliner, sofa, cushion, stool",
     "river, comet, glove, piano, salt, wheel, candle, fence, kettle, shell"),
    ("doctor", "nurse, sick, lawyer, medicine, health, hospital, dentist, patient, clinic, cure",
     "tiger, paper, spoon, bridge, wind, garden, mirror, bucket, flag, apple"),
    ("music", "note, sound, piano, sing, radio, band, melody, horn, concert, orchestra",
     "stone, bread, kite, nail, forest, coat, engine, plate, rabbit, door"),
    ("cold", "hot, snow, warm, winter, ice, wet, frigid, chilly, freeze, shiver",
     "book, horse, cup, pillow, train, leaf, hammer, ring, cheese, window"),
    ("mountain", "hill, valley, climb, summit, top, molehill, peak, plain, ridge, steep",
     "shoe, onion, bottle, clock, tent, mouse, brush, sugar, card, boat"),
    ("sweet", "sour, candy, sugar, bitter, good, taste, tooth, nice, honey, chocolate",
     "chair, wire, ocean, pen, garage, wool, bell, axe, map, frog"),
    ("needle", "thread, pin, eye, sewing, sharp, point, prick, thimble, haystack, injection",
     "cloud, banana, truck, glass, desk, snow, whistle, sock, lemon, wall"),
]

CAPABILITY = [
    ("What is the capital of France?", "The capital of France is", " Paris"),
    ("What is the capital of Japan?", "The capital of Japan is", " Tokyo"),
    ("What is the capital of Canada?", "The capital of Canada is", " Ottawa"),
    ("What is the capital of Australia?", "The capital of Australia is", " Canberra"),
    ("What is the capital of Kenya?", "The capital of Kenya is", " Nairobi"),
    ("What is 17 + 25?", "17 + 25 =", " 42"),
    ("What is 9 times 8?", "9 times 8 is", " 72"),
    ("What is 144 divided by 12?", "144 divided by 12 is", " 12"),
    ("What is the square root of 81?", "The square root of 81 is", " 9"),
    ("How many days are in a leap year?", "A leap year has", " 366"),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is", " Au"),
    ("What is the chemical symbol for sodium?", "The chemical symbol for sodium is", " Na"),
    ("What gas do plants absorb from the air?", "Plants absorb", " carbon dioxide"),
    ("What is the largest planet in the solar system?", "The largest planet is", " Jupiter"),
    ("Which planet is closest to the Sun?", "The planet closest to the Sun is", " Mercury"),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by", " Jane Austen"),
    ("Who wrote Hamlet?", "Hamlet was written by", " William Shakespeare"),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by", " Leonardo da Vinci"),
    ("What is the boiling point of water in Celsius?", "Water boils at", " 100"),
    ("How many continents are there?", "There are", " seven"),
    ("What is the longest river in Africa?", "The longest river in Africa is the", " Nile"),
    ("What language is spoken in Brazil?", "The main language of Brazil is", " Portuguese"),
    ("What is the freezing point of water in Fahrenheit?", "Water freezes at", " 32"),
    ("How many sides does a hexagon have?", "A hexagon has", " six"),
    ("What is the hardest natural substance?", "The hardest natural substance is", " diamond"),
    ("What is the currency of the United Kingdom?", "The currency of the UK is the", " pound"),
    ("Which ocean is the largest?", "The largest ocean is the", " Pacific"),
    ("In which year did World War II end?", "World War II ended in", " 1945"),
    ("What organ pumps blood around the body?", "Blood is pumped by the", " heart"),
    ("What is the smallest prime number?", "The smallest prime number is", " 2"),
    ("What is H2O commonly called?", "H2O is commonly called", " water"),
    ("Which animal is known as the king of the jungle?", "The king of the jungle is the", " lion"),
    ("What is the tallest mountain on Earth?", "The tallest mountain on Earth is", " Mount Everest"),
    ("Who developed the theory of relativity?", "The theory of relativity was developed by", " Albert Einstein"),
    ("What is the capital of Italy?", "The capital of Italy is", " Rome"),
    ("What is 15 squared?", "15 squared is", " 225"),
    ("How many minutes are in an hour?", "An hour has", " 60"),
    ("What colour do you get by mixing blue and yellow?", "Mixing blue and yellow gives", " green"),
    ("What is the main ingredient of guacamole?", "The main ingredient of guacamole is", " avocado"),
    ("Which country is home to the kangaroo?", "Kangaroos are native to", " Australia"),
]

# One plausible same-type wrong answer per CAPABILITY item (aligned by index), for the guard's
# gold-vs-distractor log-odds margin.
CAPABILITY_DISTRACTORS = [
    " Lyon", " Osaka", " Toronto", " Sydney", " Mombasa", " 41", " 63", " 14", " 8", " 365",
    " Ag", " K", " oxygen", " Saturn", " Venus", " Charlotte Bronte", " Christopher Marlowe",
    " Michelangelo", " 90", " six", " Congo", " Spanish", " 0", " eight", " granite", " euro",
    " Atlantic", " 1944", " lungs", " 1", " ice", " tiger", " K2", " Isaac Newton", " Milan",
    " 125", " 100", " purple", " tomato", " New Zealand",
]
assert len(CAPABILITY_DISTRACTORS) == len(CAPABILITY)
