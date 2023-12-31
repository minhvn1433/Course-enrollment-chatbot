import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(86)

quarter_pattern = ["fall", "autumn", "iap", "winter", "spring", "summer"]
year_pattern = [
    "2023",
    "2024",
    "2025",
    "23",
    "24",
    "25",
    "2k23",
    "2k24",
    "2k25",
    "2023-2024",
    "2024-2025",
    "23-24",
    "24-25",
]
code_pattern = [
    "1.",
    "2.",
    "3.",
    "4.",
    "5.",
    "6.",
    "7.",
    "8.",
    "9.",
    "10.",
    "11.",
    "12.",
    "14.",
    "15.",
    "16.",
    "17.",
    "18.",
    "20.",
    "21.",
    "21a.",
    "21h.",
    "21g.",
    "21l.",
    "21m.",
    "21w.",
    "22.",
    "24.",
    "as.",
    "cc.",
    "cms.",
    "csb.",
    "cse.",
    "ec.",
    "em.",
    "es.",
    "hst.",
    "ids.",
    "mas.",
    "ms.",
    "ns.",
    "scm.",
    "sp.",
    "sts.",
    "wgs.",
]
level_pattern = [
    "undergraduate",
    "graduate",
    "undergrad",
    "grad",
    "undergrads",
    "grads",
    "graduates",
    "undergraduates",
    "ugrad",
    "ugrads",
    "undergraduate-level",
    "graduate-level",
    "undergrad-level",
    "grad-level",
    "u",
    "g",
]
name_pattern = [
    "civil and environmental engineering",
    "mechanical engineering",
    "materials science and engineering",
    "architecture",
    "chemistry",
    "electrical engineering and computer science",
    "biology",
    "physics",
    "brain and cognitive sciences",
    "chemical engineering",
    "urban studies and planning",
    "earth, atmospheric, and planetary sciences",
    "economics",
    "management",
    "aeronautics and astronautics",
    "political science",
    "mathematics",
    "biological engineering",
    "humanities",
    "anthropology",
    "history",
    "global languages",
    "literature",
    "music and theater arts",
    "writing",
    "nuclear science and engineering",
    "linguistics and philosophy",
    "aerospace studies",
    "concourse",
    "comparative media studies",
    "computational and systems biology",
    "computational science and engineering",
    "edgerton center",
    "engineering management",
    "experimental study group",
    "health sciences and technology",
    "data, systems, and society",
    "media arts and sciences",
    "military science",
    "naval science",
    "supply chain management",
    "special programs",
    "science, technology, and society",
    "women's and gender studies",
]

courses_df = pd.read_csv("courses.csv")


def transform_term(term):
    term = term.lower()
    term = "".join([char for char in term if char not in "():;,"])
    term = " ".join(
        [
            word
            for word in term.split()
            if word
            not in ("acad", "year", "first", "second", "half", "of", "term", "partial")
        ]
    )
    term = term.replace("2023-2024 not offered", "")
    term = term.replace("2024-2025 not offered", "")
    term = term.replace("not offered regularly consult department", "department")
    term = term.strip()
    if "2023-2024" not in term and "2024-2025" not in term:
        term += " 2023-2024 2024-2025"
    return term


courses_df["Terms"] = courses_df["Terms"].apply(transform_term)
original_df = pd.read_csv("courses.csv")
sentences_df = pd.read_csv("sentences.csv")
device = torch.device("cpu")
labels = ["enroll", "search", "inform"]

label_encoder = LabelEncoder()
label_encoder.fit_transform(labels)

sentences = sentences_df["Sentence"].tolist()
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)


class IntentClassifier(nn.Module):
    def __init__(self):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(1000, 16)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 30, 6)
        self.fc2 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        flatten = self.flatten(embedded)
        out = self.relu(self.fc1(flatten))
        out = self.fc2(out)
        return out


model = IntentClassifier().to(device)
model.load_state_dict(torch.load(f"intent.pth"))
model.eval()


def identify_intent(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=30, padding="post", truncating="post")
    inputs = torch.tensor(padded).to(device)

    outputs = model(inputs)
    pred = torch.argmax(outputs, 1).item()
    intent = label_encoder.inverse_transform([pred])[0]

    return intent


def process_year_quarter(year, quarter):
    if "k" in year:
        year = year.replace("k", "0")
    if len(year) == 2 and quarter == "Fall":
        year = int("20" + year)
        year = f"{year}-{year + 1}"
    elif len(year) == 2 and quarter != "Fall":
        year = int("20" + year)
        year = f"{year - 1}-{year}"
    elif len(year) == 4 and quarter == "Fall":
        year = f"{year}-{int(year) + 1}"
    elif len(year) == 4 and quarter != "Fall":
        year = f"{int(year) - 1}-{year}"
    elif year == "23-24":
        year = "2023-2024"
    elif year == "24-25":
        year = "2024-2025"
    if year != "2023-2024" and year != "2024-2025":
        year = None

    return year, quarter


def identify_entities(sentence):
    sentence = sentence.lower()
    words = sentence.split()

    quarters = [word for word in words if word.startswith(tuple(quarter_pattern))]
    if quarters:
        quarter = quarters[0]
        quarter = "".join([char for char in quarter if char not in "():;,?."])
        if quarter == "fall" or quarter == "autumn":
            quarter = "Fall"
        elif quarter == "iap" or quarter == "winter":
            quarter = "IAP"
        elif quarter == "spring":
            quarter = "Spring"
        elif quarter == "summer":
            quarter = "Summer"
    else:
        quarter = None

    years = [word for word in words if word.startswith(tuple(year_pattern))]
    if years:
        year = years[0]
        year = "".join([char for char in year if char not in "():;,?."])
    else:
        year = None

    codes = [word for word in words if word.startswith(tuple(code_pattern))]
    if codes:
        code = codes[0].upper()
        code = "".join([char for char in code if char not in "():;,?"])
        code = code[:-1] if code.endswith(".") else code
    else:
        code = None

    levels = [word for word in words if word.startswith(tuple(level_pattern))]
    if levels:
        level = levels[0]
        if level.startswith("u"):
            level = "undergraduate"
        elif level.startswith("g"):
            level = "graduate"
    else:
        level = None

    names = [pattern for pattern in name_pattern if pattern in sentence]
    if names:
        name = names[0]
        name = "".join([char for char in name if char not in "():;,?."])
    else:
        name = None

    return (quarter, year, code, level, name)


def generate_response(intent, entities):
    quarter, year, code, level, name = entities
    context, slots = None, (None, None, None, None, None)
    if year is not None:
        year, quarter = process_year_quarter(year, quarter)

    if intent == "enroll":
        if code and quarter and year:
            courses = courses_df[
                (courses_df["Code"] == code)
                & courses_df["Terms"].str.contains(quarter, case=False)
                & courses_df["Terms"].str.contains(year)
            ]
            department = courses["Terms"].str.contains("department")
            if len(courses) != 0 and not department.any():
                print(
                    f"ChatGPT: For sure, I just enrolled you in {code} for {quarter} {year}."
                )
            elif len(courses) != 0 and department.any():
                print(
                    f"ChatGPT: Sorry I don't have permission to do this. Please consult department."
                )
            else:
                print("ChatGPT: Sorry I can't find this class.")
        elif code is None:
            print("ChatGPT: Which class do you want to enroll in?")
            context, slots = intent, entities
        elif quarter is None:
            print(f"ChatGPT: For which quarter?")
            context, slots = intent, entities
        elif year is None:
            print(f"ChatGPT: For which year?")
            context, slots = intent, entities

    elif intent == "search":
        if name:
            if level and quarter and year:
                print(
                    f"ChatGPT: Here's the list of {level} {name} courses offered in {quarter} {year}:"
                )
                courses = courses_df[
                    courses_df["Code"].str.startswith(
                        code_pattern[name_pattern.index(name)]
                    )
                    & courses_df["Terms"].str.contains(rf"\b[{level[0]}]\b")
                    & courses_df["Terms"].str.contains(quarter, case=False)
                    & courses_df["Terms"].str.contains(year)
                ]
                for index, row in courses.iterrows():
                    print("        ", row["Code"], row["Title"])
            elif level and quarter:
                print(
                    f"ChatGPT: Here's the list of {level} {name} courses offered in {quarter}:"
                )
                courses = courses_df[
                    courses_df["Code"].str.startswith(
                        code_pattern[name_pattern.index(name)]
                    )
                    & courses_df["Terms"].str.contains(rf"\b[{level[0]}]\b")
                    & courses_df["Terms"].str.contains(quarter, case=False)
                ]
                for index, row in courses.iterrows():
                    print("        ", row["Code"], row["Title"])
            elif level and year:
                print(
                    f"ChatGPT: Here's the list of {level} {name} courses offered in {year}:"
                )
                courses = courses_df[
                    courses_df["Code"].str.startswith(
                        code_pattern[name_pattern.index(name)]
                    )
                    & courses_df["Terms"].str.contains(rf"\b[{level[0]}]\b")
                    & courses_df["Terms"].str.contains(year)
                ]
                for index, row in courses.iterrows():
                    print("        ", row["Code"], row["Title"])
            elif quarter and year:
                print(
                    f"ChatGPT: Here's the list of {name} courses offered in {quarter} {year}:"
                )
                courses = courses_df[
                    courses_df["Code"].str.startswith(
                        code_pattern[name_pattern.index(name)]
                    )
                    & courses_df["Terms"].str.contains(quarter, case=False)
                    & courses_df["Terms"].str.contains(year)
                ]
                for index, row in courses.iterrows():
                    print("        ", row["Code"], row["Title"])
            elif level:
                print(f"ChatGPT: Here's the list of {level} {name} courses:")
                courses = courses_df[
                    courses_df["Code"].str.startswith(
                        code_pattern[name_pattern.index(name)]
                    )
                    & courses_df["Terms"].str.contains(rf"\b[{level[0]}]\b")
                ]
                for index, row in courses.iterrows():
                    print("        ", row["Code"], row["Title"])
            elif quarter:
                print(
                    f"ChatGPT: Here's the list of {name} courses offered in {quarter}:"
                )
                courses = courses_df[
                    courses_df["Code"].str.startswith(
                        code_pattern[name_pattern.index(name)]
                    )
                    & courses_df["Terms"].str.contains(quarter, case=False)
                ]
                for index, row in courses.iterrows():
                    print("        ", row["Code"], row["Title"])
            elif year:
                print(f"ChatGPT: Here's the list of {name} courses offered in {year}:")
                courses = courses_df[
                    courses_df["Code"].str.startswith(
                        code_pattern[name_pattern.index(name)]
                    )
                    & courses_df["Terms"].str.contains(year)
                ]
                for index, row in courses.iterrows():
                    print("        ", row["Code"], row["Title"])
            else:
                print(f"ChatGPT: Here's the list of {name} courses:")
                courses = courses_df[
                    courses_df["Code"].str.startswith(
                        code_pattern[name_pattern.index(name)]
                    )
                ]
                for index, row in courses.iterrows():
                    print("        ", row["Code"], row["Title"])
        elif name is None:
            print("ChatGPT: Which type of course are you looking for?")
            context, slots = intent, entities

    elif intent == "inform":
        courses = courses_df[courses_df["Code"] == code]
        if len(courses) != 0:
            for index, row in courses.iterrows():
                row = original_df.loc[index]
                print("ChatGPT: Title        :", row["Code"], row["Title"])
                print("         Cluster      :", row["Cluster"])
                print("         Prerequisites:", row["Prerequisites"])
                print("         Terms        :", row["Terms"])
                print("         Hours        :", row["Hours"])
                print("         Optional     :", row["Optional"])
                print("         Description  :", row["Description"])
                print("         Instructors  :", row["Instructors"])
        else:
            print("ChatGPT: Could you give me the specific course code?")
            context, slots = intent, entities

    return context, slots


def greet():
    print("ChatGPT: Hello! How can I assist you today? Type 'help' for more info")


def bye():
    print("ChatGPT: Bai bai 😊👋")


def help():
    print(
        """         1. Commands:

         "bye" - End conversation
         "help" - Display manual
         "list courses" - Show courses list

         2. Conversation Flow:

         a. Enroll a course:
         Example: "I want to enroll in 6.4100 for Fall 2024 to learn AI."

         b. Search for courses:
         Example: "What are the undergraduate physics classes offered in Spring 2024?"
         Note: only support course name in "list courses"

         c. Show detailed course information:
         Example: "Can you provide detail information of course 6.4100?"
    """
    )


def list_courses():
    print(
        "ChatGPT: Below is a list of the departments and programs that offer subjects at MIT:"
    )
    for course in name_pattern:
        print("         - ", end="")
        print(course)


def main():
    greet()
    context = None
    slots = (None, None, None, None, None)
    while True:
        user_input = input("You    : ")
        if user_input == "bye":
            bye()
            break
        elif user_input == "help":
            help()
            continue
        elif user_input == "list courses":
            list_courses()
            continue

        entities = identify_entities(user_input)
        entities = tuple(
            entity if entity is not None else slot
            for entity, slot in zip(entities, slots)
        )
        if context is None:
            intent = identify_intent(user_input)
        else:
            intent = context
        context, slots = generate_response(intent, entities)


if __name__ == "__main__":
    main()
