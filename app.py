from lib.Locate_ppl import Locate_ppl
from lib.Recognizer import Recognizer
from lib.PersonEmbeding import PersonEmbeding
from matplotlib import pyplot as plt
from pathlib import Path
import torch
def main():
    lp = Locate_ppl()
    rec = Recognizer()
    pe = PersonEmbeding()
    db = Path("db")
    models = Path("models")
    for obj in [db, models]:
        if not obj.is_dir():
            obj.mkdir()
    



if __name__ == "__main__":
    main()