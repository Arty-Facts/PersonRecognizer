from lib.Locate_ppl import Locate_ppl
from lib.Recognizer import Recognizer
from lib.PersonEmbeding import PersonEmbeding
from lib.EmbedingsManiger import EmbedingsManiger
from matplotlib import pyplot as plt
from pathlib import Path
import torch

def take_pic(locate_ppl):
    for found_ppl in locate_ppl:
        if len(found_ppl) > 0:
            for person in found_ppl:
                plt.imshow(person)
                plt.show()
                ynans = input("Är du nöjd med din profilbild? [Y/n]")
                if ynans in ["YES", "Yes", "yes", "Y", "y", ""]:
                    return person


def main(max_counter = 10, save_img=True):
    locate_ppl = Locate_ppl(save_img=save_img)
    person_embeding = PersonEmbeding()
    data_base = EmbedingsManiger()


    print("Hej! Vem ska registreras?")
    namn = input(": ")
    print(data_base.info)
    if namn not in data_base.info:
        print("Välkommen", namn, ".Bilder tas av dig nu.")
    else:
        print("Välkommen tillbaka",namn, ". \nVill du uppdatera din databas? [Y/n]")
        ynans = input(": ")
        if ynans in ["YES", "Yes", "yes", "Y", "y", ""]:
            print("Bilder tas av dig nu.")
        else:
            print("Hej då")
            return
    locate_ppl.set_path(f"images/{namn}")
    #bilder tas
    counter = 0
    feature_vectors = []
    for found_ppl in locate_ppl:
        if len(found_ppl) == 1:
            counter += 1
            print("Bild", counter, "/", max_counter, "är tagna.")
            embs = person_embeding.gen_training_emb(found_ppl)
            for emb in embs:
                feature_vectors.append(emb)
        elif len(found_ppl) > 1:
            print("Hittade", len(found_ppl), "personer i bilden. Se till så att det bara är en person i bilden.")

        if counter >= max_counter:
            break

    if namn not in data_base.info:
        image = take_pic(locate_ppl)
        if data_base.new(namn,image,feature_vectors):
            print("Din användare är sparad.")
        else:
            print ("Något blev fel. Din användare är inte sparad.")
    else:
        if data_base.add(namn,feature_vectors):
            print("Din användare är uppdaterad.")
        else:
            print ("Något blev fel. Din användare är inte uppdaterad.")



        



    



if __name__ == "__main__":
    main()