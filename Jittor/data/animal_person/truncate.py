from tqdm import tqdm
cnt = 0


with open("wiki_animal.json", "r") as file:
    for line in tqdm(file):
        if cnt < 10000:
            with open("animal_train.json", "a") as writeto:
                writeto.write(line)
        elif cnt < 11000:
            with open("animal_valid.json", "a") as writeto:
                writeto.write(line)
        elif cnt < 12000:
            with open("animal_test.json", "a") as writeto:
                writeto.write(line)
        else:
            break
        cnt += 1