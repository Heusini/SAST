import os


events = os.listdir("./events/")
labels = os.listdir("./labels/")
rgbs = os.listdir("./rgbs/")

events.sort(key=lambda item: (len(item), item))
labels.sort(key=lambda item: (len(item), item))
rgbs.sort(key=lambda item: (len(item), item))

skip = 10
count = 0
for e, l, r in zip(events, labels, rgbs):
    if count < skip:
        count += 1
    else:
        ev_path = f"./events/{e}"
        lb_path = f"./labels/{l}"
        rg_path = f"./rgbs/{r}"
        os.remove(ev_path)
        os.remove(lb_path)
        os.remove(rg_path)


