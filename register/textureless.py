

def find_textureless(images, pairs):
    degree = dict.fromkeys(images, 0)
    for p in pairs:
        degree[p["i"]] += 1
        degree[p["j"]] += 1
    return sorted(n for n, d in degree.items() if d < 2)


def place_textureless(images, pairs, slot_info, deadline):
    return slot_info
