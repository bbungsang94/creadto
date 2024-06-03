import json
import os

import torch


def to_npy():
    pin_root = r"D:\Creadto\creadto\dump\data"
    interactions = [
        ("머리둘레(1)", "Head Circumference(1)", ["Galbella", "Euryon, Right"], "length-h", "standard"),
        ("머리둘레(2)", "Head Circumference(2)", ["Euryon, Right", "Occiput"], "length-h", "standard"),
        ("머리둘레(3)", "Head Circumference(3)", ["Occiput", "Euryon, Left"], "length-h", "standard"),
        ("머리둘레(4)", "Head Circumference(4)", ["Euryon, Left", "Galbella"], "length-h", "standard"),
        ("머리옆호길이(1)", "Bitragion Arc(1)", ["Tragion, Right", "Occipital bone"], "length", "standard"),
        ("머리옆호길이(2)", "Bitragion Arc(2)", ["Occipital bone", "Tragion, Left"], "length", "standard"),
        ("머리앞호길이(1)", "Sagittal Arc(1)", ["Galbella", "Occipital bone"], "length-v", "standard"),
        ("머리앞호길이(2)", "Sagittal Arc(2)", ["Occipital bone", "Occiput"], "length-v", "standard"),
        ("머리앞호길이(3)", "Sagittal Arc(3)", ["Occiput", "Inion"], "length-v", "standard"),
        ("머리두께", "Head Depth", ["Galbella", "Occiput"], "depth", "standard"),
        ("머리너비", "Head Breadth", ["Euryon, Right", "Euryon, Left"], "width", "standard"),
        ("귀구슬너비", "Tragion Breadth", ["Tragion, Right", "Tragion, Left"], "width", "standard"),
        ("머리수직길이", "Head Height", ["Occipital bone", "Menton"], "height", "standard"),
        ("얼굴수직길이", "Face Height", ["Sellion", "Menton"], "height", "standard"),
        ("귀높이위치, 좌", "Ear Pos Height, Left", ["Occipital bone", "Pre-Auricular, Left"], "height", "standard"),
        ("귀높이위치, 우", "Ear Pos Height, Right", ["Occipital bone", "Pre-Auricular, Right"], "height", "standard"),
        ("귀깊이위치, 좌", "Ear Pos Width, Left", ["Occiput", "Mastoiditis, Left"], "depth", "standard"),
        ("귀깊이위치, 우", "Ear Pos Width, Right", ["Occiput", "Mastoiditis, Right"], "depth", "standard"),
        ("귀길이, 좌", "Ear Length, Left", ["Pre-Auricular, Left", "Mastoiditis, Left"], "depth", "standard"),
        ("귀길이, 우", "Ear Length, Right", ["Pre-Auricular, Right", "Mastoiditis, Right"], "depth", "standard"),
        ("턱길이, 좌", "Jaw Length, Left", ["Tragion, Left", "Menton"], "length", "standard"),
        ("턱길이, 우", "Jaw Length, Right", ["Tragion, Right", "Menton"], "length", "standard"),
    ]

    with open(os.path.join(pin_root, 'facial.json'), 'r', encoding='UTF-8-sig') as f:
        facial = json.load(f)
    with open(os.path.join(pin_root, 'circumference-facial.json'), 'r', encoding='UTF-8-sig') as f:
        circ_dict = json.load(f)

    save_file = {
        'interactions': interactions,
        'facial landmarks': facial,
        'circumference dict': circ_dict
    }
    torch.save(save_file, 'dimension_guide.pt')
    pass


if __name__ == "__main__":
    to_npy()
