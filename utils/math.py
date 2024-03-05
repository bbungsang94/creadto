import copy
import torch
import numpy as np


def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret


def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
    zeros43 = torch.zeros(
        (x.shape[0], x.shape[1], 4, 3), dtype=torch.float64).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret


def l2_distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum(2)).mean(1).mean()


def is_in_polygon(vertices, point, bypass_in_box=False):
    """
    source: http://web.archive.org/web/20080812141848/http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
    :param vertices: a tensor of points shaped (N, xy)
    :param point: xy point
    :param bypass_in_box: If True, not work checking min-max boundary
    :return: boolean
    """
    polygon = vertices.shape[0]
    a = vertices[0]
    vertices = torch.cat([vertices, copy.deepcopy(a.unsqueeze(dim=0))], dim=0)
    count = 0
    for i in range(1, polygon + 1):
        b = vertices[i]
        if bypass_in_box:
            bound_condition = True
        else:
            bound_condition = min(a[1], b[1]) < point[1] <= max(a[1], b[1]) and point[0] <= max(a[0], b[0])
        # feasible_condition = (a[1] != b[1] or a[0] != b[0])
        feasible_condition = a[1] != b[1]

        # xinters = -torch.inf
        if bound_condition and feasible_condition:
            xinters = (b[0] - a[0]) / (b[1] - a[1]) * (point[1] - a[1]) + a[0]
            if a[0] == b[0] or point[0] <= xinters:
                count += 1
        a = copy.deepcopy(b)

    return count % 2 == 1


def plot_polygon(vertices):
    from torchvision.transforms import ToPILImage
    """
    특정 파장의 이미지에서 다각형 내부에 있는 점들만 빨간색으로 표시해서 plot
    이 함수는 시각화 용이고 계산에는 사용되지 않음
    """
    to_pil = ToPILImage()
    max_value, _ = vertices.max(dim=0)
    min_value, _ = vertices.min(dim=0)
    gap_value = max_value - min_value
    board = torch.zeros((3, gap_value[1] + 1, gap_value[0] + 1))

    for row in range(0, board.shape[1]):
        for col in range(0, board.shape[2]):
            point = [col + min_value[0], row + min_value[1]]
            res = is_in_polygon(vertices, point)
            if res:
                board[1, row, col] = 0.7

    for v in vertices:
        c = v[0] - min_value[0]
        r = v[1] - min_value[1]
        board[0, r, c] = 1.0
        board[1, r, c] = 0.0
        board[2, r, c] = 1.0

    pil_img = to_pil(board)
    pil_img.save("polygon-test-visualization.jpg")
