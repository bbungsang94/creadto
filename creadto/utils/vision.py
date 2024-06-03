import cv2


def remove_light(image_bgr, blur=75):
    # 1) RGB to LAB
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # 2) LAB with median 25 to 100 and bilateral
    image_mf = cv2.medianBlur(l_channel, blur)
    inverted_image = cv2.bitwise_not(image_mf)

    # add-weighted가 아닌 add를 하면 제일 높은 블럭을 찾을 수 있다.
    image_composite = cv2.addWeighted(l_channel, 0.5, inverted_image, 0.5, 0)

    # Light removal 완료
    remove_lab = cv2.merge([image_composite, a_channel, b_channel])
    remove_bgr = cv2.cvtColor(remove_lab, cv2.COLOR_LAB2BGR)
    return remove_bgr