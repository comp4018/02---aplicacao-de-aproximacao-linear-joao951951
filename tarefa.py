import numpy as np
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt


def avg_gray_scale_conversion(img_array):
    img_gray_avg = np.mean(img_array, axis=2)
    plt.imshow(img_gray_avg, cmap='gray')
    plt.axis('off')

    new_img_gray_average = np.empty(shape=img_array.shape, dtype=np.uint8)
    for i in range(3):
        new_img_gray_average[:, :, i] = img_gray_avg

    return Image.fromarray(new_img_gray_average)


def luminance_perception_conversion(img_array):
    pesos = [0.2126, 0.7152, 0.0722]

    res = np.array(img_array * pesos, dtype=np.uint8)
    lp_result = np.array(np.sum(res, axis=2), dtype=np.uint8)

    return Image.fromarray(lp_result)


def gamma_corrected_img(img_array, gamma=1.0):
    gc_result = 255.0 * (img_array / 255.0)**gamma

    return Image.fromarray(gc_result)


def gamma_expanded_img(img_array):
    # passar os pixels da escala 0-255 pra 0-1
    # avaliar o valor convertido e aplicar fórmula
    # se o valor convertido for <= 0.04045
    #   val = val / 12.92
    # se não
    #   val = ((val + 0.055) / 1.055)**2.4
    # converter o valor modificado para int e de volta para a escala 0-255
    # retornar uma imagem feita a partir dos novos pixels

    aux = np.array(img_array / 255, dtype=np.float16)

    aux = np.where(aux <= 0.04045, aux / 12.92, ((aux + 0.055)/1.055)**2.4)
    res = np.array(aux * 255, dtype=np.uint8)
    return Image.fromarray(res)


def linear_aproximated_img(img_array):
    pesos = [0.299, 0.587, 0.114]
    res = np.array(img_array * pesos, dtype=np.uint8)
    la_result = np.array(np.sum(res, axis=2), dtype=np.uint8)

    return Image.fromarray(la_result)


if _name_ == '_main_':
    url = 'https://unsplash.com/photos/boaDpmC-_Xo/download?ixid=MnwxMjA3fDB8MXxhbGx8MXx8fHx8fDJ8fDE2MzQ2ODI0MzQ&force=true&w=640'
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert()
    img_arr = np.copy(image)

    img_avg_gray = avg_gray_scale_conversion(img_arr)

    img_lp = luminance_perception_conversion(img_arr)

    img_gc = gamma_corrected_img(np.copy(img_lp), gamma=2.2) # Escolher o gama desejado

    img_ge = gamma_expanded_img(np.copy(img_lp))
    img_ge.show()

    img_la = linear_aproximated_img(img_arr)
    img_la.show()