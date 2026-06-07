import numpy as np 
import cv2


def histogram_equal(img):
    # Chuyen sang khong gian hsv
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(lab_img)
    # Can bang do sang v
    l_equal = cv2.equalizeHist(l)
    lab_equal = cv2.merge([l_equal,a,b])

    resuilt = cv2.cvtColor(lab_equal, cv2.COLOR_LAB2BGR)
    return resuilt, l, l_equal

def gamma_correction(img, gamma = 1.2):
    inv = 1.0/gamma
    table = np.array([(i/255)**inv *255 for i in range(256)]).astype('uint8')
    return cv2.LUT(img, table)


def light_dehaze(img, strength = 0.7):
    # Tao anh blur
    blur = cv2.GaussianBlur(img, (0,0), 3)
    # img + strength *(img-blur) => lam noi bat chi tiet trong anh
    res = cv2.addWeighted(img, 1+ strength,blur, -strength,0)
    return np.clip(res, 0, 255).astype(np.uint8)

def enhance_by_clahe(img, limit = 2.0, kernel = (8,8)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = kernel)
    l_eq = clahe.apply(l)
    
    lab_eq = cv2.merge([l_eq,a,b])
    resuilt = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return resuilt, l, l_eq

def get_ksize(sigma):
    # Cong thuc noi suy nguoc cua OpenCV
    #  sigma = 0.3 * ((ksize -1)* 0.5 -1)+0.8
    ksize = int(((sigma - 0.8)/0.15) + 2.0)
    return max(3, ksize if ksize % 2 == 1 else ksize + 1)

def get_gaussian_blur(img, ksize= 0, sigma =5):
    if ksize == 0:
        ksize = get_ksize(sigma)
    sep_k = cv2.getGaussianKernel(ksize,sigma)
    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))

def ssr(img, sigma, apply_normalization = True):
    # SSR(x, y) = log(I(x, y)) - log(I(x, y)*F(x, y))
    # F la gaussian
    img = img.astype(np.float32) + 1.0
    ssr = np.log10(img) - np.log10(get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)
    if apply_normalization:
        ssr = cv2.normalize(ssr, None,0,255, cv2.NORM_MINMAX).astype(np.uint8)
    return ssr

def msr(img, sigma_scales = [15,80,250], apply_normalization = True):
    img = img.astype(np.float32)
    msr = np.zeros(img.shape, dtype=np.float32)
    
    # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales
    for sigma in sigma_scales:
        msr += ssr(img, sigma, apply_normalization= False)

    msr /= len(sigma_scales)

    if apply_normalization:
        msr = cv2.normalize(msr, None,0,255, cv2.NORM_MINMAX).astype(np.uint8)
    return msr

def color_balance(img, low_per, high_per):
    sum_pixel = img.shape[0] * img.shape[1]

    # Tinh nguong cat 
    low_count = sum_pixel * low_per/100
    high_count = sum_pixel * (100 - high_per) /100

    chanel_list = []
    if len(img.shape) == 2:
        chanel_list = [img]
    else:
        chanel_list = cv2.split(img)

    cs_img = []
    for i in range(len(chanel_list)):
        chanel = chanel_list[i]
        hist = cv2.calcHist([chanel],[0], None, [256], (0,256))
        # Bao nhiêu pixel ≤ giá trị i
        cum_hist_sum = np.cumsum(hist)

        # Tim (li,hi) khoang pixel duoc giua lai
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if(li == hi):
            cs_img.append(chanel)
            continue
        lut = np.array([
            0 if i < li
            else (255 if i> hi
            else round((i - li)/(hi -li)*255))
            for i in range(0,256)
        ], dtype = 'uint8')

        cs_ch = cv2.LUT(chanel, lut)
        cs_img.append(cs_ch)
    if (len(cs_img)==1):
        return np.squeeze(cs_img)
    elif (len(cs_img) >1):
        return cv2.merge(cs_img)
    else:
        return None

def msrcr(img,sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1 ):
    # MSRCR(x,y) = G * [MSR(x,y)*CRF(x,y) - b], G=gain and b=offset
    # CRF(x,y) = beta*[log(alpha*I(x,y) - log(I'(x,y))]
    # I'(x,y) = sum(Ic(x,y)), c={0...k-1}, k=no.of channels

    img = img.astype(np.float64)
    img_safe = img + 1.0
    msr_img = msr(img,sigma_scales= sigma_scales, apply_normalization = False)
    crf = beta * (np.log10(alpha * img_safe) - np.log10(np.sum(img_safe,axis = 2, keepdims = True)))
    msrcr_img = G * (msr_img * crf - b)
    msrcr_img = cv2.normalize(msrcr_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    msrcr_img = color_balance(msrcr_img, low_per, high_per)
    return msrcr_img

def msrcp(img, sigma_scales=[15, 80, 250], low_per=1, high_per=1, radius=15, eps=1e-3):
    # Int(x,y) = sum(Ic(x,y))/3, c={0...k-1}, k=no.of channels
    # MSR_Int(x,y) = MSR(Int(x,y)), and apply color balance
    # B(x,y) = MAX_VALUE/max(Ic(x,y))
    # A(x,y) = min(B(x,y), MSR_Int(x,y)/Int(x,y))
    # MSRCP = A*I

    img = img.astype(np.float32)
    if len(img.shape) == 2:
        INT = img + 1.0
    else:
        r,g,b = cv2.split(img)
        INT = (r + g + b)/3 + 1.0
    msr_img= msr(INT,sigma_scales, apply_normalization = False)
    # msr_img = cv2.ximgproc.guidedFilter(
    #     guide=INT.astype(np.float32),
    #     src=msr_img.astype(np.float32),
    #     radius=radius,
    #     eps=eps
    # )
    msr_img = cv2.normalize(msr_img, None, 0, 255, cv2.NORM_MINMAX)
    msr_img = msr_img.astype(np.uint8)
    msr_cb = color_balance(msr_img, low_per, high_per).astype(np.float32)
    B = 256.0 / (np.max(img, axis = 2) +1.0)
    # scale theo retinex
    ratio =  msr_cb / INT
    A = np.minimum(B, ratio)
    msrcp_img = np.clip(np.expand_dims(A,2) * img , 0.0, 255.0)
    return msrcp_img.astype(np.uint8)


def enhance_by_retinex(img, method = 'msr', sigma = 80):
    if method == 'msr':
        img_eq = msr(img.astype(np.float32))
    elif method == 'ssr':
        img_eq = ssr(img.astype(np.float32), sigma = sigma)
    elif method == 'msrcr':
        img_eq = msrcr(img)
    else:
        img_eq = msrcp(img)
    return img_eq
