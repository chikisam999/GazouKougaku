import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import cmath
from PIL import Image
from matplotlib import cm


# 画像データの読み込み
img = Image.open('./sig_with_noise.png')
img_raw = img.convert('L')
# ピクセルごとに白黒の強度の数値データとして二次元配列に格納する。img_xy
img_raw_xy = np.asarray(img_raw)
# 画像サイズを定義する。今回は128×128なのでNxとNyは128である。
Nx = 128
Ny = 128
# 円周率を定義
PI = cmath.pi

# No.1 原画像からフーリエ変換を計算する
# 2次元フーリエ級数展開を行う。課題3の手順を参考に行う。
# 以下のforループを用いた演算はかなり時間がかかるため、コメントアウトした。

"""
# 128×128の空の行列を作成
x = np.zeros((Nx, Ny))
y = np.zeros((Nx, Ny))
# 変換後の値を格納する配列を生成する。
f_uv = np.zeros((Nx, Ny), dtype=np.complex)

for k in range(0, 128):
    for l in range(0, 128):
        x[k, l] = k * 2 * math.pi / 128 - math.pi
        y[k, l] = l * 2 * math.pi / 128 - math.pi
        for n in range(0, 128):
            for m in range(0, 128):
                f_uv[k, l] += img_raw_xy[m, n] * cmath.exp(-1j * (2.0 * math.pi * k * m) / 128) * cmath.exp(-1j * (2.0 * math.pi * l * n) / 128)
        f_uv[k, l] = f_uv[k, l] / (Nx * Ny)

# フーリエ変換後のデータの折かえし
shifted_func = np.zeros((Nx, Ny), dtype=np.complex)
shifted_func[64:, 64:] = f_uv[0:64, 0:64]
shifted_func[0:64, 0:64] = f_uv[64:, 64:]
shifted_func[0:64, 64:] = f_uv[64:, 0:64]
shifted_func[64:, 0:64] = f_uv[0:64:, 64]


#対数を取って最大値で規格化を行う
logged = np.log(np.abs(shifted_func) + 1.0)
logged = (logged.real / np.max(logged)) * 255

#パワースペクトル画像を生成する。
fig, ax = plt.subplots()
ax.imshow(logged, cmap="gray", extent=[-math.pi, math.pi, math.pi, -math.pi])
ax.set_title('2D Fourier transform')
ax.set_xlabel('x-spatial frequency [rad]')
ax.set_ylabel('y-spatial frequency [rad]')
plt.show()

# 以下は手動によるシフトを実行するコード（課題3と同等）
kx_shifted = np.zeros((Nx, Ny))
ky_shifted = np.zeros((Nx, Ny))
kx_shifted = kx - cmath.pi
ky_shifted = ky - cmath.pi
f_shifted = np.zeros( (Nx, Ny) , dtype = np.complex)
f_shifted[64:, 64:] = f_uv[0:64, 0:64]
f_shifted[0:64, 0:64] = f_uv[64:, 64:]
f_shifted[0:64, 64:] = f_uv[64:, 0:64]
f_shifted[64:, 0:64] = f_uv[0:64:, 64:]
mag = np.log(np.abs(f_shifted)+1.0)
mag = (mag.real/np.max(mag)) * 255
"""

# for文を用いた演算には非常に時間がかかるため、以下はライブラリを用いたFFTを実行する。
# ライブラリを用いて画像中心に周波数の原点が来るようにシフトする。
f_uv = np.fft.fft2(img_raw_xy / (Nx * Ny) )
fshift = np.fft.fftshift(f_uv)
magnitude_spectrum = 255 * np.log(np.abs(fshift)**2)

"""
# パワースペクトルのy=0かつx方向成分だけ描画してみる。
# これより、f_uv[0]の配列の32番目と、96番目のピクセルがピークを持つことがわかる。
# つまり、パワースペクトル画像中の(32,0)と(96,0)にピークを持つことがわかる。
print(f_uv[63])
x = []
for i in range(128):
    x.append(float(i))
fig3 = plt.scatter(x, np.abs(f_uv[0])**2)
plt.show()
print(f_uv[0][32], f_uv[0][96])
print(np.abs(f_uv[0][32]))
"""

# 周期成分の周波数成分の二点に対してフィルターをかける。0だと欠損ピクセルとして扱われるため、周囲同等のノイズで埋め合わせる。、
f_uv2 = f_uv
f_uv2[0][32] = 0.000001 + 0.000001j
f_uv2[0][96] = 0.000001 + 0.000001j

# フィルタリング後のパワースペクトル画像を確認
fshift2 = np.fft.fftshift(f_uv2)
magnitude_spectrum2 = 255 * np.log(np.abs(fshift2)**2)

# フーリエ変換後の画像描画
fig2 = plt.subplot()
fig2.set_xlabel("kx[rad]")
fig2.set_ylabel("ky[rad]")
plt.imshow(magnitude_spectrum2, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# 以下ライブラリを用いた逆フーリエ変換
# シフト分を元に戻す
f_inv_shift = np.fft.ifftshift(fshift2)
# 逆フーリエ変換
img_back = np.fft.ifft2(fshift2)
# 実部を計算する
img_back_abs = np.abs(img_back)

fig, ax = plt.subplots()
ax.imshow(np.abs(f_inv_shift), cmap = 'gray')
ax.imshow(img_back_abs, cmap='gray', )
ax.set_title('Filtered Image 1')
plt.savefig("periodic_removed.png", dpi = 400)
plt.show()

# 周期成分を除去した画像データを二次元配列に格納
img_xy2 = np.asarray(img_back)

"""
以下から、周期成分を除いた信号に対するウィーナーフィルターの適用である
"""
# ウィーナーフィルターのためにノイズの標準偏差sigmaを定義(任意の値に変更して実行)
sigma = 40
# フーリエ変換後のノイズの数値を格納する二次元配列を生成
n_uv = np.zeros_like(img_raw)
# ホワイトノイズのフーリエ変換後の値(定数)をn_uvとして定義する。np.onesで全要素1の二次元配列を作成し、標準偏差sigmaをNxを割った値を掛ける。
n_uv = (sigma / Nx) * np.ones(128, dtype=float)
n_shift = np.fft.fftshift(n_uv)
noise_mag_spectrum = 255 * np.log(np.abs(n_shift)**2)
not_shifted_mag = 255 * np.log(np.abs(n_uv)**2)


# Φ(f)を計算する。Φ(f) < 0　⇒ Φ(f) = 0とすることに注意。
# X(f) = f_inv_shift, N'(f) = n_uv　である
phi_uv = np.zeros_like(img_raw)
phi_uv = (np.abs(f_inv_shift)**2 - np.abs(n_uv)**2) / (np.abs(f_inv_shift)**2)
# マイナス要素をゼロにする。
phi_uv[phi_uv < 0] = 0
# print(phi_uv)

# X~(f) = X(f) * Φ(f)を計算する
X_til = np.zeros_like(img_raw)
X_til = phi_uv * f_inv_shift
# print(X_til)
# X~(f)を逆フーリエ変換する
x_back = np.fft.ifft2(X_til)
# 実部を計算する
x_back_abs = np.abs(x_back)

# ウィーナーフィルターを掛けたのちの画像を描画
fig6, ax = plt.subplots()
ax.imshow(x_back_abs, cmap='gray', )
ax.set_title('Wiener Filtered Image (σ = %s )' % sigma)
plt.savefig("signa%s.png" % sigma, dpi = 400)
plt.show()

"""
# ウィーナーフィルターの推定に用いた散布図の描画
x = []
for i in range(128**2):
    x.append(float(i))
X =  128*128*np.abs(f_inv_shift)**2
X = np.clip(X, None, 10000)
plt.scatter(x,X)
plt.show()
"""
