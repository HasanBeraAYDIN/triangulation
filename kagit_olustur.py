"""Telefon uygulaması için test kâğıdı: solda gövde-dal birleşimi, sağda ChArUco tahtası.

Tarayıcıdaki OpenCV'de satranç tahtası bulucu yok, ArUco/ChArUco var. Bu yüzden tahta ChArUco.
Birleşim noktasının tahtaya göre yeri kesin biliniyor; uygulama gerçek mesafeyi buradan hesaplar.

  python kagit_olustur.py
Çıktılar (bu klasöre):
  test_kagidi.pdf   A4 yatay, %100 ölçekle yazdırılacak (kare kenarı 22 mm)
  test_ekran.png    aynı kâğıt, ekranda göstermek için
  kagit.jpg         simülasyon modu için 10 px/mm kopya
  kagit.json        uygulamanın okuduğu ölçüler
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
SRC = HERE.parent / "option_b" / "ekran_testi.png"

PX = 20                       # piksel / mm (ana görüntü, ~508 dpi)
PW, PH = 297, 210             # A4 yatay (mm)
SQ, MK = 22.0, 15.4           # kare ve işaret kenarı (mm)
NX, NY = 7, 5                 # kare sayısı -> 6x4 = 24 iç köşe
BX, BY = 136.0, 50.0          # tahtanın sol üst köşesi (mm)
STEM_X, STEM_Y, STEM_W = 10.0, 45.0, 120.0   # gövde resminin yeri ve boyu (mm)

page = Image.new("L", (PW * PX, PH * PX), 255).convert("RGB")

# Gövde resmi: ekran_testi.png'deki kesit (860 px, birleşim kesitte (431.2, 431.0))
src = Image.open(SRC).convert("RGB")
info = json.loads((SRC.with_suffix(".json")).read_text(encoding="utf-8"))
S, c0x, c0y = 90, 960 + 90, (1080 - 7 * 90) // 2 + 90
jx_src = c0x + info["junction_squares"][0] * S
jy_src = c0y + info["junction_squares"][1] * S
crop = src.crop((60, 110, 920, 970))
jx_crop, jy_crop = jx_src - 60, jy_src - 110
side = int(STEM_W * PX)
page.paste(crop.resize((side, side), Image.LANCZOS), (int(STEM_X * PX), int(STEM_Y * PX)))
j_mm = (STEM_X + jx_crop / 860 * STEM_W, STEM_Y + jy_crop / 860 * STEM_W)

# ChArUco tahtası
dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((NX, NY), SQ, MK, dic)
bimg = board.generateImage((int(NX * SQ * PX), int(NY * SQ * PX)), marginSize=0, borderBits=1)
page.paste(Image.fromarray(bimg).convert("RGB"), (int(BX * PX), int(BY * PX)))

# Yazılar ve 10 cm kontrol çizgisi
d = ImageDraw.Draw(page)
try:
    f1 = ImageFont.truetype("arial.ttf", 6 * PX)
    f2 = ImageFont.truetype("arial.ttf", 4 * PX)
except OSError:
    f1 = f2 = ImageFont.load_default()
d.text((10 * PX, 12 * PX), "İki fotoğraf mesafe testi", fill=(0, 0, 0), font=f1)
d.text((10 * PX, 22 * PX), "%100 ölçekle yazdırın. Bir karenin kenarını cetvelle ölçüp uygulamaya girin (tasarım: 22 mm).",
       fill=(60, 60, 60), font=f2)
y = 185 * PX
d.line([(BX * PX, y), ((BX + 100) * PX, y)], fill=(0, 0, 0), width=PX // 2)
for x in (BX, BX + 100):
    d.line([(x * PX, y - 2 * PX), (x * PX, y + 2 * PX)], fill=(0, 0, 0), width=PX // 2)
d.text((BX * PX, y + 3 * PX), "Bu çizgi 100 mm olmalı", fill=(60, 60, 60), font=f2)
# Birleşim noktasını belirten küçük işaret (resmin dışında, kenarda)
d.polygon([(STEM_X * PX - 2, j_mm[1] * PX), ((STEM_X - 4) * PX, (j_mm[1] - 2) * PX), ((STEM_X - 4) * PX, (j_mm[1] + 2) * PX)],
          fill=(200, 0, 0))

page.save(HERE / "test_kagidi.pdf", resolution=PX * 25.4)
page.resize((1920, round(1920 * PH / PW)), Image.LANCZOS).save(HERE / "test_ekran.png")
page.resize((PW * 10, PH * 10), Image.LANCZOS).save(HERE / "kagit.jpg", quality=92)

meta = {
    "squares": [NX, NY], "square_mm": SQ, "marker_mm": MK, "dictionary": "DICT_4X4_50",
    "board_origin_mm": [BX, BY], "page_mm": [PW, PH],
    # Birleşim, tahta koordinatında (kare birimi; x sağa, y aşağı, orijin tahtanın sol üst köşesi)
    "junction_squares": [(j_mm[0] - BX) / SQ, (j_mm[1] - BY) / SQ],
}
(HERE / "kagit.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(json.dumps(meta))

# Kontrol: Python dedektörü kâğıdı bulabiliyor mu?
g = np.array(page.resize((PW * 4, PH * 4)).convert("L"))
det = cv2.aruco.CharucoDetector(board)
cc, ci, _, _ = det.detectBoard(g)
print("kontrol: bulunan iç köşe", 0 if ci is None else len(ci), "/", (NX - 1) * (NY - 1))
