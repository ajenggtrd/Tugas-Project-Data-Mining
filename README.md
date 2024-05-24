# Tugas Project Data Mining
## Data Shopping Trends
### 1. Business Understanding
Dalam industri ritel, memahami perilaku pelanggan adalah kunci untuk meningkatkan penjualan dan membangun loyalitas. Data pelanggan dapat memberikan informasi tentang bagaimana pola pelanggan dengan produk dan layanan perusahaan. Salah satu metrik penting yang sering digunakan untuk mengukur interaksi ini adalah frekuensi pembelian. Dengan menganalisis perilaku pelanggan, perusahaan dapat mengidentifikasi pola pembelian, menentukan segmen pelanggan yang berbeda, dan memprediksi frekuensi pembelian di masa mendatang.
```python
#Import Library
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
colors = ["#89CFF0", "#FF69B4", "#FFD700", "#7B68EE", "#FF4500",
          "#9370DB", "#32CD32", "#8A2BE2", "#FF6347", "#20B2AA",
          "#FF69B4", "#00CED1", "#FF7F50", "#7FFF00", "#DA70D6"]
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
```
### 2. Data Understanding
Pola pembelian disebabkan oleh berbagai faktor, berikut adalah penjelasan dari setiap faktor yang mungkin memengaruhi pola pembelian berdasarkan data yang diberikan:

- Age (Usia): Usia pelanggan bisa mempengaruhi perilaku pembelian. Pelanggan yang lebih tua atau lebih muda mungkin memiliki frekuensi pembelian yang berbeda.
- Gender (Jenis Kelamin): Jenis kelamin bisa mempengaruhi kebiasaan belanja. Misalnya, produk tertentu mungkin lebih menarik bagi satu jenis kelamin dibanding yang lain
- Item Purchased (Item yg dibeli): Jenis barang yang sering dibeli oleh pelanggan dapat memberikan indikasi tentang kebiasaan pembelian mereka.
- Category (Kategori): Kategori produk yang dibeli dapat mempengaruhi jumlah pembelian sebelumnya. Beberapa kategori mungkin lebih sering dibeli dibandingkan yang lain.
- Purchase Amount (USD) (Jumlah Pembelian): Jumlah pembelian dalam nilai uang dapat memberikan gambaran tentang seberapa sering pelanggan melakukan pembelian. Pelanggan yang sering melakukan pembelian dalam jumlah besar mungkin memiliki jumlah pembelian sebelumnya yang tinggi.
- Location (Lokasi): Lokasi geografis pelanggan dapat mempengaruhi frekuensi pembelian. Pelanggan di daerah kota mungkin memiliki akses yang lebih mudah ke toko atau layanan pengiriman yang lebih cepat dibandingkan dengan pelanggan di daerah pedesaan.
- Size (Ukuran): Ukuran barang yang dibeli. Suatu ukuran tertentu mungkin lebih banyak terjual daripada ukuran lainnya.
- Color (Warna): Warna barang yang dibeli. Suatu warna tertentu mungkin lebih banyak terjual daripada warna lainnya.
- Season (Musim): Musim dapat mempengaruhi pola belanja. Misalnya, pembelian mungkin lebih tinggi selama musim semi atau musim tertentu.
- Review Rating (Rating Ulasan): Rating ulasan dari produk yang dibeli oleh pelanggan dapat mempengaruhi keputusan pembelian mereka di masa mendatang. Namun karena data dalam Review Rating masih rancu, seperti belum tentu yang memberikan rating rendah adalah pelanggan yang frekuensi pembeliannya paling sedikit atau sebaliknya. Sehingga kolom ini akan didrop nantinya.
- Subscription Status (Status Berlangganan): Pelanggan yang memiliki status berlangganan mungkin lebih sering melakukan pembelian dibandingkan yang tidak berlangganan.
- Shipping Type (Jenis Pengiriman): Jenis pengiriman yang dipilih oleh pelanggan. Suatu jenis pengiriman tertentu mungkin lebih banyak dipilih daripada jenis pengiriman lainnya.
- Discount Applied (Diskon yang Diterapkan): Penggunaan diskon dapat mempengaruhi frekuensi pembelian. Pelanggan yang sering menggunakan diskon mungkin lebih sering berbelanja. Namun karena data dalam Discount Applied berisi tipe data boolean dan data tidak menunjukkan seberapa besar jumlah diskon yang digunakan, maka kita juga tidak tahu seberapa besar pengaruh diskon ini dalam jumlah pembelian. Sehingga kolom ini akan didrop nantinya.
- Promo Code Used (Kode Promo yang Digunakan): Sama seperti diskon, penggunaan kode promo dapat mempengaruhi frekuensi pembelian. Namun sama seperti diskon, Promo Code Used ini berisi tipe data boolean dan data tidak menunjukkan seberapa besar jumlah promo yang digunakan, maka kita juga tidak tahu seberapa besar pengaruh promo ini dalam jumlah pembelian. Sehingga kolom ini akan didrop nantinya.
- Previous Purchases (Jumlah Pembelian Sebelumnya): Jumlah pembelian sebelumnya yang dilakukan oleh pelanggan. 
- Preferred Payment Method (Metode Pembayaran yang Disukai): Metode pembayaran yang disukai dapat memberikan indikasi tentang kenyamanan dan kebiasaan belanja pelanggan.
- Frequency of Purchases (Frekuensi Pembelian): Frekuensi pembelian yang dilakukan oleh pelanggan dalam kategori mingguan, bulanan, dan tahunan.
```python
df = pd.read_csv("D:\\Beban\\python\\shopping_trends.csv")
```
```python
df.info
```




    <bound method DataFrame.info of       Customer ID  Age  Gender Item Purchased     Category  \
    0               1   55    Male         Blouse     Clothing   
    1               2   19    Male        Sweater     Clothing   
    2               3   50    Male          Jeans     Clothing   
    3               4   21    Male        Sandals     Footwear   
    4               5   45    Male         Blouse     Clothing   
    ...           ...  ...     ...            ...          ...   
    3895         3896   40  Female         Hoodie     Clothing   
    3896         3897   52  Female       Backpack  Accessories   
    3897         3898   46  Female           Belt  Accessories   
    3898         3899   44  Female          Shoes     Footwear   
    3899         3900   52  Female        Handbag  Accessories   
    
          Purchase Amount (USD)       Location Size      Color  Season  \
    0                        53       Kentucky    L       Gray  Winter   
    1                        64          Maine    L     Maroon  Winter   
    2                        73  Massachusetts    S     Maroon  Spring   
    3                        90   Rhode Island    M     Maroon  Spring   
    4                        49         Oregon    M  Turquoise  Spring   
    ...                     ...            ...  ...        ...     ...   
    3895                     28       Virginia    L  Turquoise  Summer   
    3896                     49           Iowa    L      White  Spring   
    3897                     33     New Jersey    L      Green  Spring   
    3898                     77      Minnesota    S      Brown  Summer   
    3899                     81     California    M      Beige  Spring   
    
          Review Rating Subscription Status Payment Method   Shipping Type  \
    0              3.10                 Yes    Credit Card         Express   
    1              3.10                 Yes  Bank Transfer         Express   
    2              3.10                 Yes           Cash   Free Shipping   
    3              3.50                 Yes         PayPal    Next Day Air   
    4              2.70                 Yes           Cash   Free Shipping   
    ...             ...                 ...            ...             ...   
    3895           4.20                  No           Cash  2-Day Shipping   
    3896           4.50                  No         PayPal    Store Pickup   
    3897           2.90                  No    Credit Card        Standard   
    3898           3.80                  No         PayPal         Express   
    3899           3.10                  No  Bank Transfer    Store Pickup   
    
         Discount Applied Promo Code Used  Previous Purchases  \
    0                 Yes             Yes                  14   
    1                 Yes             Yes                   2   
    2                 Yes             Yes                  23   
    3                 Yes             Yes                  49   
    4                 Yes             Yes                  31   
    ...               ...             ...                 ...   
    3895               No              No                  32   
    3896               No              No                  41   
    3897               No              No                  24   
    3898               No              No                  24   
    3899               No              No                  33   
    
         Preferred Payment Method Frequency of Purchases  
    0                       Venmo            Fortnightly  
    1                        Cash            Fortnightly  
    2                 Credit Card                 Weekly  
    3                      PayPal                 Weekly  
    4                      PayPal               Annually  
    ...                       ...                    ...  
    3895                    Venmo                 Weekly  
    3896            Bank Transfer              Bi-Weekly  
    3897                    Venmo              Quarterly  
    3898                    Venmo                 Weekly  
    3899                    Venmo              Quarterly  
    
    [3900 rows x 19 columns]>

