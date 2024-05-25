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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import scipy.cluster.hierarchy as sch
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
data = pd.read_csv("D:\\Beban\\python\\shopping_trends.csv")
```
```python
data.info
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
### 3. Data Preprocessing
#### 3.1 Mengecek missing value
```python
#mengecek missing value
data.isnull().sum()
```




    Customer ID                 0
    Age                         0
    Gender                      0
    Item Purchased              0
    Category                    0
    Purchase Amount (USD)       0
    Location                    0
    Size                        0
    Color                       0
    Season                      0
    Review Rating               0
    Subscription Status         0
    Payment Method              0
    Shipping Type               0
    Discount Applied            0
    Promo Code Used             0
    Previous Purchases          0
    Preferred Payment Method    0
    Frequency of Purchases      0
    dtype: int64

Terlihat bahwa tidak ada missing value pada data.
#### 3.2 Menyamakan istilah yang punya arti yang sama pada Frequency of Purchases
```python
# Menghitung jumlah masing-masing nilai dan mengurutkan berdasarkan nilai kemunculan
frequency_counts_sorted = data['Frequency of Purchases'].value_counts(sort=False)

print(frequency_counts_sorted)
```

    Frequency of Purchases
    Fortnightly       542
    Weekly            539
    Annually          572
    Quarterly         563
    Bi-Weekly         547
    Monthly           553
    Every 3 Months    584
    Name: count, dtype: int64
Keterangan data pada kolom Frequency of Purchases
- Fortnightly (Setiap Dua Minggu): 542 pelanggan melakukan pembelian setiap dua minggu sekali.
- Weekly (Mingguan): 539 pelanggan melakukan pembelian setiap minggu.
- Annually (Tahunan): 572 pelanggan melakukan pembelian setiap tahun.
- Quarterly (Triwulanan): 563 pelanggan melakukan pembelian setiap tiga bulan sekali.
- Bi-Weekly (Dua Mingguan): 547 pelanggan melakukan pembelian setiap dua minggu sekali (sama seperti Fortnightly).
- Monthly (Bulanan): 553 pelanggan melakukan pembelian setiap bulan.
- Every 3 Months (Setiap 3 Bulan): 584 pelanggan melakukan pembelian setiap tiga bulan sekali (sama seperti Quarterly).
Terlihat bahwa ada data yang memiliki kesamaan seperti Fortnightly dan BI-Weekly yang memiliki rentang waktu yang sama yaitu dua mingguan, serta Every 3 Months dan Quarterly yang memiliki rentang waktu yang sama yaitu tiga bulanan.
```python
# Menyamakan istilah 'Fortnightly' dan 'Bi-Weekly'
data['Frequency of Purchases'] = data['Frequency of Purchases'].replace({'Bi-Weekly': 'Fortnightly'})
data['Frequency of Purchases'] = data['Frequency of Purchases'].replace({'Every 3 Months': 'Quarterly'})

frequency_counts = data['Frequency of Purchases'].value_counts()

print(frequency_counts)
```

    Frequency of Purchases
    Quarterly      1147
    Fortnightly    1089
    Annually        572
    Monthly         553
    Weekly          539
    Name: count, dtype: int64
```python
data['Frequency of Purchases'] = data['Frequency of Purchases'].replace({'Bi-Weekly': 'Fortnightly', 'Every 3 Months': 'Quarterly'})
```
#### 3.3 Meng-copy data untuk tetap mempertahankan data asli
```python
data2 = data.copy()
```
#### 3.4 Data preparation untuk clustering dan regresi
Tipe data akan diubah terlebih dahulu untuk bisa memodelkan Clustering dan Regresi 

```python
size_counts = data2['Size'].value_counts(sort=False)

print(size_counts)
```

    Size
    L     1053
    S      663
    M     1755
    XL     429
    Name: count, dtype: int64

```python
#melakukan konversi tipe data pada kolom frequency of purchases dari kategori ke numerik
sz = {'S':0,'M':1,'L':2,'XL':3}
data2['Size'] = data2['Size'].replace(sz)

size_counts = data2['Size'].value_counts()

print(size_counts)
```

    Size
    1    1755
    2    1053
    0     663
    3     429
    Name: count, dtype: int64

Akan didrop kolom Customer ID, Review Rating, Discount Applied, Promo Code Used seperti yang telah dijelaskan pada Data Understanding
```python
data2 = data2.drop(columns=['Customer ID', 'Review Rating','Discount Applied', 'Promo Code Used'])
data2
```
```python
encoder = LabelEncoder()
data2.columns
for col in data2.columns:
        data2[col] = encoder.fit_transform(data2[col]).astype(int)
data2
```


  <div id="df-2f7d1c0a-b7b2-4051-8949-3ba6f91f9a17" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Item Purchased</th>
      <th>Category</th>
      <th>Purchase Amount (USD)</th>
      <th>Location</th>
      <th>Size</th>
      <th>Color</th>
      <th>Season</th>
      <th>Subscription Status</th>
      <th>Payment Method</th>
      <th>Shipping Type</th>
      <th>Previous Purchases</th>
      <th>Preferred Payment Method</th>
      <th>Frequency of Purchases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>33</td>
      <td>16</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>13</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>23</td>
      <td>1</td>
      <td>44</td>
      <td>18</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>53</td>
      <td>20</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>22</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>14</td>
      <td>2</td>
      <td>70</td>
      <td>38</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>48</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>29</td>
      <td>36</td>
      <td>1</td>
      <td>21</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>30</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3895</th>
      <td>22</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>8</td>
      <td>45</td>
      <td>2</td>
      <td>21</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>31</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3896</th>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>14</td>
      <td>2</td>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3897</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>29</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>23</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3898</th>
      <td>26</td>
      <td>0</td>
      <td>17</td>
      <td>2</td>
      <td>57</td>
      <td>22</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>23</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3899</th>
      <td>34</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>61</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>32</td>
      <td>5</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>3900 rows × 15 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2f7d1c0a-b7b2-4051-8949-3ba6f91f9a17')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2f7d1c0a-b7b2-4051-8949-3ba6f91f9a17 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2f7d1c0a-b7b2-4051-8949-3ba6f91f9a17');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-8c01b44d-cefe-452c-bb0f-24786c0cdb75">
  <button class="colab-df-quickchart" onclick="quickchart('df-8c01b44d-cefe-452c-bb0f-24786c0cdb75')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-8c01b44d-cefe-452c-bb0f-24786c0cdb75 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

```python
data2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3900 entries, 0 to 3899
    Data columns (total 15 columns):
     #   Column                    Non-Null Count  Dtype
    ---  ------                    --------------  -----
     0   Age                       3900 non-null   int32
     1   Gender                    3900 non-null   int32
     2   Item Purchased            3900 non-null   int32
     3   Category                  3900 non-null   int32
     4   Purchase Amount (USD)     3900 non-null   int32
     5   Location                  3900 non-null   int32
     6   Size                      3900 non-null   int32
     7   Color                     3900 non-null   int32
     8   Season                    3900 non-null   int32
     9   Subscription Status       3900 non-null   int32
     10  Payment Method            3900 non-null   int32
     11  Shipping Type             3900 non-null   int32
     12  Previous Purchases        3900 non-null   int32
     13  Preferred Payment Method  3900 non-null   int32
     14  Frequency of Purchases    3900 non-null   int32
    dtypes: int32(15)
    memory usage: 228.6 KB
    

```python
#mengecek nilai unik setiap kolom
data2.nunique().sort_values()
```
    Gender                       2
    Subscription Status          2
    Category                     4
    Size                         4
    Season                       4
    Frequency of Purchases       5
    Payment Method               6
    Shipping Type                6
    Preferred Payment Method     6
    Item Purchased              25
    Color                       25
    Location                    50
    Previous Purchases          50
    Age                         53
    Purchase Amount (USD)       81
    dtype: int64

```python
data2.columns
```

    Index(['Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)',
           'Location', 'Size', 'Color', 'Season', 'Subscription Status',
           'Payment Method', 'Shipping Type', 'Previous Purchases',
           'Preferred Payment Method', 'Frequency of Purchases'],
          dtype='object')

### 4. Modeling
#### 4.1 Visualisasi
```python
grouped_data = data.groupby('Frequency of Purchases')['Age'].mean().reset_index()
```


```python
plt.figure(figsize=(10, 5))
plt.plot(grouped_data['Frequency of Purchases'], grouped_data['Age'], marker='o', linestyle='-', color='b', label='Nilai')
plt.xlabel('Frequency of Purchases')
plt.ylabel('Average Age')
plt.title('Grafik Garis Rata-Rata Usia Berdasarkan Frequency of Purchases')
plt.legend()
plt.grid(True)
plt.show()

```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Grafik%20Garis%20Rata-Rata%20Usia%20Berdasarkan%20Frequency%20of%20Purchases.png">

Terdapat kesalahan pada judul grafik yang seharusnya adalah "Grafik Garis Rata-rata Frekuensi Pembelian Berdasarkan Usia".



```python
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Frequency of Purchases', y='Previous Purchases', marker='o', ci=None)
plt.xlabel('Frequency of Purchases')
plt.ylabel('Previous Purchases')
plt.title('Hubungan Frekuensi Pembelian dengan Jumlah Pembelian Sebelumnya')
plt.xticks(rotation=45)
plt.show()
```

    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_16976\136495084.py:2: FutureWarning: 
    
    The `ci` parameter is deprecated. Use `errorbar=None` for the same effect.
    
      sns.lineplot(data=data, x='Frequency of Purchases', y='Previous Purchases', marker='o', ci=None)
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Hubungan%20Frekuensi%20Pembelian%20dengan%20Jumlah%20Pembelian%20Sebelumnya.png">

    
```python
plt.figure(figsize=(8, 6))
data['Frequency of Purchases'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightblue', 'lightpink', 'lightyellow', 'lightcoral'])
plt.title('Proporsi Frekuensi Pembelian')
plt.ylabel('')
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Proporsi%20Frekuensi%20Pembelian.png">

```python
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Frequency of Purchases', palette='pastel')
plt.xlabel('Frekuensi Pembelian')
plt.ylabel('Jumlah Pelanggan')
plt.title('Jumlah Pelanggan berdasarkan Frekuensi Pembelian')
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Jumlah%20Pelanggan%20berdasarkan%20Frekuensi%20Pembelian.png">
Terdapat kesalahan pada judul plot yang seharusnya adalah "Frekuensi Pembelian berdasarkan Jumlah Pelanggan". Pada diagram tersebut menunjukkan bahwa mayoritas pelanggan berbelanja dalam rentang waktu 3 bulan sekali.

```python
plt.figure(figsize=(10, 6))
total_purchase = data.groupby('Frequency of Purchases')['Purchase Amount (USD)'].sum().reset_index()
sns.barplot(data=total_purchase, x='Frequency of Purchases', y='Purchase Amount (USD)', palette='Set2')
plt.xlabel('Frekuensi Pembelian')
plt.ylabel('Total Pembelian (USD)')
plt.title('Total Pembelian berdasarkan Frekuensi Pembelian')
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Total%20Pembelian%20berdasarkan%20Frekuensi%20Pembelian.png">
Terdapat kesalahan pada judul plot yang seharusnya adalah "Frekuensi Pembelian berdasarkan Total Pembelian (USD))". Dikarenakan berdasarkan diagram dan plot sebelumnya telah menunjukkan bahwa mayoritas pelanggan berbelanja dalam rentang waktu 3 bulan sekali, jelas bahwa total pembelian terbanyak ada di rentang waktu 3 bulan.

```python
plt.figure(figsize=(10, 6))
sns.swarmplot(data=data, x='Subscription Status', y='Age', palette='Set1')
plt.xlabel('Status Langganan')
plt.ylabel('Usia')
plt.title('Hubungan Status Langganan dengan Usia Pelanggan')
plt.show()
```

    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_16976\1980800469.py:2: FutureWarning: Passing `palette` without assigning `hue` is deprecated.
      sns.swarmplot(data=data, x='Subscription Status', y='Age', palette='Set1')
    c:\Users\LENOVO\anaconda3\Lib\site-packages\seaborn\categorical.py:3544: UserWarning: 20.2% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
      warnings.warn(msg, UserWarning)
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Hubungan%20Status%20Langganan%20dengan%20Usia%20Pelanggan.png">

```python
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Subscription Status', palette='pastel')
plt.xlabel('Status Langganan')
plt.ylabel('Jumlah Pelanggan')
plt.title('Jumlah Pelanggan berdasarkan Status Langganan')
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Jumlah%20Pelanggan%20berdasarkan%20Status%20Langganan.png">
Terdapat kesalahan pada judul plot yang seharusnya adalah "Status Langganan berdasarkan Jumlah Pelanggan". Diagram diatas menunjukkan bahwa lebih banyak pelanggan yang tidak melakukan subscribe dibanding yang subscribe

```python
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Subscription Status', multiple='stack', palette='Set2')
plt.xlabel('Usia')
plt.ylabel('Jumlah Pelanggan')
plt.title('Distribusi Usia untuk Setiap Status Langganan')
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Distribusi%20Usia%20untuk%20Setiap%20Status%20Langganan.png">

```python
plt.figure(figsize=(8, 6))
data['Subscription Status'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('Proporsi Status Langganan')
plt.ylabel('')
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Proporsi%20Status%20Langganan.png">
Diagram diatas menunjukkan jumlah pelanggan yang melakukan subscribe dan tidak melakukan subscribe dalam persen.

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Purchase Amount (USD)', hue='Gender', palette='Set1')
plt.xlabel('Usia')
plt.ylabel('Jumlah Pembelian (USD)')
plt.title('Hubungan Usia dan Jumlah Pembelian')
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Hubungan%20usia%20dan%20jumlah%20pembelian.png">

#### 4.2. Clustering
```python
LE = LabelEncoder()
data2["Frequency of Purchases"] = LE.fit_transform(data2["Frequency of Purchases"])
CL = data2[['Frequency of Purchases', 'Age',"Purchase Amount (USD)"]]

```


```python
CL.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Frequency of Purchases</th>
      <td>3900.0</td>
      <td>1.997949</td>
      <td>1.308914</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>3900.0</td>
      <td>26.068462</td>
      <td>15.207589</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>26.0</td>
      <td>39.0</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>Purchase Amount (USD)</th>
      <td>3900.0</td>
      <td>39.764359</td>
      <td>23.685392</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>40.0</td>
      <td>61.0</td>
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
CL.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Frequency of Purchases</th>
      <th>Age</th>
      <th>Purchase Amount (USD)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Frequency of Purchases</th>
      <td>1.000000</td>
      <td>0.008240</td>
      <td>-0.009397</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.008240</td>
      <td>1.000000</td>
      <td>-0.010424</td>
    </tr>
    <tr>
      <th>Purchase Amount (USD)</th>
      <td>-0.009397</td>
      <td>-0.010424</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
kmeans = KMeans(n_clusters=3,init="k-means++" )
kmeans.fit(CL)
y_kmeans = kmeans.predict(CL)
y_kmeans
```

    c:\Users\LENOVO\anaconda3\Lib\site-packages\sklearn\cluster\_kmeans.py:1412: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      super()._check_params_vs_input(X, default_n_init=10)
    




    array([1, 0, 2, ..., 1, 2, 2])




```python
def plot_3d_clusters(y_pred,cluster_type) :
      fig = plt.figure(figsize=(8, 6))
      ax = fig.add_subplot(111, projection='3d')

      colors = ['blue', 'red', 'black']
      for cluster_label in range(3):
            ax.scatter(CL[y_pred == cluster_label]['Frequency of Purchases'],
                              CL[y_pred == cluster_label]['Age'],
                              CL[y_pred == cluster_label]["Purchase Amount (USD)"],
                              label=f'Cluster {cluster_label}', c=colors[cluster_label])


      ax.set_xlabel('Frequency of Purchases')
      ax.set_ylabel('Age')
      ax.set_zlabel('Purchase Amount (USD)',)
      ax.set_title(f'{cluster_type} Clustering ')
      ax.legend()
      plt.show()
plot_3d_clusters(kmeans.labels_,"K-Means")
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/K-Means.png">

    



```python
wcss = []
for i in range(1,11):       #range value
  kmeans = KMeans( n_clusters=i , init="k-means++" , random_state = 42,n_init=10)
  kmeans.fit(CL)
  wcss.append(kmeans.inertia_)

#plotting graph
plt.plot(range(1,11),wcss)
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/wcss.png">


```python
ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean",linkage="ward")
y_pred = ac.fit_predict(CL)
y_pred
plot_3d_clusters(y_pred,"Agglomerative")
```

    c:\Users\LENOVO\anaconda3\Lib\site-packages\sklearn\cluster\_agglomerative.py:1005: FutureWarning: Attribute `affinity` was deprecated in version 1.2 and will be removed in 1.4. Use `metric` instead
      warnings.warn(
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Agglomerative.png">


1. Sumbu x mewakili Frequency of purchases, berkisar antara 0 hingga 6.
2. Sumbu y mewakili Age, yang berkisar antara 20 hingga 100.
3. Sumbu z mewakili Purchases Amount, yang berkisar antara 30 hingga 100.

Dari plot tersebut, kita dapat mengamati hal-hal berikut:
- Cluster 0 (titik biru), terdiri dari pelanggan yang sering melakukan pembelian (nilai x tinggi) dan relatif muda (nilai y rendah). Mereka juga cenderung memiliki Jumlah Pembelian yang lebih rendah (nilai z yang lebih rendah).
- Cluster 1 (titik merah), memiliki pelanggan yang melakukan pembelian dalam jumlah sedang (nilai x sedang) dan berusia paruh baya (nilai y sedang). Mereka cenderung memiliki Jumlah Pembelian yang moderat (nilai z sedang).
- Cluster 2 (titik hijau), terdiri dari pelanggan yang jarang melakukan pembelian (nilai x rendah) dan berusia lebih tua (nilai y tinggi). Mereka cenderung memiliki Jumlah Pembelian yang lebih tinggi (nilai z yang lebih tinggi).

```python
kelas_biru = data[(data['Purchase Amount (USD)'] >= 70) & (data['Age'] < 30)]
kelas_merah = data[(data['Purchase Amount (USD)'] >= 30) & (data['Purchase Amount (USD)'] < 70) & (data['Age'] >= 30) & (data['Age'] < 70)]
kelas_hijau = data[(data['Purchase Amount (USD)'] < 30) & (data['Age'] >= 70)]

count_kelas_biru = len(kelas_biru)
count_kelas_merah = len(kelas_merah)
count_kelas_hijau = len(kelas_hijau)

plt.figure(figsize=(10, 6))
sns.barplot(x=['Cluster Rendah', 'Cluster Sedang', 'Cluster Tinggi'], y=[count_kelas_biru, count_kelas_merah, count_kelas_hijau], palette=['blue', 'red', 'green'])

plt.xlabel('Cluster')
plt.ylabel('Jumlah Data')

plt.title('Jumlah Data dalam Setiap Cluster berdasarkan Purchase Amount dan Age')

plt.grid(True)
plt.show()
```
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Jumlah%20Data%20dalam%20Setiap%20Cluster%20berdasarkan%20Purchase%20Amount%20dan%20Age.png">

#### 4.3 Regression
##### 4.3.1 Melihat distribusi data
```python
sns.distplot(data2['Frequency of Purchases'],kde=True)
plt.title('Distribution of Frequency of Purchases')
```

    C:\Users\LENOVO\AppData\Local\Temp\ipykernel_16976\3536154466.py:1: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      sns.distplot(data2['Frequency of Purchases'],kde=True)
    




    Text(0.5, 1.0, 'Distribution of Frequency of Purchases')
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Distribution%20of%20Frequency%20of%20Purchases.png">




```python
columns=['Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)',
       'Location', 'Size', 'Color', 'Season', 'Subscription Status',
       'Payment Method', 'Shipping Type', 'Previous Purchases',
       'Preferred Payment Method']
plt.figure(figsize=(10,8))

num_columns = 4
num_rows = (len(columns) + num_columns - 1) // num_columns  # Pembulatan ke atas untuk memastikan cukup ruang

plt.figure(figsize=(20, num_rows * 4))

for i, column in enumerate(columns):
    plt.subplot(num_rows, num_columns, i + 1)
    sns.histplot(data2[column], kde=True)
    plt.title(column)

plt.tight_layout()
plt.show()
```


    <Figure size 1000x800 with 0 Axes>
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/download%20(14).png">



```python
columns=['Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)',
       'Location', 'Size', 'Color', 'Season', 'Subscription Status',
       'Payment Method', 'Shipping Type', 'Previous Purchases',
       'Preferred Payment Method']
plt.figure(figsize=(10,8))

num_columns = 4
num_rows = (len(columns) + num_columns - 1) // num_columns  # Pembulatan ke atas untuk memastikan cukup ruang

plt.figure(figsize=(20, num_rows * 4))

for i, column in enumerate(columns):
    plt.subplot(num_rows, num_columns, i + 1)
    sns.boxplot(data2[column], orient='h')
    plt.title(column)

plt.tight_layout()
plt.show()
```


    <Figure size 1000x800 with 0 Axes>
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/download%20(15).png">

##### 4.3.2 Mendeteksi outliers pada data
```python
#mendeteksi outliers
z_scores = np.abs((data2 - data2.mean()) / data2.std())

outliers = data2[(z_scores > 3).any(axis=1)]
print("Outliers berdasarkan z-score:")
print(outliers)
```

    Outliers berdasarkan z-score:
    Empty DataFrame
    Columns: [Age, Gender, Item Purchased, Category, Purchase Amount (USD), Location, Size, Color, Season, Subscription Status, Payment Method, Shipping Type, Previous Purchases, Preferred Payment Method, Frequency of Purchases]
    Index: []
Tidak ada outliers pada data.
##### 4.3.3 Melihat korelasi antar variabel
```python

plt.figure(figsize=(12,12))
sns.heatmap(data2.corr(),annot=True)
plt.title('Correlation Matrix')
```


    Text(0.5, 1.0, 'Correlation Matrix')
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/Correlation%20matrix.png">


##### 4.3.4 Regression
Membagi data training dan data test
```python
X = data2.drop(['Purchase Amount (USD)'],axis=1)
Y = data2['Purchase Amount (USD)']
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
```


```python
onehot_encoder = OneHotEncoder()
Y_encoded = onehot_encoder.fit_transform(Y.values.reshape(-1,1)).toarray()

```
```python
# Konversi Y_train ke dalam bentuk array
Y_train_array = np.array(Y_train)

# Konversi Y_test ke dalam bentuk array
Y_test_array = np.array(Y_test)
```


```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
#mengecek proporsi X train, X test, Y train dan Y test
print('Jumlah baris dan kolom dari x_train adalah:',X_train_scaled.shape,', sedangkan jumlah baris dan kolom dari y_train adalah:',Y_train.shape)
print('Jumlah baris dan kolom dari x_test adalah:',X_test_scaled.shape,', sedangkan jumlah baris dan kolom dari y_test adalah:',Y_test.shape)

```

    Jumlah baris dan kolom dari x_train adalah: (3120, 14) , sedangkan jumlah baris dan kolom dari y_train adalah: (3120,)
    Jumlah baris dan kolom dari x_test adalah: (780, 14) , sedangkan jumlah baris dan kolom dari y_test adalah: (780,)

###### 4.3.4.1 Linear Regression
```python
def pred_model(model,X_train_scaled,Y_train,X_test_scaled,Y_test):
  c=model()
  c.fit(X_train_scaled,Y_train)
  y_train_pred = c.predict(X_train_scaled)
  y_test_pred = c.predict(X_test_scaled)
  print(model)
  print('Coefficient: ',c.coef_)
  print('Intercept: ',c.intercept_)
  print(f'MSE Testing:{mean_squared_error(Y_test,y_test_pred)}')
  print(f'MSE Training:{mean_squared_error(Y_train,y_train_pred)}\n')
  print(f'MAE Testing:{mean_absolute_error(Y_test,y_test_pred)}')
  print(f'MAE Training:{mean_absolute_error(Y_train,y_train_pred)}\n')
  print(f'R2 Testing:{r2_score(Y_test,y_test_pred)}')
  print(f'R2 Training:{r2_score(Y_train,y_train_pred)}\n')
  print(f'RMSE Testing:{np.sqrt(mean_squared_error(Y_test,y_test_pred))}')
  print(f'RMSE Training:{np.sqrt(mean_squared_error(Y_train,y_train_pred))}')

  plt.figure(figsize=[15,4])
  plt.subplot(1,2,2)
  plt.scatter(Y_train,y_train_pred)
  plt.plot([Y_train.min(),Y_train.max()],[Y_train.min(),Y_train.max()],'r--')
  plt.title('Comparison of Y train (Actual value) with y train pred (Prediction result)')
  plt.xlabel('Y_train')
  plt.ylabel('y_train_pred')
  plt.show()

  plt.figure(figsize=[15,4])
  plt.subplot(1,2,2)
  plt.scatter(Y_test,y_test_pred)
  plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()],'r--')
  plt.title('Comparison of Y test (Actual value) with y test pred (Prediction result)')
  plt.xlabel('Y_test')
  plt.ylabel('y_test_pred')
  plt.show()
```
```python
pred_model(LinearRegression,X_train,Y_train,X_test,Y_test)
```

    <class 'sklearn.linear_model._base.LinearRegression'>
    Coefficient:  [ 0.00654354  0.00342444 -0.00794197  0.00345095 -0.01621437  0.00477491
     -0.00136782  0.00144745 -0.00685942 -0.01117339  0.0023811  -0.00038088
      0.00991134  0.00711008]
    Intercept:  0.48529821655910244
    MSE Testing:0.2520054208846168
    MSE Training:0.2491443334498762
    
    MAE Testing:0.5011156579112225
    MAE Training:0.4982886668997524
    
    R2 Testing:-0.010173532483402026
    R2 Training:0.003028974075284241
    
    RMSE Testing:0.502001415221727
    RMSE Training:0.4991436000289658
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/model%20linear.png">

###### 4.3.4.2 Lasso Regression
```python
pred_model(Lasso,X_train_scaled,Y_train,X_test_scaled,Y_test)
```

    <class 'sklearn.linear_model._coordinate_descent.Lasso'>
    Coefficient:  [ 0.  0. -0.  0. -0.  0. -0.  0. -0. -0.  0. -0.  0.  0.]
    Intercept:  0.49006410256410254
    MSE Testing:0.2505573019395135
    MSE Training:0.24990127794214334
    
    MAE Testing:0.5004585798816569
    MAE Training:0.4998025558842866
    
    R2 Testing:-0.004368691360952859
    R2 Training:0.0
    
    RMSE Testing:0.5005569916997599
    RMSE Training:0.49990126819417385
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/model%20lasso.png">

###### 4.3.4.3 Ridge Regression
```python
pred_model(Ridge,X_train_scaled,Y_train,X_test_scaled,Y_test)
```

    <class 'sklearn.linear_model._ridge.Ridge'>
    Coefficient:  [ 0.00654238  0.00344114 -0.00790741  0.00345238 -0.01606718  0.00476693
     -0.00136625  0.00144675 -0.00682661 -0.01122873  0.00238485 -0.00552771
      0.00993464  0.00928448]
    Intercept:  0.49006410256410254
    MSE Testing:0.25200464006816087
    MSE Training:0.24914433354035376
    
    MAE Testing:0.5011154226389031
    MAE Training:0.49828918487327495
    
    R2 Testing:-0.010170402550265356
    R2 Training:0.0030289737132309646
    
    RMSE Testing:0.5020006375176838
    RMSE Training:0.49914360011959863
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/model%20ridge.png">

###### 4.3.4.4 ElasticNet Regression
```python
pred_model(ElasticNet,X_train_scaled,Y_train,X_test_scaled,Y_test)
```

    <class 'sklearn.linear_model._coordinate_descent.ElasticNet'>
    Coefficient:  [ 0.  0. -0.  0. -0.  0. -0.  0. -0. -0.  0. -0.  0.  0.]
    Intercept:  0.49006410256410254
    MSE Testing:0.2505573019395135
    MSE Training:0.24990127794214334
    
    MAE Testing:0.5004585798816569
    MAE Training:0.4998025558842866
    
    R2 Testing:-0.004368691360952859
    R2 Training:0.0
    
    RMSE Testing:0.5005569916997599
    RMSE Training:0.49990126819417385
<img src="https://github.com/ajenggtrd/Tugas-Project-Data-Mining/blob/main/model%20elasticnet.png">
    
###### 4.3.4.5 RandomForest Regression
```python
X_train_svr, X_test_svr, Y_train_svr, Y_test_svr = train_test_split(X, Y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_svr, Y_train_svr)
cross_val_scores = cross_val_score(rf_model, X, Y, cv=5, scoring='neg_mean_squared_error')
cross_val_rmse_scores = np.sqrt(-cross_val_scores)

# Evaluasi model
y_train_pred = rf_model.predict(X_train_svr)
y_test_pred = rf_model.predict(X_test_svr)

mse_train = mean_squared_error(Y_train_svr, y_train_pred)
mse_test = mean_squared_error(Y_test_svr, y_test_pred)
mae_train = mean_absolute_error(Y_train_svr, y_train_pred)
mae_test = mean_absolute_error(Y_test_svr, y_test_pred)
r2_train = r2_score(Y_train_svr, y_train_pred)
r2_test = r2_score(Y_test_svr, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

print(f'MSE Training:{mse_train}')
print(f'MSE Testing:{mse_test}')
print(f'MAE Training:{mae_train}')
print(f'MAE Testing:{mae_test}')
print(f'R2 Training:{r2_train}')
print(f'R2 Testing:{r2_test}')
print(f'RMSE Training:{rmse_train}')
print(f'RMSE Testing:{rmse_test}')
print(f'Cross-Validation RMSE: {cross_val_rmse_scores}')
print(f'Average Cross-Validation RMSE: {cross_val_rmse_scores.mean()}')

```

    MSE Training:0.03638022435897436
    MSE Testing:0.2537148717948718
    MAE Training:0.18477884615384615
    MAE Testing:0.4943076923076922
    R2 Training:0.8544216153732633
    R2 Testing:-0.017025932953826706
    RMSE Training:0.19073600698078577
    RMSE Testing:0.5037011731124633
    Cross-Validation RMSE: [0.50410761 0.51236593 0.5056945  0.50831433 0.50536126]
    Average Cross-Validation RMSE: 0.5071687261458543
###### 4.3.4.6 Stacking Regression


```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42)),
    ('ridge', Ridge())
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
stacking_model.fit(X_train_svr, Y_train_svr)

y_train_pred_stack = stacking_model.predict(X_train_svr)
y_test_pred_stack = stacking_model.predict(X_test_svr)

mse_train_stack = mean_squared_error(Y_train_svr, y_train_pred_stack)
mse_test_stack = mean_squared_error(Y_test_svr, y_test_pred_stack)
mae_train_stack = mean_absolute_error(Y_train_svr, y_train_pred_stack)
mae_test_stack = mean_absolute_error(Y_test_svr, y_test_pred_stack)
r2_train_stack = r2_score(Y_train_svr, y_train_pred_stack)
r2_test_stack = r2_score(Y_test_svr, y_test_pred_stack)
rmse_train_stack = np.sqrt(mse_train_stack)
rmse_test_stack = np.sqrt(mse_test_stack)

print(f'MSE Training Stacking: {mse_train_stack}')
print(f'MSE Testing Stacking: {mse_test_stack}')
print(f'MAE Training Stacking: {mae_train_stack}')
print(f'MAE Testing Stacking: {mae_test_stack}')
print(f'R2 Training Stacking: {r2_train_stack}')
print(f'R2 Testing Stacking: {r2_test_stack}')
print(f'RMSE Training Stacking: {rmse_train_stack}')
print(f'RMSE Testing Stacking: {rmse_test_stack}')


```

    MSE Training Stacking: 0.5052942022047413
    MSE Testing Stacking: 0.48114893066692316
    MAE Training Stacking: 0.5935747437810583
    MAE Testing Stacking: 0.5792783978689279
    R2 Training Stacking: -0.03455652327053382
    R2 Testing Stacking: -0.001899572236446545
    RMSE Training Stacking: 0.7108404899868475
    RMSE Testing Stacking: 0.6936489967317211
    
###### 4.3.4.7 Logistic Regression
```python
bins = [0, 50, 80, 100]
labels = ['30-50', '51-80', '81-100']

data2['PA Group'] = pd.cut(data2['Purchase Amount (USD)'], bins=bins, labels=labels, right=False)

encoder = LabelEncoder()
data2['PA Group'] = encoder.fit_transform(data2['PA Group'])

X = data2.drop(['Purchase Amount (USD)','PA Group'],axis=1)
Y = data2['PA Group']

# Bagi data menjadi training dan testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

def model(classifier):
    classifier.fit(X_train, Y_train)
    prediction = classifier.predict(X_test)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    accuracy = accuracy_score(Y_test, prediction)
    precision = precision_score(Y_test, prediction, average='weighted')
    recall = recall_score(Y_test, prediction, average='weighted')
    f1 = f1_score(Y_test, prediction, average='weighted')

    print("Accuracy: {:.2%}".format(accuracy))
    print("Precision: {:.2%}".format(precision))
    print("Recall: {:.2%}".format(recall))
    print("F1 Score: {:.2%}".format(f1))

classifier_lr = LogisticRegression(random_state=0, C=10, penalty='l2', solver='lbfgs', max_iter=500, multi_class='ovr')
model(classifier_lr)

```

    Accuracy: 63.25%
    Precision: 40.00%
    Recall: 63.25%
    F1 Score: 49.01%
    

    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    
Dari beberapa model diatas, terlihat bahwa model Logistic Regression lebih baik dalam prediksi data dibandingkan model lainnya. Akan dievaluasi apakah hasil dari Logistic Regression tersebut sudah cukup baik atau belum menggunakan Cross Validation Score.
##### 4.3.5 Evaluasi Model
```python
classifier_lr = LogisticRegression(random_state=0, C=10, penalty='l2', solver='lbfgs', max_iter=500, multi_class='ovr')

# Hitung cross-validation score
cv_scores = cross_val_score(classifier_lr, X_train, Y_train, cv=5, scoring='accuracy')

print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())
```

    Cross-Validation Scores: [0.60989011 0.60989011 0.61172161 0.61172161 0.61172161]
    Mean Cross-Validation Score: 0.610989010989011
    

Rata-rata skor validasi silang adalah 0.611, yang merupakan nilai yang cukup baik, menunjukkan bahwa model secara keseluruhan dapat melakukan prediksi dengan baik

