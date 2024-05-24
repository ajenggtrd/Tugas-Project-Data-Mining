# Tugas-Project-Data-Mining
python
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
from sklearn.preprocessing import StandardScaler



python
df = pd.read_csv("D:\\Beban\\python\\shopping_trends.csv")



python
df.info





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




python
#mengecek missing value
df.isnull().sum()





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




python
plt.figure(figsize = (20, 6))
ax = df["Gender"].value_counts().plot(kind = 'bar', color = colors, rot = 0)
ax.set_xticklabels(('Male', 'Female'))

for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Employment Type', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);



    
![png](output_4_0.png)
    



python
plt.figure(figsize = (20, 6))

counts = df["Gender"].value_counts()
explode = (0, 0.1)

counts.plot(kind = 'pie', fontsize = 12, colors = colors, explode = explode, autopct = '%1.1f%%')
plt.xlabel('Gender', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()



    
![png](output_5_0.png)
    



python
fig, ax = plt.subplots(figsize = (20, 5))

ax.hist(df['Age'], bins = 25, edgecolor = 'black', alpha = 0.7, color = 'skyblue', density = True)
df['Age'].plot(kind = 'kde', color = 'red', ax = ax)

ax.set_xlabel('Age')
ax.set_ylabel('Count / Density')
ax.set_title('Age Distribution Histogram with Density Curve')
ax.legend(['Density Curve', 'Histogram'])
plt.show()



    
![png](output_6_0.png)
    



python
df["Category"].value_counts()





    Category
    Clothing       1737
    Accessories    1240
    Footwear        599
    Outerwear       324
    Name: count, dtype: int64




python
plt.figure(figsize = (20, 6))

counts = df["Category"].value_counts()
explode = (0, 0.0, 0.0, 0.1)

counts.plot(kind = 'pie', fontsize = 12, colors = colors, explode = explode, autopct = '%1.1f%%')
plt.xlabel('Gender', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()



    
![png](output_8_0.png)
    



python
df["Item Purchased"].value_counts()





    Item Purchased
    Blouse        171
    Jewelry       171
    Pants         171
    Shirt         169
    Dress         166
    Sweater       164
    Jacket        163
    Belt          161
    Sunglasses    161
    Coat          161
    Sandals       160
    Socks         159
    Skirt         158
    Shorts        157
    Scarf         157
    Hat           154
    Handbag       153
    Hoodie        151
    Shoes         150
    T-shirt       147
    Sneakers      145
    Boots         144
    Backpack      143
    Gloves        140
    Jeans         124
    Name: count, dtype: int64




python
plt.figure(figsize = (16, 7))
df["Item Purchased"].value_counts().sort_values(ascending = True).plot(kind = 'barh', color = sns.color_palette('tab20'), edgecolor = 'black')
plt.ylabel('Item Purchased', fontsize = 16)
plt.xlabel('\nNumber of Occurrences', fontsize = 16)
plt.title('Item Purchased\n', fontsize = 16)
plt.xticks(rotation = 0, ha = 'center', fontsize = 16)
plt.tight_layout()
plt.show()



    
![png](output_10_0.png)
    



python
df["Location"].value_counts()





    Location
    Montana           96
    California        95
    Idaho             93
    Illinois          92
    Alabama           89
    Minnesota         88
    Nebraska          87
    New York          87
    Nevada            87
    Maryland          86
    Delaware          86
    Vermont           85
    Louisiana         84
    North Dakota      83
    Missouri          81
    West Virginia     81
    New Mexico        81
    Mississippi       80
    Indiana           79
    Georgia           79
    Kentucky          79
    Arkansas          79
    North Carolina    78
    Connecticut       78
    Virginia          77
    Ohio              77
    Tennessee         77
    Texas             77
    Maine             77
    South Carolina    76
    Colorado          75
    Oklahoma          75
    Wisconsin         75
    Oregon            74
    Pennsylvania      74
    Washington        73
    Michigan          73
    Alaska            72
    Massachusetts     72
    Wyoming           71
    Utah              71
    New Hampshire     71
    South Dakota      70
    Iowa              69
    Florida           68
    New Jersey        67
    Hawaii            65
    Arizona           65
    Kansas            63
    Rhode Island      63
    Name: count, dtype: int64




python
plt.figure(figsize = (16, 6))
df["Location"].value_counts()[:10].sort_values(ascending = False).plot(kind = 'bar', color = sns.color_palette('inferno'), edgecolor = 'black')
plt.xlabel('Location', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('\nNumber of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
plt.xticks(rotation = 0, ha = 'center', fontsize = 16)
plt.tight_layout()
plt.show()



    
![png](output_12_0.png)
    



python
df["Size"].value_counts()





    Size
    M     1755
    L     1053
    S      663
    XL     429
    Name: count, dtype: int64




python
plt.figure(figsize = (20, 6))
ax = df["Size"].value_counts().plot(kind = 'bar', color = colors, rot = 0)
ax.set_xticklabels(('Medium', 'Large', 'Small', 'Extra Large'))

for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Size', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);



    
![png](output_14_0.png)
    



python
plt.figure(figsize = (20, 6))

counts = df["Category"].value_counts()
explode = (0, 0.0, 0.0, 0.1)

counts.plot(kind = 'pie', fontsize = 12, colors = colors, explode = explode, autopct = '%1.1f%%')
plt.xlabel('Size', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()



    
![png](output_15_0.png)
    



python
df["Color"].value_counts()





    Color
    Olive        177
    Yellow       174
    Silver       173
    Teal         172
    Green        169
    Black        167
    Cyan         166
    Violet       166
    Gray         159
    Maroon       158
    Orange       154
    Charcoal     153
    Pink         153
    Magenta      152
    Blue         152
    Purple       151
    Peach        149
    Red          148
    Beige        147
    Indigo       147
    Lavender     147
    Turquoise    145
    White        142
    Brown        141
    Gold         138
    Name: count, dtype: int64




python
plt.figure(figsize = (16, 6))
df["Color"].value_counts()[:10].sort_values(ascending = True).plot(kind = 'barh', color = sns.color_palette('tab20'), edgecolor = 'black')
plt.xlabel('Color', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('\nNumber of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
plt.xticks(rotation = 0, ha = 'center', fontsize = 16)
plt.tight_layout()
plt.show()



    
![png](output_17_0.png)
    



python
df["Season"].value_counts()





    Season
    Spring    999
    Fall      975
    Winter    971
    Summer    955
    Name: count, dtype: int64




python
plt.figure(figsize = (20, 6))

counts = df["Season"].value_counts()
explode = (0, 0, 0, 0)

counts.plot(kind = 'pie', fontsize = 12, colors = colors, explode = explode, autopct = '%1.1f%%')
plt.xlabel('Size', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()



    
![png](output_19_0.png)
    



python
df["Subscription Status"].value_counts()





    Subscription Status
    No     2847
    Yes    1053
    Name: count, dtype: int64




python
plt.figure(figsize = (20, 6))
ax = df["Subscription Status"].value_counts().plot(kind = 'bar', color = colors, rot = 0)
ax.set_xticklabels(('No', 'Yes'))

for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Subscription Status', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);



    
![png](output_21_0.png)
    



python
LE = LabelEncoder()
df["Frequency of Purchases"] = LE.fit_transform(df["Frequency of Purchases"])
CL = df[['Frequency of Purchases', 'Age',"Purchase Amount (USD)"]]




python
CL.describe().T





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
      <td>3,900.00</td>
      <td>2.97</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>6.00</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>3,900.00</td>
      <td>44.07</td>
      <td>15.21</td>
      <td>18.00</td>
      <td>31.00</td>
      <td>44.00</td>
      <td>57.00</td>
      <td>70.00</td>
    </tr>
    <tr>
      <th>Purchase Amount (USD)</th>
      <td>3,900.00</td>
      <td>59.76</td>
      <td>23.69</td>
      <td>20.00</td>
      <td>39.00</td>
      <td>60.00</td>
      <td>81.00</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>




python
CL.corr()





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
      <td>1.00</td>
      <td>0.02</td>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.02</td>
      <td>1.00</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>Purchase Amount (USD)</th>
      <td>-0.02</td>
      <td>-0.01</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




python
kmeans = KMeans(n_clusters=3,init="k-means++" )
kmeans.fit(CL)
y_kmeans = kmeans.predict(CL)
y_kmeans





    array([1, 0, 2, ..., 1, 2, 2])




python
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



    
![png](output_26_0.png)
    



python
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



    
![png](output_27_0.png)
    



python
ac = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
y_pred = ac.fit_predict(CL)
y_pred
plot_3d_clusters(y_pred,"Agglomerative")



    
![png](output_28_0.png)
    



python
dendogram = sch.dendrogram(sch.linkage(CL,method="ward"))
plt.show()



    
![png](output_29_0.png)
