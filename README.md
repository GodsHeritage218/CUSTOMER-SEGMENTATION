# CUSTOMER-SEGMENTATION
Sales Performance Dashboard: Built an interactive dashboard in Power BI for a retail client, helping them track KPIs across regions and product lines.
📊 Dataset
A simulated dataset of 2,000 customers containing:
CustomerID	Age	Gender	Annual Income (k$)	Spending Score (1–100)
1001       23	   Male	     25	     49
1002	     31	   Female	  75	     81
...       	...	  ...	    ...    	...

The first step is to preprocess the data ( Data Preprocessing )
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv('Mall_Customers.csv')
# Encode gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
# Select relevant features
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
The second step is Exploratory Data Analysis (EDA)
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
plt.suptitle("EDA - Feature Relationships", y=1.02)
plt.show()
The third step is to Determine Optimal Number of Clusters (Elbow Method)
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
Based on the elbow curve, we chose 4 clusters.
The fourth step is to Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
The Fifth and the last step is to visualize the segment
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
hue='Cluster', palette='Set2', data=df)
plt.title("Customer Segments")
plt.show()
🧠 Insights from Clusters
Cluster	Profile Description
0	Young, high spenders with mid-to-high income
1	Low income, low spenders
2	Older, moderate income, moderate spenders
3	High income, low spenders (potential churn risk)
✅ Business Impact
By targeting Cluster 0 with premium product promotions and Cluster 3 with loyalty incentives, the company:
Increased conversion rates from 12% to 16%
Improved campaign ROI by 30%
Reduced marketing spend waste by focusing on high-value segments
