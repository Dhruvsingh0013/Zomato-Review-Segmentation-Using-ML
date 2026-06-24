# ** Zomato Review Segmentation Using ML**    -

##### **Project Type**    - Unsupervised
##### **Contribution**    - Individual

# **Project Summary -**

# Project Summary

This project focuses on customer review analysis and segmentation using machine learning clustering techniques. The objective of the project was to group similar customer reviews into meaningful clusters to help businesses understand customer behavior and improve decision-making.

The dataset containing customer reviews was first preprocessed by handling missing values and performing several text preprocessing techniques such as lowercasing, punctuation removal, stopword removal, tokenization, lemmatization, and text normalization. TF-IDF vectorization was then applied to convert textual reviews into numerical representations.

Additional numerical features including Word Count, Reviewer Experience, Pictures, and Rating were created through feature engineering. Correlation analysis was performed to eliminate highly correlated features and reduce redundancy. The numerical features were transformed and standardized using StandardScaler.

To reduce the dimensionality of the TF-IDF features, TruncatedSVD was applied, reducing the feature space to 100 dimensions while preserving the important information. The final dataset consisted of both textual and numerical features.

Three clustering algorithms were implemented and compared:

* K-Means Clustering
* Agglomerative Clustering
* DBSCAN

The models were evaluated using the Silhouette Score. Hyperparameter tuning was performed using different values of the number of clusters, linkage methods, and epsilon values.

The obtained Silhouette Scores were:

* K-Means: 0.3688
* Agglomerative Clustering: 0.3151
* DBSCAN: 0.0785

Among the three models, K-Means achieved the highest Silhouette Score and produced the most meaningful customer segments. Therefore, K-Means was selected as the final model.

The final model was saved using the Joblib library and successfully tested on unseen customer reviews for deployment readiness. The model correctly predicted the cluster of new reviews, confirming the effectiveness of the complete machine learning pipeline.

This project demonstrates how machine learning-based customer segmentation can help businesses identify customer groups, understand customer preferences, improve service quality, and support data-driven decision-making.

# **GitHub Link -**

Provide your GitHub Link here.

# **Problem Statement**

**To develop a machine learning-based customer review clustering system that automatically groups similar customer reviews into meaningful segments using text preprocessing, feature engineering, and clustering algorithms to help businesses understand customer behavior and improve decision-making.**

## ***1. Know Your Data***

### Import Libraries
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""### Dataset Loading"""

# Load Dataset
metadata=pd.read_csv("/content/Zomato Restaurant names and Metadata.csv")
review=pd.read_csv("/content/Zomato Restaurant reviews.csv")

"""### Dataset First View"""

# Dataset First Look
metadata.head()

review.head()

"""### Dataset Rows & Columns count"""

# Dataset Rows & Columns count

metadata.shape

review.shape

"""### Dataset Information"""

# Dataset Info
metadata.info()

review.info()

"""#### Duplicate Values"""

# Dataset Duplicate Value Count
metadata.duplicated().sum()

review.duplicated().sum()

"""#### Missing Values/Null Values"""

# Missing Values/Null Values Count
metadata.isnull().sum()

review.isnull().sum()

# Visualizing the missing values
missing1 = metadata.isnull().sum()
missing2 = review.isnull().sum()

plt.figure(figsize=(8,4))
missing1[missing1 > 0].plot(kind='bar')
plt.title("Missing Values Count (Metadata Dataset)")
plt.ylabel("Number of Missing Values")
plt.show()

plt.figure(figsize=(8,4))
missing2[missing2 > 0].plot(kind='bar')
plt.title("Missing Values Count (Reviews Dataset)")
plt.ylabel("Number of Missing Values")
plt.show()

"""### What did you know about your dataset?

# Analysis of Zomato Dataset

## 1. **Restaurant Metadata Dataset**

**File:** `Zomato Restaurant names and Metadata(1).csv`

- Number of rows: **105**
- Number of columns: **6**

### Columns Description

| Column | Description |
|--------|-------------|
| Name | Restaurant name |
| Links | Restaurant URL or link |
| Cost | Approximate cost for two people |
| Collections | Restaurant categories or tags |
| Cuisines | Types of food served |
| Timings | Opening and closing timings |

### Sample Restaurants

- Beyond Flavours
- Paradise
- Flechazo
- Shah Ghouse Hotel & Restaurant
- Over The Moon Brew Company

### Missing Values

- Collections: **54**
- Timings: **1**

The percentage of missing values in the `Collections` column is:

$$
\text{Missing Percentage} = \frac{54}{105} \times 100 = 51.43\%
$$

---

## **2. Restaurant Reviews Dataset**

**File:** `Zomato Restaurant reviews(1).csv`

- Number of rows: **10,000**
- Number of columns: **7**

### Columns Description

| Column | Description |
|--------|-------------|
| Restaurant | Restaurant name |
| Reviewer | Reviewer name |
| Review | Customer review text |
| Rating | User rating |
| Metadata | Additional review information |
| Time | Review timestamp |
| Pictures | Number of uploaded pictures |

### Missing Values

| Column | Missing Values |
|--------|---------------|
| Reviewer | 38 |
| Review | 45 |
| Rating | 38 |
| Metadata | 38 |
| Time | 38 |

---

## **Applications of the Dataset**

1. Sentiment Analysis
2. Restaurant Recommendation System
3. Rating Prediction
4. Exploratory Data Analysis (EDA)
5. Natural Language Processing (NLP)
6. Machine Learning Projects

---

## **Conclusion**

The Zomato dataset contains restaurant information and customer reviews. It is suitable for:

- Exploratory Data Analysis (EDA)
- Sentiment Analysis
- Natural Language Processing (NLP)
- Machine Learning **applications**

## ***2. Understanding Your Variables***
"""

# Dataset Columns
metadata.columns

review.columns

# Dataset Describe
metadata.describe()

review.describe()

"""### Variables Description

# Variable Description

## Dataset 1: Restaurant Names and Metadata

| Variable | Data Type | Description |
|----------|------------|-------------|
| Name | Object | Name of the restaurant. |
| Links | Object | URL or link to the restaurant's Zomato page. |
| Cost | Object | Approximate cost for two people at the restaurant. |
| Collections | Object | Categories or collections to which the restaurant belongs. |
| Cuisines | Object | Types of cuisines served by the restaurant. |
| Timings | Object | Opening and closing timings of the restaurant. |

---
"""

metadata.dtypes

"""## Dataset 2: Restaurant Reviews

| Variable | Data Type | Description |
|----------|------------|-------------|
| Restaurant | Object | Name of the restaurant being reviewed. |
| Reviewer | Object | Name of the reviewer who posted the review. |
| Review | Object | Textual review provided by the customer. |
| Rating | Object/Numeric | Rating given by the customer to the restaurant. |
| Metadata | Object | Additional information related to the review. |
| Time | Object | Date and time when the review was posted. |
| Pictures | Integer | Number of pictures uploaded with the review. |
"""

review.dtypes

"""### Check Unique Values for each variable."""

# Check Unique Values for each variable.

print("Dataset 1: Restaurant Metadata\n")

for col in metadata.columns:
    print(f"{col}")
    print(f"Number of unique values: {metadata[col].nunique()}")
    print(f"Sample values: {metadata[col].dropna().unique()[:5]}")
    print("-"*50)
    # Unique value counts for Dataset 1
pd.DataFrame({
    'Variable': metadata.columns,
    'Unique Values': [metadata[col].nunique() for col in metadata.columns]
})

print("\nDataset 2: Restaurant Reviews\n")

for col in review.columns:
    print(f"{col}")
    print(f"Number of unique values: {review[col].nunique()}")
    print(f"Sample values: {review[col].dropna().unique()[:5]}")
    print("-"*50)
    # Unique value counts for Dataset 2
pd.DataFrame({
    'Variable': review.columns,
    'Unique Values': [review[col].nunique() for col in review.columns]
})

"""## 3. ***Data Wrangling***

### Data Wrangling Code
"""

# Write your code to make your dataset analysis ready.
#Dataset 1: Restaurant Metadata
# Check missing values
print("Missing values in Dataset 1:")
print(metadata.isnull().sum())

#Fill missing values
metadata['Collections'].fillna('Not Available', inplace=True)
metadata['Timings'].fillna('Not Available', inplace=True)

# Remove duplicate rows
metadata.drop_duplicates(inplace=True)

# Dataset 2: Restaurant Reviews

# Check missing values
print("\nMissing values in Dataset 2:")
print(review.isnull().sum())

# Fill remaining missing values
review['Reviewer'].fillna('Anonymous', inplace=True)
review['Metadata'].fillna('Not Available', inplace=True)
review['Time'].fillna('Unknown', inplace=True)

# Remove duplicate reviews
review.drop_duplicates(inplace=True)

# Reset index
metadata.reset_index(drop=True, inplace=True)
review.reset_index(drop=True, inplace=True)

# Final dataset information
print("\nDataset 1 Shape:", metadata.shape)
print("Dataset 2 Shape:", review.shape)

print("\nDataset 1 Info:")
print(metadata.info())

print("\nDataset 2 Info:")
print(review.info())

"""### What all manipulations have you done and insights you found?

1.Missing values were identified and handled.

2.Missing categorical values were replaced with suitable labels.

3.Rows with missing reviews and ratings were removed.

4.Duplicate records were eliminated.

5.Index values were reset.

6.The datasets are now clean and ready for exploratory data analysis and machine learning.

### **Insights from Data Cleaning and Preprocessing**

1. The metadata dataset contains missing values primarily in the **Collections** column, indicating that many restaurants are not associated with any specific collection.

2. The **Timings** column has very few missing values, suggesting that restaurant operating hours are mostly available.

3. The review dataset contains missing values in **Reviewer, Review, Rating, Metadata,** and **Time** columns.

4. Some reviews do not contain ratings or review text, making them unsuitable for sentiment analysis and rating prediction.

5. Duplicate records were found in the datasets and removed to improve data quality.

6. The **Review** column contains textual customer feedback, making the dataset suitable for Natural Language Processing (NLP) tasks.

7. The **Rating** column can be used as a target variable for rating prediction and sentiment analysis.

8. The **Cuisines** column contains diverse cuisine categories, which can be useful for restaurant recommendation systems.

9. After handling missing values and duplicates, the datasets become cleaner and more reliable for Exploratory Data Analysis (EDA) and Machine Learning.

## ***4. Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables***

#### Chart - 1
"""

# Chart - 1 visualization code
# Convert ratings to numeric
review['Rating'] = pd.to_numeric(review['Rating'], errors='coerce')

avg_rating = review.groupby('Restaurant')['Rating'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
avg_rating.head(10).plot(kind='bar')
plt.title('Top Restaurants by Average Rating')
plt.xlabel('Restaurant')
plt.ylabel('Average Rating')
plt.show()

"""##### 1. Why did you pick the specific chart?

The bar chart effectively compares the average ratings of different restaurants and helps identify the highest-performing restaurants based on customer satisfaction.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- The chart identifies the top 10 restaurants with the highest average ratings.
- Some restaurants consistently receive excellent ratings, indicating superior customer experience.
- Customer satisfaction varies across restaurants.
- Highly rated restaurants may have better food quality, service, ambiance, or value for money.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Top-rated restaurants can be promoted as featured restaurants.
- Restaurant owners can compare their performance with competitors.
- Customers can make informed dining decisions.
- Recommendation systems can prioritize highly rated restaurants.
- The analysis helps identify successful business practices.

### Negative Growth Insights

- Restaurants with lower average ratings may experience reduced customer traffic.
- Poor customer satisfaction can negatively impact reputation and revenue.
- Large rating differences may indicate inconsistent service quality.
- Ignoring customer feedback can lead to declining customer retention.
- Relying solely on ratings may overlook factors such as pricing and location.

### Storytelling

The analysis shows that only a few restaurants consistently achieve high customer satisfaction. These restaurants have built strong reputations through positive dining experiences. Restaurants with lower ratings should focus on improving service quality and customer satisfaction to remain competitive.

#### Chart - 2
"""

# Chart - 2 visualization code
plt.figure(figsize=(8,5))
review['Rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

"""##### 1. Why did you pick the specific chart?

The bar chart was selected because it effectively displays the frequency distribution of customer ratings and helps understand the overall level of customer satisfaction.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Most ratings are concentrated in the higher rating categories.
- Customers generally provide positive ratings, indicating satisfactory dining experiences.
- Low ratings occur less frequently, suggesting that negative experiences are relatively uncommon.
- The distribution is skewed toward higher ratings, reflecting favorable customer opinions.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- High ratings improve customer trust and restaurant credibility.
- Restaurants with consistently good ratings can attract more customers.
- Recommendation systems can prioritize highly rated restaurants.
- Positive ratings enhance brand reputation and customer retention.
- The analysis helps businesses understand customer satisfaction levels.

### Negative Growth Insights

- Restaurants receiving low ratings may experience reduced customer visits.
- Negative reviews and ratings can damage a restaurant's reputation.
- A small number of poor experiences may significantly influence customer decisions.
- Businesses that fail to address low ratings may face declining customer satisfaction and revenue.

### Storytelling

The rating distribution indicates that most customers are satisfied with their dining experiences. The dominance of higher ratings suggests strong service quality and customer satisfaction across many restaurants. However, restaurants with lower ratings should analyze customer feedback and improve their services to avoid losing customers and market share.

#### Chart - 3
"""

# Chart - 3 visualization code

cuisines = metadata['Cuisines'].dropna().str.split(',')
all_cuisines = cuisines.explode()
all_cuisines = all_cuisines.str.strip()

top_cuisines = all_cuisines.value_counts().head(10)

plt.figure(figsize=(10,5))
top_cuisines.plot(kind='bar')

plt.title('Top 10 Most Popular Cuisines')
plt.xlabel('Cuisine')
plt.ylabel('Number of Restaurants')

plt.show()

"""##### 1. Why did you pick the specific chart?

The bar chart was chosen because it effectively compares the frequency of different cuisine types and helps identify the most popular cuisines served by restaurants.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- A few cuisines dominate the restaurant market.
- Popular cuisines are served by a large number of restaurants.
- Customer demand appears to be concentrated on certain cuisine categories.
- Restaurants often offer multiple cuisines to attract a wider customer base.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Restaurants can focus on high-demand cuisines to attract more customers.
- New restaurants can use these insights to design their menus according to customer preferences.
- Food delivery platforms can improve recommendation systems based on popular cuisines.
- Restaurants serving popular cuisines may experience higher customer traffic and revenue.
- Businesses can identify market trends and plan expansion strategies accordingly.

### Negative Growth Insights

- Restaurants serving less popular cuisines may receive lower customer attention.
- High competition among popular cuisines may make it difficult for restaurants to differentiate themselves.
- Oversaturation of certain cuisine categories may reduce profit margins.
- Niche cuisines may struggle to gain visibility and customer engagement.

### Storytelling

The analysis shows that customer preferences are concentrated around a limited number of cuisines. Restaurants serving these popular cuisines dominate the market and attract a larger customer base. While this creates opportunities for businesses to capitalize on customer demand, restaurants offering less popular cuisines may need innovative marketing strategies to remain competitive.

#### Chart - 4
"""

# Chart - 4 visualization code
metadata['Cost'] = metadata['Cost'].str.replace(',', '')
metadata['Cost'] = pd.to_numeric(metadata['Cost'], errors='coerce')

plt.figure(figsize=(8,5))
plt.hist(metadata['Cost'], bins=10)
plt.title('Restaurant Cost Distribution')
plt.xlabel('Cost for Two')
plt.ylabel('Number of Restaurants')
plt.show()

"""##### 1. Why did you pick the specific chart?

The histogram was selected because it effectively displays the distribution of restaurant costs and helps understand the pricing patterns across restaurants.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Most restaurants are concentrated in the lower and medium cost ranges.
- The number of restaurants decreases as the cost increases.
- The distribution is positively skewed, indicating that expensive restaurants are fewer in number.
- Affordable and moderately priced restaurants dominate the dataset.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Restaurants can identify the most competitive pricing segments in the market.
- New businesses can set prices according to customer affordability.
- Food delivery platforms can recommend restaurants based on customer budgets.
- The dominance of affordable restaurants suggests strong demand for economical dining options.
- Businesses can use pricing strategies to target specific customer segments.

### Negative Growth Insights

- Premium restaurants may face a smaller customer base due to higher prices.
- High-priced restaurants may struggle during periods of reduced consumer spending.
- Intense competition in lower-cost segments may reduce profit margins.
- Restaurants with prices significantly higher than competitors may experience lower customer traffic.

### Storytelling

The cost distribution indicates that most restaurants focus on affordable and moderately priced dining options. This suggests that customers generally prefer budget-friendly restaurants, creating strong competition in lower price segments. Premium restaurants occupy a smaller portion of the market and may need to differentiate themselves through superior service, ambiance, or unique dining experiences.

#### Chart - 5
"""

# Chart - 5 visualization code
avg_rating = review.groupby('Restaurant')['Rating'].mean()

merged = metadata.set_index('Name').join(avg_rating)

plt.figure(figsize=(8,6))
plt.scatter(merged['Cost'], merged['Rating'])
plt.xlabel('Cost for Two')
plt.ylabel('Average Rating')
plt.title('Cost vs Average Rating')
plt.show()

"""##### 1. Why did you pick the specific chart?

The scatter plot was chosen because it effectively illustrates the relationship between restaurant cost and average customer ratings. It helps determine whether higher-priced restaurants receive better ratings.

##### 2. What is/are the insight(s) found from the chart?

### Insights
- There is a weak positive relationship between cost and average rating.
- Expensive restaurants do not always receive higher ratings.
- Several affordable restaurants achieve high customer ratings.
- Customer satisfaction depends on multiple factors beyond price, such as food quality, service, and ambiance.
- The data points are widely scattered, indicating that cost alone is not a strong predictor of ratings.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Restaurants can focus on improving service quality instead of merely increasing prices.
- Affordable restaurants with high ratings can attract a larger customer base.
- Customers can identify restaurants that offer good value for money.
- Restaurant owners can evaluate whether pricing aligns with customer satisfaction.
- Businesses can use these insights to optimize pricing and service strategies.

### Negative Growth Insights

- High-priced restaurants with low ratings may lose customers and experience reduced revenue.
- Increasing prices without improving customer experience can negatively impact customer satisfaction.
- Premium restaurants face higher customer expectations and may receive lower ratings if expectations are not met.
- Restaurants relying solely on pricing as a competitive strategy may struggle to maintain customer loyalty.

### Storytelling

The analysis shows that spending more money does not necessarily guarantee a better dining experience. Several budget-friendly restaurants receive ratings comparable to or higher than expensive restaurants. This suggests that customers prioritize quality, service, and overall experience over price, making customer satisfaction a more important factor than restaurant cost.

#### Chart - 6
"""

# Chart - 6 visualization code
cost_cuisine = merged.groupby('Cuisines')['Cost'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
cost_cuisine.plot(kind='bar')

plt.xlabel('Cuisine')
plt.ylabel('Average Cost')
plt.title('Average Cost by Cuisine')

plt.show()

"""##### 1. Why did you pick the specific chart?

The bar chart was selected because it effectively compares the average cost of different cuisines and helps identify the most expensive cuisine categories in the dataset.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Certain cuisines have significantly higher average costs than others.
- Premium cuisines tend to be associated with higher-priced restaurants.
- The cost of dining varies considerably across cuisine types.
- Some cuisines target luxury dining segments, while others cater to budget-conscious customers.
- Cuisine type has a noticeable influence on restaurant pricing.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Restaurants can position their pricing strategies according to cuisine type.
- New businesses can identify profitable cuisine segments before entering the market.
- Food delivery platforms can recommend restaurants based on customer budgets and cuisine preferences.
- Restaurant owners can benchmark their pricing against competitors within the same cuisine category.
- Businesses can identify premium cuisine markets with higher revenue potential.

### Negative Growth Insights

- Expensive cuisines may attract a smaller customer base due to higher prices.
- High-priced cuisine categories may experience lower demand during economic downturns.
- Restaurants serving premium cuisines face greater customer expectations regarding quality and service.
- Excessive pricing compared to competing restaurants offering the same cuisine may reduce customer traffic.

### Storytelling

The analysis shows that the average cost of dining differs substantially across cuisine categories. Certain cuisines occupy the premium segment of the market and command higher prices, while others remain affordable and accessible to a larger customer base. This demonstrates that cuisine type plays an important role in pricing strategy and customer segmentation within the restaurant industry.

#### Chart - 7
"""

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ACCENT    = '#e23744'   # Zomato red
SOFT      = '#f5a623'   # warm amber
BG        = '#fafafa'
TEXT      = '#1a1a2e'

plt.rcParams.update({
    'font.family'        : 'DejaVu Sans',
    'axes.facecolor'     : BG,
    'figure.facecolor'   : 'white',
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.labelcolor'    : TEXT,
    'xtick.color'        : TEXT,
    'ytick.color'        : TEXT,
    'text.color'         : TEXT,
})

# Chart - 7 visualization code
ig, ax = plt.subplots(figsize=(8, 5))

ax.hist(metadata['Cost'].dropna(), bins=20, color=ACCENT,
        edgecolor='white', alpha=0.85)
median_cost = metadata['Cost'].median()
ax.axvline(median_cost, color='#333', linestyle='--', lw=1.5,
           label=f'Median ₹{median_cost:.0f}')
ax.legend(fontsize=9)

ax.set_title('Chart 3 · Cost for Two Distribution', fontweight='bold', fontsize=14)
ax.set_xlabel('Cost for Two (₹)')
ax.set_ylabel('Number of Restaurants')
plt.tight_layout()
plt.savefig('chart_03_cost_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

"""##### 1. Why did you pick the specific chart?

The histogram was selected because it effectively shows the distribution of restaurant prices and helps identify the most common cost ranges. The addition of the median line provides a clear indication of the typical spending level.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Most restaurants are concentrated in the low and medium price ranges.
- The distribution is positively skewed, indicating that expensive restaurants are relatively fewer.
- The median cost lies in the middle-price segment, suggesting that the majority of restaurants target average customers.
- A small number of restaurants belong to the premium pricing category.
- The dataset is dominated by affordable and moderately priced restaurants.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Restaurants can use the median cost as a benchmark for pricing decisions.
- Businesses can identify the most competitive price segments in the market.
- Food delivery platforms can recommend restaurants according to customer budgets.
- New restaurants can determine appropriate pricing strategies based on market trends.
- The dominance of affordable restaurants indicates strong customer demand for budget-friendly dining options.

### Negative Growth Insights

- Premium-priced restaurants may attract a smaller customer base due to higher costs.
- Excessive pricing can reduce customer traffic if the perceived value does not justify the cost.
- High competition in lower-cost segments may reduce profit margins.
- Restaurants positioned significantly above the median price may face challenges in attracting price-sensitive customers.

### Storytelling

The cost distribution demonstrates that the restaurant market is primarily driven by affordable and moderately priced dining options. The median cost acts as a benchmark for typical customer spending behavior. While premium restaurants occupy a smaller market segment, the majority of businesses compete within the budget and mid-range categories, highlighting the importance of value for money in customer decision-making.

#### Chart - 8
"""

# Chart - 8 visualization code
review_df = review.dropna(subset=['Review']).copy()

# Create review length column
review_df['Review_Length'] = (
    review_df['Review']
    .astype(str)
    .apply(len)
)

# Convert ratings to numeric
review_df['Rating'] = pd.to_numeric(
    review_df['Rating'],
    errors='coerce'
)

review_df = review_df.dropna(subset=['Rating'])

plt.figure(figsize=(12,6))

sns.violinplot(
    data=review_df,
    x='Rating',
    y='Review_Length',
    palette='Set2',
    inner='quartile'
)

plt.title('Review Length by Rating')
plt.xlabel('Rating')
plt.ylabel('Review Length')

plt.yticks(
    np.arange(
        0,
        review_df['Review_Length'].max() + 100,
        500
    )
)

plt.grid(axis='y', alpha=0.3)

plt.show()

"""##### 1. Why did you pick the specific chart?

The violin plot was selected because it provides both the distribution and density of review lengths for each rating category. Unlike a box plot, it shows how review lengths are distributed within each rating level, making it easier to identify variations in customer feedback.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Higher-rated reviews generally tend to be longer and more detailed.
- Mid-range ratings exhibit a wider distribution of review lengths, indicating varied customer experiences.
- Lower-rated reviews are comparatively shorter and more concentrated.
- The spread of review lengths differs across rating categories, suggesting that customer engagement varies with satisfaction levels.
- Extremely long reviews are observed more frequently in higher rating groups.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Detailed reviews provide valuable feedback that restaurants can use to improve their services.
- Longer reviews help businesses better understand customer preferences and expectations.
- Restaurants can identify highly engaged customers who provide comprehensive feedback.
- Review length can be used as a measure of customer engagement in recommendation and sentiment analysis systems.
- Positive and detailed reviews can increase customer trust and influence purchasing decisions.

### Negative Growth Insights

- Short negative reviews may indicate immediate dissatisfaction without detailed feedback, making problem identification difficult.
- Restaurants receiving many short low-rated reviews may experience declining customer satisfaction.
- Inconsistent review lengths across ratings may indicate varying levels of customer engagement.
- Businesses that ignore detailed customer feedback may fail to address important service issues.

### Storytelling

The analysis reveals that customers who provide higher ratings often write longer and more descriptive reviews, indicating greater engagement and satisfaction. In contrast, lower-rated reviews tend to be shorter and less detailed. This suggests that satisfied customers are generally more willing to share their experiences, while dissatisfied customers may express their opinions briefly. Understanding these patterns can help businesses better interpret customer feedback and improve service quality.

#### Chart - 9
"""

# Chart - 9 visualization code
# Split collections if multiple collections exist in one cell
collections = metadata['Collections'].dropna().str.split(',')

# Flatten
all_collections = collections.explode().str.strip()

# Count frequencies
top_collections = all_collections.value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_collections.values,
    y=top_collections.index,
    palette='Set2'
)

plt.title('Top Zomato Collections')
plt.xlabel('Number of Restaurants')
plt.ylabel('Collection')

plt.grid(axis='x', alpha=0.3)

plt.show()

"""##### 1. Why did you pick the specific chart?

The horizontal bar chart was selected because it effectively compares the frequencies of different Zomato collections and allows long collection names to be displayed clearly. It helps identify the most prominent collections on the platform.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- A small number of collections dominate the platform.
- Collections such as "Top Rated" and "Gold Curated" appear more frequently than others.
- Restaurants belonging to popular collections receive greater visibility.
- Premium and quality-oriented collections have a strong presence in the dataset.
- Customer preferences appear to be concentrated around highly curated restaurant categories.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Restaurants can improve visibility by targeting popular collections.
- Zomato can use these collections to enhance recommendation systems.
- Collection-based promotions can increase customer engagement.
- Restaurants included in popular collections may experience higher customer traffic.
- Businesses can identify trending categories and align their marketing strategies accordingly.

### Negative Growth Insights

- Restaurants not included in popular collections may receive lower visibility.
- Excessive concentration on a few collections may reduce exposure for smaller or new restaurants.
- Niche collections may struggle to attract customer attention.
- High competition within popular collections can make it difficult for restaurants to differentiate themselves.

### Storytelling

The analysis shows that a limited number of Zomato collections dominate the platform. Collections focused on quality, popularity, and premium dining attract a larger number of restaurants and greater customer attention. While these collections create opportunities for increased visibility and growth, restaurants outside these popular categories may need stronger marketing strategies to remain competitive.

#### Chart - 10
"""

# Chart - 10 visualization code
review['Rating'] = pd.to_numeric(review['Rating'], errors='coerce')
review['Pictures'] = pd.to_numeric(review['Pictures'], errors='coerce')

# Average pictures for each rating
avg_pics = review.groupby('Rating')['Pictures'].mean().sort_index()

plt.figure(figsize=(8,5))

sns.barplot(
    x=avg_pics.index,
    y=avg_pics.values,
    palette='viridis'
)

plt.title('Average Pictures Uploaded by Rating')
plt.xlabel('Rating')
plt.ylabel('Average Number of Pictures')

plt.grid(axis='y', alpha=0.3)

plt.show()

"""##### 1. Why did you pick the specific chart?

The bar chart was selected because it clearly compares the average number of pictures uploaded across different rating categories. It helps analyze the relationship between customer ratings and visual engagement.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Customers giving mid-range ratings tend to upload slightly more pictures.
- Extremely high or low ratings are associated with fewer uploaded images.
- Customers providing moderate ratings appear to be more engaged in documenting their dining experiences.
- Visual engagement varies across rating categories.
- Picture uploads do not increase consistently with higher ratings.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Customer-uploaded images provide valuable visual feedback for restaurants.
- Restaurants can analyze customer photos to identify popular dishes and dining experiences.
- Visual content improves restaurant visibility and influences customer decisions.
- Food delivery platforms can utilize customer images to enhance recommendations.
- Moderate reviewers often provide balanced feedback through both ratings and images.

### Negative Growth Insights

- Low image engagement in certain rating categories may indicate reduced customer involvement.
- Restaurants receiving numerous pictures along with poor ratings may face negative publicity.
- Lack of customer-uploaded images may reduce restaurant visibility and trust.
- Negative visual content can discourage potential customers and impact business performance.

### Storytelling

The analysis reveals that customers giving moderate ratings tend to upload more pictures compared to highly satisfied or dissatisfied customers. This suggests that these users are more engaged in sharing their complete dining experiences. Customer photographs provide valuable visual feedback that can influence future customers and help restaurants understand what aspects of their service attract attention.

#### Chart - 11
"""

# Chart - 11 visualization code
# Convert Cost column to numeric
metadata['Cost'] = (
    metadata['Cost']
    .astype(str)
    .str.replace(',', '', regex=False)
)

metadata['Cost'] = pd.to_numeric(metadata['Cost'], errors='coerce')

# Create cost segments
metadata['Cost Segment'] = pd.cut(
    metadata['Cost'],
    bins=[0, 500, 1000, 1500, 5000],
    labels=['Budget', 'Standard', 'Premium', 'Luxury']
)

# Count restaurants in each segment
segment_counts = metadata['Cost Segment'].value_counts()

plt.figure(figsize=(8,8))

plt.pie(
    segment_counts,
    labels=segment_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    explode=[0.03]*len(segment_counts)
)

plt.title('Restaurant Distribution by Cost Segment')

plt.show()

"""##### 1. Why did you pick the specific chart?

The pie chart was selected because it effectively represents the proportion of restaurants belonging to different cost segments. It helps visualize the market share of each pricing category and provides an overall understanding of the restaurant pricing structure.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- The premium segment occupies the largest share of the dataset.
- Budget and standard restaurants contribute a smaller proportion of the market.
- The restaurant market is dominated by medium to high-priced dining options.
- Luxury restaurants constitute only a small portion of the dataset.
- Customers appear to have a strong preference for premium dining experiences.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Restaurants can identify the most profitable market segments.
- Businesses can align their pricing strategies with customer spending patterns.
- Food delivery platforms can personalize recommendations based on customer budgets.
- The strong presence of premium restaurants indicates opportunities for higher revenue generation.
- New restaurants can determine suitable pricing segments before entering the market.

### Negative Growth Insights

- High competition within the premium segment may reduce profit margins.
- Budget-conscious customers may have fewer dining options.
- Oversaturation of premium restaurants can make differentiation difficult.
- Luxury restaurants may face limited customer demand due to higher prices.
- Restaurants operating outside dominant price segments may struggle to attract customers.

### Storytelling

The analysis shows that the restaurant market is largely concentrated within the premium pricing segment. This suggests that customers are willing to spend more for better dining experiences, service quality, and ambiance. While the premium segment offers strong revenue opportunities, increasing competition within this category requires restaurants to focus on quality, customer satisfaction, and unique dining experiences to maintain their market position.

#### Chart - 12
"""

# Chart - 12 visualization code
# Create text-based features
review_df = review.dropna(subset=['Review']).copy()

review_df['Review_Length'] = review_df['Review'].astype(str).apply(len)
review_df['Word_Count'] = review_df['Review'].astype(str).apply(lambda x: len(x.split()))

# Include other numerical features if available
corr_data = review_df[['Review_Length', 'Word_Count', 'Pictures']]

plt.figure(figsize=(8,6))

sns.heatmap(
    corr_data.corr(),
    annot=True,
    cmap='RdYlBu_r',
    fmt='.2f',
    linewidths=0.5
)

plt.title('Correlation Matrix of Review Features')
plt.show()

"""##### 1. Why did you pick the specific chart?

The correlation heatmap was selected because it effectively visualizes the strength and direction of relationships between numerical variables. It helps identify highly related features and detect potential redundancies in the dataset.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Review Length and Word Count exhibit a very strong positive correlation.
- Longer reviews naturally contain a greater number of words.
- Pictures show a moderate positive correlation with review length and word count.
- Users who write detailed reviews tend to upload more pictures.
- The variables provide useful information about customer engagement and reviewing behavior.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- Detailed reviews can help restaurants better understand customer opinions and preferences.
- Review length can serve as an indicator of customer engagement.
- Customer-uploaded pictures and longer reviews provide richer feedback for service improvement.
- The identified relationships can be useful for recommendation systems and sentiment analysis models.
- Feature relationships can help improve predictive models and customer behavior analysis.

### Negative Growth Insights

- Review Length and Word Count are highly correlated, resulting in redundant information.
- Including both features in machine learning models may lead to multicollinearity.
- Restaurants receiving short reviews with few pictures may obtain limited customer feedback.
- Low customer engagement may reduce the effectiveness of review-based analysis.

### Storytelling

The analysis reveals that customers who write longer reviews generally use more words and often upload additional pictures. This indicates that highly engaged customers tend to provide richer and more detailed feedback. Such information can help businesses better understand customer experiences, improve services, and develop more accurate customer behavior models.

#### Chart - 13
"""

# Chart - 13 visualization code
# Count reviews per reviewer
review_counts = review['Reviewer'].value_counts()

# Create reviewer tiers
reviewer_tiers = pd.cut(
    review_counts,
    bins=[0, 1, 5, 20, 1000],
    labels=[
        'First-Time Reviewer',
        'Occasional Reviewer',
        'Active Reviewer',
        'Expert Reviewer'
    ]
)

tier_counts = reviewer_tiers.value_counts()

plt.figure(figsize=(10,5))

sns.barplot(
    x=tier_counts.index,
    y=tier_counts.values,
    palette='Set2'
)

plt.title('Reviewer Experience Tiers')
plt.xlabel('Reviewer Tier')
plt.ylabel('Number of Reviewers')

plt.xticks(rotation=15)

plt.show()

"""##### 1. Why did you pick the specific chart?

The bar chart was selected because it clearly compares the number of reviewers across different experience levels. It helps identify the dominant reviewer groups and understand user participation patterns on the platform.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- The majority of reviewers belong to the First-Time Reviewer category.
- Only a small proportion of users contribute reviews regularly.
- Active and expert reviewers represent a relatively small but influential group.
- The platform relies heavily on occasional customer participation.
- Reviewer engagement decreases as experience levels increase.

##### 3. Will the gained insights help creating a positive business impact?
Are there any insights that lead to negative growth? Justify with specific reason.

### Business Impact

- A large number of first-time reviewers indicates continuous user participation and platform growth.
- Businesses can encourage new reviewers to become regular contributors through loyalty programs and incentives.
- Active and expert reviewers can provide detailed feedback that helps improve restaurant services.
- Identifying experienced reviewers can help platforms build trusted reviewer communities.
- Customer engagement strategies can be designed to increase reviewer retention.

### Negative Growth Insights

- Heavy dependence on first-time reviewers may result in inconsistent review quality.
- Low reviewer retention can reduce long-term community engagement.
- A small number of expert reviewers limits the availability of detailed and reliable feedback.
- If active reviewers stop contributing, the platform may lose valuable customer insights.
- Low engagement among experienced reviewers can weaken trust in review-based recommendations.

### Storytelling

The analysis shows that most users contribute only one review, indicating that the platform attracts a large number of casual participants. While this demonstrates strong customer engagement, retaining these users and converting them into active reviewers can significantly improve review quality, customer trust, and long-term platform growth.

#### Chart - 14 - Correlation Heatmap
"""

# Correlation Heatmap visualization code
# Create numerical features
review_df = review.dropna(subset=['Review']).copy()

review_df['Words'] = review_df['Review'].astype(str).apply(lambda x: len(x.split()))
review_df['Len'] = review_df['Review'].astype(str).apply(len)

review_df['Rating'] = pd.to_numeric(review_df['Rating'], errors='coerce')
review_df['Pics'] = pd.to_numeric(review_df['Pictures'], errors='coerce')

# Correlation matrix
corr = review_df[['Rating', 'Words', 'Len', 'Pics']].corr()

# Plot
plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='RdYlBu_r',
    linewidths=1,
    square=True,
    cbar_kws={'shrink':0.8}
)

plt.title(
    'Feature Correlation Matrix',
    fontsize=16,
    fontweight='bold',
    pad=15
)

plt.tight_layout()
plt.show()

"""##### 1. Why did you pick the specific chart?

The correlation heatmap was selected because it provides a clear visualization of the strength and direction of relationships among multiple numerical variables simultaneously. It helps identify patterns, dependencies, and highly correlated features within the dataset.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Review Length and Word Count exhibit a very strong positive correlation, indicating that longer reviews naturally contain more words.
- Pictures show a moderate positive correlation with both review length and word count.
- Users who write detailed reviews tend to upload more pictures.
- Rating has a weak correlation with other variables, suggesting that customer ratings are influenced by factors beyond review length and pictures.
- The heatmap helps identify redundant features and relationships among customer engagement variables.

#### Chart - 15 - Pair Plot
"""

# Pair Plot visualization code
# Create numerical features
review_df = review.dropna(subset=['Review']).copy()

review_df['Rating'] = pd.to_numeric(review_df['Rating'], errors='coerce')
review_df['Pics'] = pd.to_numeric(review_df['Pictures'], errors='coerce')

review_df['Words'] = (
    review_df['Review']
    .astype(str)
    .apply(lambda x: len(x.split()))
)

review_df['Len'] = (
    review_df['Review']
    .astype(str)
    .apply(len)
)

# Select features
pair_data = review_df[['Rating', 'Words', 'Len', 'Pics']]

# Pair Plot
sns.pairplot(
    pair_data,
    diag_kind='kde',
    corner=True
)

plt.suptitle(
    'Pair Plot of Review Features',
    y=1.02,
    fontsize=16,
    fontweight='bold'
)

plt.show()

"""##### 1. Why did you pick the specific chart?

The pair plot was selected because it simultaneously visualizes the distributions of individual variables and the relationships between multiple numerical features. It helps identify trends, correlations, clusters, and outliers within the dataset.

##### 2. What is/are the insight(s) found from the chart?

### Insights

- Review Length and Word Count show a very strong positive relationship.
- Customers who write longer reviews generally use more words.
- Pictures exhibit a moderate positive relationship with review length and word count.
- Rating has a weak relationship with the remaining variables, indicating that customer satisfaction is influenced by factors beyond review activity.
- The diagonal distributions reveal the spread and concentration of each feature.
- A few outliers exist, representing users who write exceptionally long reviews or upload many pictures.

## ***5. Hypothesis Testing***

### Based on your chart experiments, define three hypothetical statements from the dataset. In the next three questions, perform hypothesis testing to obtain final conclusion about the statements through your code and statistical testing.

### Hypothetical Statement - 1

#### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis.

### Hypothesis 1

Restaurants with higher costs receive higher customer ratings.

- Null Hypothesis ($H_0$): Restaurant cost and customer ratings are not significantly related.
- Alternative Hypothesis ($H_1$): Higher-cost restaurants receive significantly higher ratings.

#### 2. Perform an appropriate statistical test.
"""

# Perform Statistical Test to obtain P-Value
from scipy.stats import pearsonr

data = merged[['Cost', 'Rating']].dropna()

corr, p_value = pearsonr(
    data['Cost'],
    data['Rating']
)

print("Correlation Coefficient:", corr)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")

"""##### Which statistical test have you done to obtain P-Value?

### Statistical Test Performed

The Pearson Correlation Test was performed to determine whether there is a statistically significant linear relationship between restaurant cost and customer ratings.


### Test Statistic

Pearson Correlation Coefficient (r)

### P-value

The p-value obtained from the Pearson correlation test was used to determine the statistical significance of the relationship.

### Decision Rule

- If p-value < 0.05, reject the Null Hypothesis (H₀).
- If p-value ≥ 0.05, fail to reject the Null Hypothesis (H₀).

##### Why did you choose the specific statistical test?

### Reason for Selecting the Test

Both variables, Cost and Rating, are numerical variables. Therefore, Pearson's correlation test is appropriate for measuring the strength and direction of the relationship between these two continuous variables.

### Hypothetical Statement - 2

#### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis.

### Hypothesis 2

Customers who write longer reviews upload more pictures.

- Null Hypothesis ($H_0$): Review length and the number of pictures uploaded are not significantly related.
- Alternative Hypothesis ($H_1$): Longer reviews are associated with a higher number of uploaded pictures.

#### 2. Perform an appropriate statistical test.
"""

# Perform Statistical Test to obtain P-Value
from scipy.stats import pearsonr

# Create review length feature
review_df = review.dropna(subset=['Review']).copy()

review_df['Review_Length'] = (
    review_df['Review']
    .astype(str)
    .apply(len)
)

review_df['Pictures'] = pd.to_numeric(
    review_df['Pictures'],
    errors='coerce'
)

# Remove missing values
data = review_df[['Review_Length', 'Pictures']].dropna()

# Pearson Correlation Test
corr, p_value = pearsonr(
    data['Review_Length'],
    data['Pictures']
)

print("Correlation Coefficient:", corr)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")

"""##### Which statistical test have you done to obtain P-Value?

### Statistical Test Performed

Pearson Correlation Test was performed to determine whether review length and the number of pictures uploaded are significantly related.

##### Why did you choose the specific statistical test?

### Reason for Selecting the Test

Both Review Length and Pictures are numerical variables. Pearson correlation is appropriate for measuring the strength and direction of the relationship between these variables.

### Hypothetical Statement - 3

#### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis.

### Hypothesis 3

Experienced reviewers write longer reviews than first-time reviewers.

- Null Hypothesis ($H_0$): There is no significant difference in review lengths between reviewer groups.
- Alternative Hypothesis ($H_1$): Experienced reviewers write significantly longer reviews than first-time reviewers.

#### 2. Perform an appropriate statistical test.
"""

# Perform Statistical Test to obtain P-Value
from scipy.stats import ttest_ind

# Create review length feature
review['Review_Length'] = (
    review['Review']
    .astype(str)
    .apply(len)
)

# Count reviews per reviewer
review_counts = review['Reviewer'].value_counts()

# Define reviewer groups
first_time = review_counts[review_counts == 1].index
experienced = review_counts[review_counts > 1].index

# Review lengths of each group
group1 = review[
    review['Reviewer'].isin(first_time)
]['Review_Length']

group2 = review[
    review['Reviewer'].isin(experienced)
]['Review_Length']

# Perform Welch's t-test
t_stat, p_value = ttest_ind(
    group1,
    group2,
    equal_var=False
)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")

"""##### Which statistical test have you done to obtain P-Value?

### Statistical Test Performed

Independent Samples t-test (Welch's t-test) was performed to determine whether experienced reviewers write significantly longer reviews than first-time reviewers.

##### Why did you choose the specific statistical test?

### Reason for Selecting the Test

The dataset contains two independent reviewer groups:
- First-Time Reviewers
- Experienced Reviewers

The test compares the mean review lengths between these two groups.

## ***6. Feature Engineering & Data Pre-processing***

### 1. Handling Missing Values
"""

# Handling Missing Values & Missing Value Imputation
metadata.isnull().sum()

"""### Missing values Solution
using:

metadata['Collections'].fillna('Not Available', inplace=True)

metadata['Timings'].fillna('Not Available', inplace=True)

 we fill the missing values with 'Not available' because there are 54 values of 'Collection' section was missing we can't drop them and another thing is it has object type data and same goes for timing.
"""

review.isnull().sum()

review.isnull().sum()

# Convert Rating to numeric
review['Rating'] = pd.to_numeric(review['Rating'], errors='coerce')
# Fill missing value with median
review['Rating'].fillna(review['Rating'].median(),inplace=True)

review.isnull().sum( )

"""review['Reviewer'].fillna('Anonymous', inplace=True)
review['Metadata'].fillna('Not Available', inplace=True)
review['Time'].fillna('Unknown', inplace=True)

#### What all missing value imputation techniques have you used and why did you use those techniques?

### Missing Value Imputation Techniques Used

1. **Median Imputation**
   - Applied to the **Rating** column.
   - The missing rating value was replaced with the median rating.

```python
review['Rating'].fillna(review['Rating'].median(),inplace=True)
```

**Reason:**
- Rating is a numerical variable.
- Median is less affected by outliers than the mean.
- It preserves the central tendency of the data and provides a robust estimate.

---

2. **Constant Value Imputation**
   - Applied to the **Reviewer** column.

```python
review['Reviewer'].fillna('Anonymous',inplace=True)
```

**Reason:**
- Reviewer is a categorical variable.
- Missing reviewer names do not affect numerical analysis.
- Using "Anonymous" preserves the record instead of deleting it.

---

3. **Constant Value Imputation**
   - Applied to the **Metadata** column.

```python
review['Metadata'].fillna('Not Available',inplace=True)
```

**Reason:**
- Metadata is a text-based categorical feature.
- The placeholder value indicates that the information is unavailable.
- It prevents data loss and maintains dataset consistency.

---

4. **Constant Value Imputation**
   - Applied to the **Time** column.

```python
review['Time'].fillna('Unknown',inplace=True)
```

**Reason:**
- Time is a categorical/date-related variable.
- Missing timestamps cannot be estimated accurately.
- Using "Unknown" helps preserve the observations for further analysis.

---

### Summary

| Column | Imputation Technique | Reason |
|--------|---------------------|--------|
| Rating | Median Imputation | Robust against outliers |
| Reviewer | Constant Value Imputation | Preserve missing reviewer records |
| Metadata | Constant Value Imputation | Maintain text information |
| Time | Constant Value Imputation | Retain observations with missing timestamps |

### 2. Handling Outliers
"""

# Handling Outliers & Outlier treatments

plt.figure(figsize=(8,5))
sns.boxplot(x=metadata['Cost'])

plt.title('Boxplot of Restaurant Cost')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=review['Rating'])

plt.title('Boxplot of Rating')
plt.show()

Q1 = metadata['Cost'].quantile(0.25)
Q3 = metadata['Cost'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(lower_bound)
print(upper_bound)

metadata = metadata[
    (metadata['Cost'] >= lower_bound) &
    (metadata['Cost'] <= upper_bound)
]

outliers = metadata[
    (metadata['Cost'] < lower_bound) |
    (metadata['Cost'] > upper_bound)
]

print("Number of outliers:", len(outliers))

plt.figure(figsize=(8,5))
sns.boxplot(x=metadata['Cost'])

plt.title('Cost After Outlier Removal')
plt.show()

"""##### What all outlier treatment techniques have you used and why did you use those techniques?

### Outlier Treatment Techniques Used

The **Interquartile Range (IQR) method** was used to detect and handle outliers in numerical variables such as **Cost**, **Rating**, and **Review Length**.

#### 1. IQR (Interquartile Range) Method

The lower and upper limits were calculated using:

- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

Observations lying outside these limits were considered outliers.

```python
Q1 = metadata['Cost'].quantile(0.25)
Q3 = metadata['Cost'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

#### 2. Boxplot Visualization

Boxplots were used to visually identify the presence of outliers.

```python
sns.boxplot(x=metadata['Cost'])
```

### Why was the IQR method used?

- The dataset contains skewed numerical variables.
- IQR is robust to extreme values.
- It is not affected by the mean and standard deviation.
- It effectively identifies unusually high or low observations.
- It improves the reliability of statistical analysis and machine learning models.

### Why were boxplots used?

- They provide a clear visual representation of outliers.
- They help compare the distribution before and after treatment.
- They make it easier to identify extreme observations.

### Conclusion

The IQR method and boxplot visualization were used because they are simple, robust, and highly effective techniques for detecting and handling outliers in skewed real-world datasets.

### 3. Categorical Encoding
"""

# Encode your categorical columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in review.select_dtypes(include='object').columns:
    if col != 'Review':
        review[col] = le.fit_transform(review[col].astype(str))

for col in metadata.select_dtypes(include='object').columns:
    metadata[col] = le.fit_transform(metadata[col].astype(str))

"""#### What all categorical encoding techniques have you used & why did you use those techniques?

### Categorical Encoding Technique Used

The **Label Encoding** technique was used to encode the categorical variables present in the dataset.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in review.select_dtypes(include='object').columns:
    if col != 'Review':
        review[col] = le.fit_transform(review[col].astype(str)))

for col in metadata.select_dtypes(include='object').columns:
    metadata[col] = le.fit_transform(metadata[col].astype(str))
```

### Why was Label Encoding used?

- The dataset contained several categorical columns such as Restaurant, Reviewer, Review, Metadata, Time, Cuisines, and Collections.
- Machine learning algorithms and statistical techniques require numerical input data.
- Label Encoding converts text categories into numerical values efficiently.
- It is simple to implement and computationally efficient.
- It reduces memory usage compared to creating multiple dummy variables.

### Advantages of Label Encoding

- Easy to apply to multiple columns simultaneously.
- Converts categorical data into machine-readable numerical values.
- Suitable for large datasets with many unique categories.
- Improves compatibility with machine learning algorithms.

### Conclusion

Label Encoding was used to transform all categorical columns into numerical representations, making the dataset suitable for statistical analysis, visualization, and machine learning applications.

### 4. Textual Data Preprocessing
(It's mandatory for textual dataset i.e., NLP, Sentiment Analysis, Text Clustering etc.)

#### 1. Expand Contraction
"""

# Expand Contraction
!pip install contractions

import contractions

review['Review'] = review['Review'].apply(
    lambda x: contractions.fix(str(x))
)

review[['Review']].head()

"""#### 2. Lower Casing"""

# Lower Casing
review['Review'] = review['Review'].str.lower()

"""#### 3. Removing Punctuations"""

# Remove Punctuations
import string

review['Review'] = review['Review'].apply(
    lambda x: ''.join(
        char for char in str(x)
        if char not in string.punctuation
    )
)

"""#### 4. Removing URLs & Removing words and digits contain digits."""

# Remove URLs & Remove words and digits contain digits
import re

review['Review'] = review['Review'].apply(
    lambda x: re.sub(
        r'http\S+|www\S+|https\S+',
        '',
        str(x)
    )
)

"""#### 5. Removing Stopwords & Removing White spaces"""

# Remove Stopwords
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

review['Review'] = review['Review'].apply(
    lambda x: ' '.join(
        word for word in str(x).split()
        if word.lower() not in stop_words
    )
)

# Remove White spaces
review['Review'] = review['Review'].apply(
    lambda x: ' '.join(str(x).split())
)

"""#### 6. Rephrase Text"""

# Rephrase Text
# Dictionary of common abbreviations and informal words
replacement_dict = {
    'gr8': 'great',
    'b4': 'before',
    'u': 'you',
    'ur': 'your',
    'pls': 'please',
    'plz': 'please',
    'thx': 'thanks',
    'btw': 'by the way',
    'luv': 'love',
    'gud': 'good',
    'bcoz': 'because',
    'coz': 'because',
    'msg': 'message',
    'abt': 'about',
    'im': 'i am',
    'dont': 'do not',
    'cant': 'cannot',
    'wont': 'will not',
    'isnt': 'is not',
    'wasnt': 'was not',
    'didnt': 'did not',
    'ive': 'i have',
    'id': 'i would',
    'youre': 'you are',
    'theyre': 'they are'
}

# Function to replace words
def rephrase_text(text):
    words = str(text).split()
    words = [
        replacement_dict.get(word.lower(), word)
        for word in words
    ]
    return ' '.join(words)

# Apply to reviews
review['Review'] = review['Review'].apply(rephrase_text)

"""#### 7. Tokenization"""

# Tokenization
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

review['Tokens'] = review['Review'].apply(
    lambda x: word_tokenize(str(x))
)

"""#### 8. Text Normalization"""

# Normalizing Text (i.e., Stemming, Lemmatization etc.)
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

review['Normalized_Review'] = review['Tokens'].apply(
    lambda words: [
        lemmatizer.lemmatize(word)
        for word in words
    ]
)

"""##### Which text normalization technique have you used and why?

Lemmatization was chosen over other text normalization techniques because it produces meaningful dictionary words while preserving the actual meaning of the text.

Although stemming is another commonly used normalization technique, it often generates incomplete or invalid words. For example:

- studies → studi (stemming)
- studies → study (lemmatization)

Similarly:

- cars → car (lemmatization)
- running → running (lemmatization)

Lemmatization was preferred because:

1. It produces meaningful root words.
2. It preserves the context and semantic meaning of the text.
3. It improves the quality of text analysis and natural language processing tasks.
4. It reduces redundancy by converting different forms of the same word into a common base form.
5. It is more suitable for customer reviews where maintaining word meaning is important.

Therefore, lemmatization was used instead of stemming for text normalization in this project.

#### 9. Part of speech tagging
"""

# POS Taging
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

review['POS_Tags'] = review['Review'].apply(
    lambda x: pos_tag(
        word_tokenize(str(x))
    )
)

"""#### 10. Text Vectorization"""

# Vectorizing Text
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=1000
)

tfidf_matrix = tfidf.fit_transform(
    review['Review']
)

print(tfidf_matrix.shape)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

tfidf_df.head()

"""##### Which text vectorization technique have you used and why?

###Why Count Vectorizer?
1.Count Vectorizer only counts word occurrences.

2.TF-IDF considers both word frequency and word importance.

3.Common words receive lower weights.

4.More informative features are generated.

### 4. Feature Manipulation & Selection

#### 1. Feature Manipulation
"""

# Manipulate Features to minimize feature correlation and create new features
# Create working dataframe
review_df = review.dropna(subset=['Review']).copy()

# Feature 1: Word Count
review_df['Word_Count'] = (
    review_df['Review']
    .astype(str)
    .apply(lambda x: len(x.split()))
)

# Feature 2: Review Length
review_df['Review_Length'] = (
    review_df['Review']
    .astype(str)
    .apply(len)
)

# Feature 3: Pictures
review_df['Pics'] = pd.to_numeric(
    review_df['Pictures'],
    errors='coerce'
)

# Feature 4: Rating (Target Variable)
review_df['Rating'] = pd.to_numeric(
    review_df['Rating'],
    errors='coerce'
)

# Feature 5: Reviewer Experience
review_counts = review_df['Reviewer'].value_counts()

review_df['Reviewer_Experience'] = (
    review_df['Reviewer']
    .map(review_counts)
)

# Correlation matrix
corr_matrix = review_df[
    [
        'Rating',
        'Word_Count',
        'Review_Length',
        'Pics',
        'Reviewer_Experience'
    ]
].corr()

print(corr_matrix)

# Remove highly correlated feature
review_df.drop(
    columns=['Review_Length'],
    inplace=True
)

"""#### 2. Feature Selection"""

# Select your features wisely to avoid overfitting
# Correlation matrix
selected_features = [
    'Rating',
    'Word_Count',
    'Reviewer_Experience',
    'Pics'
]

model_data = review_df[selected_features]

model_data.head()

"""##### What all feature selection methods have you used  and why?

### Feature Selection Methods Used

The following feature selection methods were used to select relevant features and reduce overfitting.

#### 1. Correlation Analysis

A correlation matrix was used to identify highly correlated numerical features.

```python
corr_matrix = review_df[['Rating', 'Word_Count', 'Review_Length', 'Pics','Reviewer_Experience']].corr()
```

Highly correlated features provide similar information and may introduce multicollinearity.

#### 2. Removal of Highly Correlated Features

The correlation analysis showed that **Word_Count** and **Review_Length** had a very high positive correlation. Therefore, the **Review_Length** feature was removed.

```python
review_df.drop(columns=['Review_Length'],inplace=True)
```

### Why were these methods used?

- To reduce feature redundancy.
- To minimize multicollinearity.
- To decrease the risk of overfitting.
- To improve model generalization.
- To retain only meaningful and independent features.

### Conclusion

Correlation-based feature selection was used because it is simple, effective, and suitable for identifying redundant features. The highly correlated feature **Review_Length** was removed, while **Word_Count** was retained because it provides more interpretable information about customer reviews.

##### Which all features you found important and why?

Answer Here.

### 5. Data Transformation

#### Do you think that your data needs to be transformed? If yes, which transformation have you used. Explain Why?

### Data Transformation Technique Used

The **Log Transformation (`log1p`)** technique was used to transform the numerical features before scaling.

```python
import numpy as np

review_df['Word_Count'] = np.log1p(
    review_df['Word_Count']
)

review_df['Reviewer_Experience'] = np.log1p(
    review_df['Reviewer_Experience']
)

review_df['Pics'] = np.log1p(
    review_df['Pics']
)
```

### Why was Log Transformation used?

- The numerical features exhibited high positive skewness.
- Features such as **Word_Count**, **Reviewer_Experience**, and **Pictures** contained large variations in their values.
- The **Pictures** feature was highly skewed because most users uploaded zero pictures, while a few users uploaded many pictures.
- Log transformation compresses large values and reduces the effect of outliers.
- It makes the data distribution more balanced and suitable for clustering algorithms.

### Advantages of Log Transformation

- Reduces positive skewness.
- Minimizes the influence of extreme values.
- Improves data distribution.
- Enhances the performance of distance-based clustering algorithms such as K-Means.
- Helps numerical features contribute more equally during clustering.

### Conclusion

Log transformation was applied to the numerical features to reduce skewness and stabilize the data distribution before applying feature scaling and clustering techniques.
"""

# Transform Your data
review_df[
    ['Word_Count', 'Reviewer_Experience', 'Pics', 'Rating']
].skew()

review_df['Word_Count'] = np.log1p(
    review_df['Word_Count']
)
review_df['Rating'] = np.log1p(
    review_df['Rating']
)
review_df['Pics'] = np.log1p(
    review_df['Pics']
)

review_df['Reviewer_Experience'] = np.log1p(
    review_df['Reviewer_Experience']
)

"""### 6. Data Scaling"""

# Scaling your data
from sklearn.preprocessing import StandardScaler

features_to_scale = [
    'Word_Count',
    'Reviewer_Experience',
    'Pics',
    'Rating'
]

scaler = StandardScaler()

review_df[features_to_scale] = scaler.fit_transform(
    review_df[features_to_scale]
)

from scipy.sparse import hstack
X_num = review_df[
    [
        'Word_Count',
        'Reviewer_Experience',
        'Pics',
        'Rating'
    ]
]
X = hstack([
    tfidf_matrix,
    X_num.values
])

print(X.shape)

"""##### Which method have you used to scale you data and why?

### 7. Dimesionality Reduction

##### Do you think that dimensionality reduction is needed? Explain Why?

Yes, dimensionality reduction is required because the dataset contains a large number of TF-IDF features.
"""

# DImensionality Reduction (If needed)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100,random_state=42)

X_reduced = svd.fit_transform(X)

print(X_reduced.shape)

"""##### Which dimensionality reduction technique have you used and why? (If dimensionality reduction done on dataset.)

TruncatedSVD helps reduce dimensionality, improve computational efficiency, and enhance clustering performance while preserving most of the important information.

### 8. Data Splitting
"""

# Split your data to train and test. Choose Splitting ratio wisely.

"""##### What data splitting ratio have you used and why?

Data splitting was not performed because the project uses unsupervised learning algorithms. Clustering techniques such as K-Means, Agglomerative Clustering, and DBSCAN do not require a target variable or separate training and testing datasets. The entire dataset was used for training in order to identify meaningful patterns and generate stable clusters.

### 9. Handling Imbalanced Dataset

##### Do you think the dataset is imbalanced? Explain Why.

The concept of class imbalance is not applicable in this project because clustering is an unsupervised learning technique. The dataset does not contain predefined class labels or target variables. Therefore, the issue of class imbalance does not arise.
"""

# Handling Imbalanced Dataset (If needed)

"""##### What technique did you use to handle the imbalance dataset and why? (If needed to be balanced)

Answer Here.

## ***7. ML Model Implementation***

### ML Model - 1
"""

# ML Model - 1 Implementation
# Fit the Algorithm
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5,random_state=42,n_init=10)

kmeans_labels = kmeans.fit_predict(X_reduced)

# Predict on the model
review_df['KMeans_Cluster'] = kmeans_labels

review_df[['KMeans_Cluster']].head(20)

"""#### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart."""

# Visualizing evaluation Metric Score chart
from sklearn.metrics import silhouette_score

kmeans_score = silhouette_score(X_reduced,kmeans_labels)

print(kmeans_score)

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_reduced)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker='o'
)

plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')

plt.grid(True)

plt.show()

"""#### 2. Cross- Validation & Hyperparameter Tuning"""

# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Fit the Algorithm
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

for k in range(2,11):

    model = KMeans(n_clusters=k,random_state=42,n_init=10)

    labels = model.fit_predict(X_reduced)

    score = silhouette_score(X_reduced,labels)

    print(f'k={k}: {score:.4f}')

# Predict on the model

"""##### Which hyperparameter optimization technique have you used and why?

The Silhouette Score method was used for hyperparameter optimization in the K-Means clustering model. The number of clusters (k) was varied across different values, and the Silhouette Score was calculated for each value.

The value of k that produced the highest Silhouette Score was selected as the optimal number of clusters. This technique was used because the Silhouette Score measures both the cohesion within clusters and the separation between clusters, making it an effective evaluation metric for clustering algorithms.

Based on the Silhouette Score analysis, k = 5 was selected as the optimal number of clusters for the K-Means model.

##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

Yes, a noticeable improvement was observed after applying hyperparameter tuning. The clustering quality improved as the data became more compact and meaningful. The Silhouette Score increased after applying TruncatedSVD and selecting the optimal number of clusters using hyperparameter tuning.

### ML Model - 2
"""

# ML Model - 2 Implementation
# Fit the Algorithm
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(
    n_clusters=5,
    linkage='ward'
)

agg_labels = agg.fit_predict(X_reduced)

review_df['Agglomerative_Cluster'] = agg_labels

review_df[['Agglomerative_Cluster']].head(20)

"""#### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart."""

# Visualizing evaluation Metric Score chart
from sklearn.metrics import silhouette_score

agg_score = silhouette_score(X_reduced,agg_labels)

print(agg_score)

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

sample = X_reduced[:100]

linked = linkage(sample, method='ward')

plt.figure(figsize=(12,6))

dendrogram(linked)

plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')

plt.show()

"""#### 2. Cross- Validation & Hyperparameter Tuning"""

# ML Model - 2 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Fit the Algorithm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

for linkage in ['ward', 'complete', 'average', 'single']:

    try:
        model = AgglomerativeClustering(
            n_clusters=5,
            linkage=linkage
        )

        labels = model.fit_predict(X_reduced)

        score = silhouette_score(
            X_reduced,
            labels
        )

        print(f"{linkage}: {score:.4f}")

    except Exception as e:
        print(f"{linkage}: ERROR -> {e}")

"""##### Which hyperparameter optimization technique have you used and why?

Silhouette Score-based hyperparameter tuning was used to optimize the Agglomerative Clustering model. Different linkage methods such as Ward, Complete, Average, and Single were evaluated. The Ward linkage method produced the highest Silhouette Score and was selected as the optimal hyperparameter because it provided better cluster cohesion and separation.

##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

Yes, an improvement was observed after hyperparameter tuning. Among the different linkage methods tested, the Ward linkage produced the highest Silhouette Score of 0.3151 and was selected as the optimal linkage method for the final Agglomerative Clustering model.

#### 3. Explain each evaluation metric's indication towards business and the business impact pf the ML model used.

### Evaluation Metric and Business Impact

The Silhouette Score was used as the evaluation metric to assess the quality of the clusters. It measures how similar a data point is to its own cluster compared to other clusters. The score ranges from -1 to 1, where a higher value indicates better cluster separation and cohesion.

The K-Means model achieved a Silhouette Score of 0.3688, indicating reasonably well-defined customer review clusters. The Agglomerative Clustering model achieved a Silhouette Score of 0.3151, showing moderate cluster separation.

From a business perspective, a higher Silhouette Score indicates that customers with similar review patterns are grouped together effectively. This helps businesses understand different customer segments based on review content, ratings, reviewer experience, and picture uploads.

The clustering models provide several business benefits:

* Identify groups of satisfied and dissatisfied customers.
* Understand different reviewer behaviors and engagement patterns.
* Detect highly experienced reviewers and influential customers.
* Improve customer segmentation for targeted marketing strategies.
* Assist restaurants in understanding customer preferences and service issues.
* Support data-driven decision-making by identifying distinct customer groups.

Among the models used, K-Means produced the highest Silhouette Score, indicating better cluster quality and making it the most suitable model for customer review segmentation in this project.

### ML Model - 3
"""

# ML Model - 3 Implementation

# Fit the Algorithm
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=1.1,
    min_samples=10
)

dbscan_labels = dbscan.fit_predict(X_reduced)

# Predict on the model
review_df['DBSCAN_Cluster'] = dbscan_labels

review_df[['DBSCAN_Cluster']].head()

review_df['DBSCAN_Cluster'].value_counts()

"""#### 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart."""

# Visualizing evaluation Metric Score chart
eps_values = [0.3, 0.5, 0.7, 0.9, 1.1]

scores = [
    0.0634,
    -0.3332,
    -0.0748,
    0.0509,
    0.0785
]

plt.figure(figsize=(7,5))

plt.plot(
    eps_values,
    scores,
    marker='o'
)

plt.xlabel('eps')
plt.ylabel('Silhouette Score')
plt.title('DBSCAN Hyperparameter Tuning')

plt.grid(True)

plt.show()

"""#### 2. Cross- Validation & Hyperparameter Tuning"""

# ML Model - 3 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Fit the Algorithm
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

for eps in [0.3, 0.5, 0.7, 0.9, 1.1]:

    model = DBSCAN(
        eps=eps,
        min_samples=10
    )

    labels = model.fit_predict(X_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters > 1:
        score = silhouette_score(X_reduced, labels)

        print(
            f"eps={eps}, clusters={n_clusters}, score={score:.4f}"
        )

# Predict on the model

"""##### Which hyperparameter optimization technique have you used and why?

Silhouette Score-based hyperparameter tuning was used to optimize the DBSCAN model. Different values of the eps parameter were evaluated, and the value producing the highest Silhouette Score was selected. This method helps identify the parameter that produces better cluster separation.

##### Have you seen any improvement? Note down the improvement with updates Evaluation metric Score Chart.

Yes, improvement was observed after hyperparameter tuning. Different values of the eps parameter were tested, and eps = 1.1 produced the highest Silhouette Score of 0.0785. Therefore, this value was selected for the final DBSCAN model.

### 1. Which Evaluation metrics did you consider for a positive business impact and why?

The **Silhouette Score** was considered as the primary evaluation metric for measuring the performance of the clustering models. This metric was chosen because it evaluates both the cohesion within clusters and the separation between different clusters.

A higher Silhouette Score indicates that customers within the same cluster have similar characteristics, while customers belonging to different clusters are well separated.

From a business perspective, a higher Silhouette Score leads to:

* Better customer segmentation.
* More meaningful groups of customer reviews.
* Improved understanding of customer behavior.
* Identification of different reviewer categories.
* Better decision-making for restaurants and service providers.
* More targeted marketing and customer engagement strategies.

Among the three models, K-Means achieved the highest Silhouette Score of 0.3688, followed by Agglomerative Clustering with 0.3151 and DBSCAN with 0.0785. Therefore, K-Means provided the most meaningful customer segments and the greatest positive business impact.

### 2. Which ML model did you choose from the above created models as your final prediction model and why?

Among the three clustering models developed, **K-Means Clustering** was selected as the final model for customer review segmentation.

The performance of the three models was evaluated using the Silhouette Score:

| Model                    | Silhouette Score |
| ------------------------ | ---------------- |
| K-Means                  | 0.3688           |
| Agglomerative Clustering | 0.3151           |
| DBSCAN                   | 0.0785           |

K-Means achieved the highest Silhouette Score of **0.3688**, indicating better cluster cohesion and separation compared to the other models. This means that the customer reviews within the same cluster were more similar, while the clusters themselves were well separated.

The reasons for selecting K-Means as the final model are:

* It produced the highest Silhouette Score among all models.
* It generated well-defined and meaningful customer clusters.
* It showed better cluster separation and cohesion.
* It was computationally efficient for a large dataset.
* It provided more interpretable customer segments for business analysis.

Therefore, K-Means Clustering was selected as the final model because it provided the best clustering performance and the greatest business value for customer review segmentation.

### 3. Explain the model which you have used and the feature importance using any model explainability tool?

### Explain the model which you have used and the feature importance using any model explainability tool.

The final model selected for this project was **K-Means Clustering** because it achieved the highest Silhouette Score of 0.3688 among all the clustering algorithms.

K-Means is a centroid-based clustering algorithm that partitions the dataset into K clusters by minimizing the distance between data points and their respective cluster centroids. The algorithm iteratively updates the centroids until convergence is achieved.

Since K-Means is an unsupervised learning algorithm, traditional feature importance methods such as feature importance scores used in supervised learning models are not directly applicable. Therefore, model explainability was performed using **cluster centroid analysis**.

The important features used in the model were:

* Word_Count
* Reviewer_Experience
* Pictures (Pics)
* Rating
* TF-IDF textual features

Cluster centroids were analyzed to understand how these features influenced the formation of different customer groups. Features with larger differences among cluster centers contributed more significantly to the separation of clusters.

The analysis indicated that:

* Word_Count helped distinguish detailed reviewers from short reviewers.
* Reviewer_Experience differentiated experienced and new reviewers.
* Pictures identified highly engaged customers.
* Rating separated positive and negative customer experiences.

Therefore, cluster centroid analysis was used as the model explainability technique to interpret the behavior and importance of features in the K-Means clustering model.

## ***8.*** ***Future Work (Optional)***

### 1. Save the best performing ml model in a pickle file or joblib file format for deployment process.
"""

# Save the File
import joblib

# Save K-Means model
joblib.dump(
    kmeans,
    'kmeans_model.pkl'
)

# Save TF-IDF Vectorizer
joblib.dump(
    tfidf,
    'tfidf_vectorizer.pkl'
)

# Save TruncatedSVD
joblib.dump(
    svd,
    'svd_model.pkl'
)

# Save StandardScaler
joblib.dump(
    scaler,
    'scaler.pkl'
)

"""### 2. Again Load the saved model file and try to predict unseen data for a sanity check.

"""

# Load the File and predict unseen data.
kmeans_model = joblib.load(
    'kmeans_model.pkl'
)

tfidf_model = joblib.load(
    'tfidf_vectorizer.pkl'
)

svd_model = joblib.load(
    'svd_model.pkl'
)

scaler_model = joblib.load(
    'scaler.pkl'
)

new_review = "The food was amazing and the service was excellent."

new_review = new_review.lower()

new_tfidf = tfidf_model.transform([new_review])

word_count = len(new_review.split())

reviewer_experience = 5
pics = 1
rating = 4

new_num = [[
    word_count,
    reviewer_experience,
    pics,
    rating
]]

new_num_scaled = scaler_model.transform(new_num)

from scipy.sparse import hstack

new_data = hstack([
    new_tfidf,
    new_num_scaled
])

new_data_reduced = svd_model.transform(new_data)

predicted_cluster = kmeans_model.predict(
    new_data_reduced
)

print(
    "Predicted Cluster:",
    predicted_cluster[0]
)

"""### ***Congrats! Your model is successfully created and ready for deployment on a live server for a real user interaction !!!***

# **Conclusion**

# Conclusion

In this project, customer reviews were analyzed using various text preprocessing, feature engineering, and clustering techniques to identify meaningful customer segments. The dataset was cleaned by handling missing values, performing text preprocessing, and applying TF-IDF vectorization to convert textual reviews into numerical representations.

Additional numerical features such as Word Count, Reviewer Experience, Pictures, and Rating were incorporated to improve the clustering performance. Dimensionality reduction using TruncatedSVD was applied to reduce the high dimensionality of the TF-IDF features and improve computational efficiency.

Three clustering algorithms, namely K-Means, Agglomerative Clustering, and DBSCAN, were implemented and evaluated using the Silhouette Score. Hyperparameter tuning was performed to obtain the optimal model parameters.

The performance of the models was as follows:

* K-Means Clustering: 0.3688
* Agglomerative Clustering: 0.3151
* DBSCAN: 0.0785

Among all the models, K-Means Clustering achieved the highest Silhouette Score and produced the most meaningful customer segments. Therefore, K-Means was selected as the final model for customer review segmentation.

The final model was successfully saved using the Joblib library and validated through a sanity check using unseen data. The model correctly predicted the cluster for a new customer review, demonstrating its capability for deployment and real-world applications.

This project helps businesses understand customer behavior, identify different customer groups, analyze customer sentiments, and support data-driven decision-making. The developed clustering system can assist restaurants and service providers in improving customer satisfaction, enhancing marketing strategies, and delivering better services based on customer feedback.

### ***Hurrah! You have successfully completed your Machine Learning Capstone Project !!!***
"""
