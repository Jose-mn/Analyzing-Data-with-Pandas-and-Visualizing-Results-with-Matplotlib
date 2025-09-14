# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# load and explore dataset

# Load dataset with error handling
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = iris.target_names[iris.target]

    # Show first rows
    display(df.head())

    # Dataset info
    print("\nDataset Info:")
    print(df.info())

    # Missing values
    print("\nMissing Values:\n", df.isnull().sum())

    # Handle missing values (if any)
    df = df.fillna(df.mean(numeric_only=True))

except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print("An error occurred:", e)


# 2: Basic data analysis
#basic statistics
df.describe()

# Grouping example: Average petal length per species
avg_petal_length = df.groupby("species")["petal length (cm)"].mean()
avg_petal_length
 
 # 3: Data visualization
 #line chart
plt.figure(figsize=(8,5))
plt.plot(df.index[:50], df["sepal length (cm)"][:50], color="blue", marker="o")
plt.title("Line Chart: Sepal Length Trend (first 50 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.show()

# bar chart
plt.figure(figsize=(8,5))
sns.barplot(x=avg_petal_length.index, y=avg_petal_length.values, palette="Set2")
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# histogram
plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"], bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# scatter plot
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="Set1")
plt.title("Scatter Plot: Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
