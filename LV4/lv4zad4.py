import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data(file_path):
    df = pd.read_csv(file_path)
    print("\nOsnovne informacije o skupu podataka:")
    print(df.info())
    return df

def analyze_price_distribution(df):
    sorted_df = df.sort_values("selling_price")
    
    print("\nNajjeftiniji automobil:")
    print(sorted_df.head(1))
    
    print("\nNajskuplji automobil:")
    print(sorted_df.tail(1))
    
    return sorted_df

def analyze_year_data(df, year=2012):
    count = df["year"].value_counts().get(year, 0)
    print(f"\nBroj automobila proizvedenih {year}. godine: {count}")
    return count

def calculate_avg_km_by_fuel(df):
    fuel_km = df[["km_driven", "fuel"]]

    avg_km = fuel_km.groupby("fuel")["km_driven"].mean()
    
    print("\nProsječna kilometraža po tipu goriva:")
    print(avg_km)
    
    return avg_km

def visualize_data(df):
    plt.figure(figsize=(15, 10))
    
    print("\nGeneriram pairplot...")
    sns.pairplot(df, hue='fuel')
    plt.suptitle("Pairplot po tipu goriva", y=1.02)
    
    print("\nGeneriram relplot...")
    sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
    plt.title("Odnos kilometraže i cijene po tipu goriva")
    
    df_clean = df.drop(['name', 'mileage'], axis=1)
    obj_cols = df_clean.select_dtypes(object).columns
    num_cols = df_clean.select_dtypes(np.number).columns
    
    print("\nGeneriram countplotove...")
    fig, axes = plt.subplots(1, len(obj_cols), figsize=(15, 5))
    for idx, col in enumerate(obj_cols):
        sns.countplot(x=col, data=df_clean, ax=axes[idx])
        axes[idx].set_title(f"Distribucija za {col}")
    
    print("\nGeneriram boxplot...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='fuel', y='selling_price', data=df_clean)
    plt.title("Distribucija cijena po tipu goriva")
    
    print("\nGeneriram histogram...")
    plt.figure(figsize=(8, 6))
    sns.histplot(df_clean['selling_price'], kde=True)
    plt.title("Distribucija prodajnih cijena")
    
    plt.tight_layout()
    plt.show()

def main():
    df = load_and_explore_data('cars_processed.csv')
    
    sorted_df = analyze_price_distribution(df)
    
    analyze_year_data(df)
    
    calculate_avg_km_by_fuel(df)
    
    visualize_data(df)

if __name__ == "__main__":
    main()
