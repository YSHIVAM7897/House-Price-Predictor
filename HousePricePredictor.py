import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = ['bedrooms', 'bathrooms', 'sqft', 'age', 'garage', 'location_factor']
        self.location_mapping = {
            'budget': 0.8,
            'average': 1.0,
            'good': 1.2,
            'premium': 1.5
        }

    def generate_data(self, n_samples=1000):
        """Generate synthetic house data for demonstration"""
        np.random.seed(42)

        # Generate features
        bedrooms = np.random.randint(1, 6, n_samples)
        bathrooms = np.random.randint(1, 4, n_samples)
        sqft = np.random.normal(2000, 800, n_samples)
        sqft = np.clip(sqft, 500, 5000)

        age = np.random.randint(0, 50, n_samples)
        garage = np.random.randint(0, 4, n_samples)
        location_factor = np.random.choice([0.8, 1.0, 1.2, 1.5], n_samples, p=[0.3, 0.4, 0.2, 0.1])

        # Calculate price based on features
        base_price = (
            sqft * 150 +
            bedrooms * 10000 +
            bathrooms * 15000 +
            garage * 8000 -
            age * 1000
        )

        price = base_price * location_factor + np.random.normal(0, 20000, n_samples)
        price = np.clip(price, 50000, 1000000)

        self.data = pd.DataFrame({
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft': sqft.astype(int),
            'age': age,
            'garage': garage,
            'location_factor': location_factor,
            'price': price.astype(int)
        })

        return self.data

    def train_model(self):
        """Train the house price prediction model"""
        print("🏠 Training House Price Prediction Model...")
        print("=" * 50)

        # Generate data if not exists
        if not hasattr(self, 'data'):
            self.generate_data()

        # Prepare features and target
        X = self.data[self.feature_names]
        y = self.data['price']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Model Training Complete!")
        print(f"📊 Mean Absolute Error: ${mae:,.2f}")
        print(f"🎯 R² Score: {r2:.3f} ({r2*100:.1f}% accuracy)")
        print(f"📈 Training samples: {len(X_train)}")
        print(f"🧪 Test samples: {len(X_test)}")

        return mae, r2

    def predict_price(self, bedrooms, bathrooms, sqft, age, garage, location):
        """Predict house price based on features"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Convert location to factor
        location_factor = self.location_mapping.get(location.lower(), 1.0)

        # Prepare input
        input_data = np.array([[bedrooms, bathrooms, sqft, age, garage, location_factor]])
        input_scaled = self.scaler.transform(input_data)

        # Make prediction
        predicted_price = self.model.predict(input_scaled)[0]

        # Determine category
        if predicted_price < 200000:
            category = "Budget"
        elif predicted_price < 400000:
            category = "Moderate"
        elif predicted_price < 600000:
            category = "Expensive"
        else:
            category = "Luxury"

        return predicted_price, category

    def analyze_data(self):
        """Analyze the dataset and show insights"""
        if not hasattr(self, 'data'):
            self.generate_data()

        print("\n🔍 Dataset Analysis")
        print("=" * 50)

        # Basic statistics
        print("📊 Dataset Overview:")
        print(f"   Total houses: {len(self.data):,}")
        print(f"   Average price: ${self.data['price'].mean():,.0f}")
        print(f"   Price range: ${self.data['price'].min():,.0f} - ${self.data['price'].max():,.0f}")
        print(f"   Average size: {self.data['sqft'].mean():,.0f} sqft")
        print(f"   Average age: {self.data['age'].mean():.1f} years")

        # Price by features
        print("\n💰 Average Price by Features:")
        print("   By Bedrooms:")
        for bedrooms in sorted(self.data['bedrooms'].unique()):
            avg_price = self.data[self.data['bedrooms'] == bedrooms]['price'].mean()
            print(f"     {bedrooms} bedroom(s): ${avg_price:,.0f}")

        print("   By Neighborhood:")
        location_names = {0.8: 'Budget', 1.0: 'Average', 1.2: 'Good', 1.5: 'Premium'}
        for factor, name in location_names.items():
            avg_price = self.data[self.data['location_factor'] == factor]['price'].mean()
            print(f"     {name}: ${avg_price:,.0f}")

    def plot_analysis(self):
        """Create visualizations of the data"""
        if not hasattr(self, 'data'):
            self.generate_data()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🏠 House Price Analysis Dashboard', fontsize=16, fontweight='bold')

        # Price distribution
        axes[0, 0].hist(self.data['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].ticklabel_format(style='plain', axis='x')

        # Price vs Square Footage
        scatter = axes[0, 1].scatter(self.data['sqft'], self.data['price'],
                                   c=self.data['bedrooms'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('Price vs Square Footage (colored by bedrooms)')
        axes[0, 1].set_xlabel('Square Footage')
        axes[0, 1].set_ylabel('Price ($)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Bedrooms')

        # Average price by bedrooms
        bedroom_prices = self.data.groupby('bedrooms')['price'].mean()
        axes[1, 0].bar(bedroom_prices.index, bedroom_prices.values, color='lightcoral')
        axes[1, 0].set_title('Average Price by Number of Bedrooms')
        axes[1, 0].set_xlabel('Number of Bedrooms')
        axes[1, 0].set_ylabel('Average Price ($)')

        # Price by location
        location_names = {0.8: 'Budget', 1.0: 'Average', 1.2: 'Good', 1.5: 'Premium'}
        self.data['location_name'] = self.data['location_factor'].map(location_names)
        location_prices = self.data.groupby('location_name')['price'].mean()
        axes[1, 1].bar(location_prices.index, location_prices.values, color='lightgreen')
        axes[1, 1].set_title('Average Price by Neighborhood Type')
        axes[1, 1].set_xlabel('Neighborhood Type')
        axes[1, 1].set_ylabel('Average Price ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def interactive_prediction(self):
        """Interactive command-line interface for predictions"""
        print("\n🎯 Interactive House Price Prediction")
        print("=" * 50)

        while True:
            try:
                print("\nEnter house details:")
                bedrooms = int(input("Number of bedrooms (1-5): "))
                bathrooms = int(input("Number of bathrooms (1-4): "))
                sqft = int(input("Square footage (500-5000): "))
                age = int(input("Age of house in years (0-50): "))
                garage = int(input("Number of garage spaces (0-3): "))

                print("\nNeighborhood types:")
                print("1. Budget")
                print("2. Average")
                print("3. Good")
                print("4. Premium")
                location_choice = int(input("Select neighborhood type (1-4): "))

                location_map = {1: 'budget', 2: 'average', 3: 'good', 4: 'premium'}
                location = location_map[location_choice]

                # Make prediction
                price, category = self.predict_price(bedrooms, bathrooms, sqft, age, garage, location)

                print(f"\n🏠 House Summary:")
                print(f"   Bedrooms: {bedrooms}")
                print(f"   Bathrooms: {bathrooms}")
                print(f"   Square Footage: {sqft:,} sqft")
                print(f"   Age: {age} years")
                print(f"   Garage: {garage} spaces")
                print(f"   Neighborhood: {location.title()}")

                print(f"\n💰 Predicted Price: ${price:,.0f}")
                print(f"🏷️  Category: {category}")

                # Continue or exit
                continue_pred = input("\nMake another prediction? (y/n): ").lower()
                if continue_pred != 'y':
                    break

            except (ValueError, KeyError) as e:
                print(f"❌ Invalid input. Please try again.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

# Demo the system
def main():
    print("🏠 House Price Prediction System")
    print("=" * 50)

    # Initialize predictor
    predictor = HousePricePredictor()

    # Train model
    predictor.train_model()

    # Analyze data
    predictor.analyze_data()

    # Show visualizations
    predictor.plot_analysis()

    # Example predictions
    print("\n🎯 Example Predictions:")
    print("=" * 50)

    examples = [
        (3, 2, 2000, 10, 1, 'average'),
        (4, 3, 2500, 5, 2, 'good'),
        (2, 1, 1200, 20, 0, 'budget'),
        (5, 4, 3500, 2, 3, 'premium')
    ]

    for bedrooms, bathrooms, sqft, age, garage, location in examples:
        price, category = predictor.predict_price(bedrooms, bathrooms, sqft, age, garage, location)
        print(f"🏠 {bedrooms}BR/{bathrooms}BA, {sqft:,}sqft, {age}yr old, {garage} garage, {location}")
        print(f"   💰 Predicted Price: ${price:,.0f} ({category})")

    # Interactive mode
    try_interactive = input("\n🎮 Try interactive prediction mode? (y/n): ").lower()
    if try_interactive == 'y':
        predictor.interactive_prediction()

    print("\n✅ Demo completed! The model is ready for use.")

if __name__ == "__main__":
    main()
