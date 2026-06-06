import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('Model/netflix_churn_model.pkl')

# Load the new data for prediction
data_predict = pd.read_csv('Data/data_predict.csv')

# Define the features to be used for prediction
features = ['age', 'gender', 'subscription_type',
            'watch_hours', 'last_login_days', 'region',
            'device', 'monthly_fee', 'payment_method',
            'number_of_profiles', 'avg_watch_time_per_day', 'favorite_genre']

# Prepare the data for prediction
x_new = data_predict[features]

# Make predictions using the loaded model
prediction = model.predict(x_new)

# Add the predictions to the original DataFrame
data_predict['Churn_Prediction'] = prediction
# Map the predictions to 'yes' and 'no' for better readability
data_predict['Churn_Prediction'] = data_predict['Churn_Prediction'].map({1: 'yes' , 0: 'no'})


# Display the distribution of predictions
print(data_predict['Churn_Prediction'].value_counts())

# Visualize the distribution of predictions
plt.figure(figsize=(10, 6))
sns.countplot(data=data_predict, x='Churn_Prediction', hue='Churn_Prediction', palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn_Prediction')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('Visualizations/churn_prediction_distribution.png')  # Save the plot as an image file
plt.close()

# Save the predictions to a new CSV file
data_predict.to_csv('Data/data_predict_with_churn.csv', index=False)
