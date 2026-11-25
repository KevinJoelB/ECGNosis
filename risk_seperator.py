import pandas as pd

def get_low_risk_data():
    low_risk_df = df[df['Risk_Category'].str.lower() == 'low risk']
    low_risk_df = low_risk_df.drop(columns=['True_Label', 'Predicted_Label'])
    low_risk_df = low_risk_df.sort_values(by='Prediction_Probability', ascending=True)
    low_risk_df.to_csv("low_risk_sorted.csv", index=False)
    print("Low risk data saved to low_risk_sorted.csv")

def get_high_risk_data():
    high_risk_df = df[df['Risk_Category'].str.lower() == 'high risk']
    high_risk_df = high_risk_df.drop(columns=['True_Label', 'Predicted_Label'])
    high_risk_df = high_risk_df.sort_values(by='Prediction_Probability', ascending=False)
    high_risk_df.to_csv("high_risk_sorted.csv", index=False)
    print("High risk data saved to high_risk_sorted.csv")

if __name__ == "__main__":
    df = pd.read_csv("results\ecg_risk_predictions.csv")
    get_low_risk_data()
    get_high_risk_data()