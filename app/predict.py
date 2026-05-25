import pandas as pd
import numpy as np
import joblib

def load_pipeline(model_path='ridge_model.pkl', preprocessor_path='preprocessor.pkl'):
    """Loads the trained model and preprocessor."""
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure the .pkl files are in the same directory as this script.")
        return None, None

def predict_and_rank(repositories, model, preprocessor):
    """
    Takes a list of raw repository dictionaries, preprocesses them,
    predicts the star count, and returns a ranked list.
    """
    # 1. Convert the raw list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(repositories)
    
    # 2. Apply the exact same log transformations used during training
    # We use np.log1p to handle any potential zeros safely
    df['forks_count_log'] = np.log1p(df['forks_count'])
    df['open_issues_count_log'] = np.log1p(df['open_issues_count'])
    df['days_since_last_push_log'] = np.log1p(df['days_since_last_push'])
    
    # 3. Select only the features the model expects, in the exact order
    features = [
        'topic_count', 
        'forks_count_log', 
        'open_issues_count_log', 
        'days_since_last_push_log',
        'has_projects', 
        'has_wiki'
    ]
    X_inference = df[features]
    
    # 4. Scale the continuous features using the saved preprocessor
    X_scaled = preprocessor.transform(X_inference)
    
    # 5. Make predictions (these will be in log-scale)
    log_predictions = model.predict(X_scaled)
    
    # 6. Convert predictions back to normal star counts using np.expm1
    actual_predictions = np.expm1(log_predictions)
    
    # 7. Add the predictions to our original dataframe and rank them
    df['predicted_stars'] = actual_predictions
    
    # Sort from highest predicted stars to lowest
    ranked_df = df.sort_values(by='predicted_stars', ascending=False).reset_index(drop=True)
    
    return ranked_df

if __name__ == "__main__":
    # --- 1. Load the Pipeline ---
    # Note to DevOps: You can swap 'ridge_model.pkl' for 'rf_model.pkl' or 'gb_model.pkl'
    # depending on which model performed best during the final training run.
    print("Loading ML pipeline...")
    model, preprocessor = load_pipeline(model_path='ridge_model.pkl', preprocessor_path='preprocessor.pkl')
    
    if model and preprocessor:
        # --- 2. Provide the 5 sample repositories for the App demonstration ---
        # Note to DevOps: These can be dynamically populated via the GitHub API in production
        sample_repos = [
            {
                "name": "Super Fast API",
                "has_projects": True,
                "has_wiki": True,
                "topic_count": 8,
                "forks_count": 1200,
                "open_issues_count": 45,
                "days_since_last_push": 1
            },
            {
                "name": "Legacy Data Parser",
                "has_projects": False,
                "has_wiki": False,
                "topic_count": 2,
                "forks_count": 15,
                "open_issues_count": 120,
                "days_since_last_push": 400
            },
            {
                "name": "React UI Components",
                "has_projects": True,
                "has_wiki": False,
                "topic_count": 15,
                "forks_count": 8500,
                "open_issues_count": 300,
                "days_since_last_push": 3
            },
            {
                "name": "Niche ML Library",
                "has_projects": False,
                "has_wiki": True,
                "topic_count": 5,
                "forks_count": 85,
                "open_issues_count": 12,
                "days_since_last_push": 14
            },
            {
                "name": "Personal Blog Template",
                "has_projects": False,
                "has_wiki": False,
                "topic_count": 1,
                "forks_count": 2,
                "open_issues_count": 0,
                "days_since_last_push": 60
            }
        ]

        # --- 3. Run Inference and Print Results ---
        print("\nRunning predictions and ranking repositories...\n")
        ranked_results = predict_and_rank(sample_repos, model, preprocessor)
        
        print("-" * 50)
        print(f"{'Rank':<5} | {'Repository Name':<25} | {'Predicted Stars'}")
        print("-" * 50)
        
        for index, row in ranked_results.iterrows():
            # Format the output for a clean terminal display
            rank = index + 1
            repo_name = row['name']
            stars = int(round(row['predicted_stars'])) # Round to nearest whole star
            print(f"#{rank:<4} | {repo_name:<25} | {stars:,} stars")
        
        print("-" * 50)