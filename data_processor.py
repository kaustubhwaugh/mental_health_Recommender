import pandas as pd
import numpy as np
import ast  # For safely evaluating string-represented lists
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

def preprocess_data_for_model2_updated(question_bank_path, merged_dataset_path):
    """
    Preprocesses the (NEW) merged dataset for Model 2 training,
    where response columns are fixed (Response_Q1 to Response_Q20)
    and QuestionIds column indicates the actual questions.

    Args:
        question_bank_path (str): Path to the QuestionBank.csv file.
        merged_dataset_path (str): Path to the new merged_dataset.csv file.

    Returns:
        tuple: (X_processed, y_processed_presence, y_processed_severity)
               - X_processed (pd.DataFrame): DataFrame of feature vectors (89 features per user).
               - y_processed_presence (pd.DataFrame): DataFrame of binary presence labels for conditions.
               - y_processed_severity (pd.DataFrame): DataFrame of numerically encoded severity labels.
    """
    try:
        question_bank_df = pd.read_csv(question_bank_path)
        merged_df = pd.read_csv(merged_dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure file paths are correct.")
        return None, None, None
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None, None, None

    # Get the canonical list of all question IDs from the question bank
    try:
        all_question_ids = sorted(question_bank_df['Question_Id'].unique(), key=lambda x: int(x[1:]))
        num_total_questions = len(all_question_ids)
    except Exception as e:
        print(
            f"Error processing QuestionBank.csv: {e}. Ensure 'Question_Id' column exists and is formatted like 'Q1', 'Q2', etc.")
        return None, None, None

    processed_features_list = []

    # Define the fixed response column names as they appear in the new merged_dataset.csv
    response_column_headers = [f"Response_Q{i + 1}" for i in range(20)]

    for index, row in merged_df.iterrows():
        user_feature_vector = [-1] * num_total_questions  # Initialize with -1 (not asked)

        try:
            # Safely evaluate the string representation of the list of question IDs
            actual_question_ids_for_user = ast.literal_eval(row['QuestionIds'])

            if not isinstance(actual_question_ids_for_user, list) or len(actual_question_ids_for_user) != 20:
                print(
                    f"Warning: QuestionIds for UserID {row.get('UserID', 'Unknown')} at index {index} is not a list of 20 questions. Found: {row['QuestionIds']}. Skipping row for X features.")
                processed_features_list.append(user_feature_vector)  # Append vector of -1s
                continue

        except (ValueError, SyntaxError, TypeError) as e:
            print(
                f"Warning: Could not parse QuestionIds string for UserID {row.get('UserID', 'Unknown')} at index {index}. Error: {e}. Value: {row['QuestionIds']}. Skipping row for X features.")
            processed_features_list.append(user_feature_vector)  # Append vector of -1s
            continue

        # Iterate through the 20 responses and map them
        for i in range(20):
            response_col_name = response_column_headers[i]  # e.g., Response_Q1, Response_Q2, ...
            actual_q_id = actual_question_ids_for_user[i].strip()

            if actual_q_id in all_question_ids and response_col_name in merged_df.columns:
                try:
                    question_master_index = all_question_ids.index(actual_q_id)
                    response_value = row[response_col_name]

                    if pd.notna(response_value):
                        user_feature_vector[question_master_index] = int(response_value)
                    # else: keep as -1 if response is NaN but question was listed
                except ValueError:
                    print(
                        f"Warning: Question ID {actual_q_id} from user's QuestionIds not found in master question bank. UserID {row.get('UserID', 'Unknown')}.")
                except Exception as e:
                    print(
                        f"Warning: Error processing question {actual_q_id} (response from {response_col_name}) for UserID {row.get('UserID', 'Unknown')}: {e}")
            elif response_col_name not in merged_df.columns:
                print(
                    f"Warning: Expected response column {response_col_name} not found in merged_dataset for UserID {row.get('UserID', 'Unknown')}.")
            elif actual_q_id not in all_question_ids:
                print(
                    f"Warning: QuestionID {actual_q_id} in user's list not in bank for UserID {row.get('UserID', 'Unknown')}.")

        processed_features_list.append(user_feature_vector)

    X_processed = pd.DataFrame(processed_features_list, columns=all_question_ids)
    X_processed.index = merged_df.index  # Preserve original index if needed later

    # --- Label Processing (y) ---
    presence_columns = ['Is_Stress_Present', 'Is_Anxiety_Present', 'Is_Depression_Present']
    missing_presence_cols = [col for col in presence_columns if col not in merged_df.columns]
    if missing_presence_cols:
        print(f"Error: Missing presence label columns in merged_dataset: {missing_presence_cols}")
        return None, None, None
    y_processed_presence = merged_df[presence_columns].astype(int)

    severity_mapping = {'None': 0, 'Mild': 1, 'Moderate': 2, 'High': 3, 'Severe': 3}

    severity_columns_original = ['Stress_Severity', 'Anxiety_Severity', 'Depression_Severity']
    missing_severity_cols = [col for col in severity_columns_original if col not in merged_df.columns]
    if missing_severity_cols:
        print(f"Error: Missing severity label columns in merged_dataset: {missing_severity_cols}")
        return None, None, None

    y_processed_severity = pd.DataFrame(index=merged_df.index)
    for col in severity_columns_original:
        y_processed_severity[col] = merged_df[col].astype(str).map(severity_mapping).fillna(-1).astype(int)

    return X_processed, y_processed_presence, y_processed_severity


def train_model2_prototypes(X_features, y_presence_labels, y_severity_labels):
    """
    Trains prototype models for Model 2 (condition presence and severity).

    Args:
        X_features (pd.DataFrame): Processed feature matrix.
        y_presence_labels (pd.DataFrame): Processed binary presence labels.
        y_severity_labels (pd.DataFrame): Processed numerical severity labels.

    Returns:
        dict: A dictionary containing trained models and evaluation reports.
    """
    if X_features is None or y_presence_labels is None or y_severity_labels is None:
        print("Error: Input data is not available for training Model 2.")
        return None

    models_and_evals = {'evaluations': {}}

    # --- 1. Train Models for Condition Presence (Multi-label) ---
    print("\n--- Training Condition Presence Models ---")
    X_features_p = X_features.copy()
    y_presence_labels_p = y_presence_labels.copy()

    common_idx_p = X_features_p.index.intersection(y_presence_labels_p.index)
    X_features_p = X_features_p.loc[common_idx_p]
    y_presence_labels_p = y_presence_labels_p.loc[common_idx_p]

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_features_p, y_presence_labels_p, test_size=0.25, random_state=42
    )

    # Using Logistic Regression with MultiOutputClassifier
    lr_presence_classifier = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
    # Changed n_jobs from -1 to 1 to avoid loky errors on cleanup
    multi_target_lr_presence = MultiOutputClassifier(lr_presence_classifier, n_jobs=1)
    multi_target_lr_presence.fit(X_train_p, y_train_p)
    models_and_evals['presence_model_lr'] = multi_target_lr_presence

    y_pred_lr_p = multi_target_lr_presence.predict(X_test_p)
    print("\nLogistic Regression - Presence Prediction Report (Multi-Label):")
    report_lr_p_full = classification_report(y_test_p, y_pred_lr_p, target_names=y_presence_labels_p.columns.tolist(), zero_division=0)
    print(report_lr_p_full)
    models_and_evals['evaluations']['presence_lr_report'] = classification_report(y_test_p, y_pred_lr_p,
                                                                                  target_names=y_presence_labels_p.columns.tolist(),
                                                                                  zero_division=0, output_dict=True)

    # Using RandomForestClassifier with MultiOutputClassifier
    rf_presence_classifier = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_estimators=100)
    # Changed n_jobs from -1 to 1 to avoid loky errors on cleanup
    multi_target_rf_presence = MultiOutputClassifier(rf_presence_classifier, n_jobs=1)
    multi_target_rf_presence.fit(X_train_p, y_train_p)
    models_and_evals['presence_model_rf'] = multi_target_rf_presence

    y_pred_rf_p = multi_target_rf_presence.predict(X_test_p)
    print("\nRandom Forest - Presence Prediction Report (Multi-Label):")
    report_rf_p_full = classification_report(y_test_p, y_pred_rf_p, target_names=y_presence_labels_p.columns.tolist(), zero_division=0)
    print(report_rf_p_full)
    models_and_evals['evaluations']['presence_rf_report'] = classification_report(y_test_p, y_pred_rf_p,
                                                                                  target_names=y_presence_labels_p.columns.tolist(),
                                                                                  zero_division=0, output_dict=True)

    # --- 2. Train Models for Condition Severity (Separate Multi-class for each condition) ---
    print("\n--- Training Condition Severity Models ---")
    models_and_evals['severity_models_lr'] = {}
    models_and_evals['severity_models_rf'] = {}
    models_and_evals['evaluations']['severity_reports_lr'] = {}
    models_and_evals['evaluations']['severity_reports_rf'] = {}

    presence_cols_ordered = ['Is_Stress_Present', 'Is_Anxiety_Present', 'Is_Depression_Present']
    severity_cols_ordered = ['Stress_Severity', 'Anxiety_Severity', 'Depression_Severity']

    for i in range(len(presence_cols_ordered)):
        condition_presence_col = presence_cols_ordered[i]
        condition_severity_col = severity_cols_ordered[i]
        condition_name = condition_presence_col.replace('Is_', '').replace('_Present', '')

        print(f"\nTraining Severity Model for: {condition_name}")

        y_sev_condition = y_severity_labels[condition_severity_col]

        X_features_s = X_features.copy()
        y_sev_condition_s = y_sev_condition.copy()
        common_idx_s = X_features_s.index.intersection(y_sev_condition_s.index)
        X_features_s = X_features_s.loc[common_idx_s]
        y_sev_condition_s = y_sev_condition_s.loc[common_idx_s]

        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_features_s, y_sev_condition_s, test_size=0.25, random_state=42, stratify=y_sev_condition_s
        )

        # Logistic Regression for Severity
        # Define the base logistic regression estimator (without multi_class='ovr')
        base_lr_severity = LogisticRegression(solver='liblinear', random_state=42,
                                              class_weight='balanced', max_iter=1000)

        # Wrap it with OneVsRestClassifier
        # n_jobs=1 can be set here for the OvR process if desired, similar to MultiOutputClassifier
        lr_severity = OneVsRestClassifier(base_lr_severity, n_jobs=1)

        lr_severity.fit(X_train_s, y_train_s)
        models_and_evals['severity_models_lr'][condition_name] = lr_severity
        y_pred_lr_s = lr_severity.predict(X_test_s)
        unique_labels_s = np.unique(np.concatenate((y_test_s.unique(), y_pred_lr_s)))
        print(f"Logistic Regression - {condition_name} Severity Report (Labels: {unique_labels_s}):")
        report_lr_s_full = classification_report(y_test_s, y_pred_lr_s, zero_division=0, labels=unique_labels_s, target_names=[str(l) for l in unique_labels_s]) # Ensure target_names are strings if labels are numeric
        print(report_lr_s_full)
        models_and_evals['evaluations']['severity_reports_lr'][condition_name] = classification_report(y_test_s,
                                                                                                       y_pred_lr_s,
                                                                                                       zero_division=0,
                                                                                                       labels=unique_labels_s,
                                                                                                       target_names=[str(l) for l in unique_labels_s],
                                                                                                       output_dict=True)

        # RandomForestClassifier for Severity
        rf_severity = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_estimators=100)
        rf_severity.fit(X_train_s, y_train_s)
        models_and_evals['severity_models_rf'][condition_name] = rf_severity
        y_pred_rf_s = rf_severity.predict(X_test_s)
        unique_labels_s_rf = np.unique(np.concatenate((y_test_s.unique(), y_pred_rf_s)))
        print(f"Random Forest - {condition_name} Severity Report (Labels: {unique_labels_s_rf}):")
        report_rf_s_full = classification_report(y_test_s, y_pred_rf_s, zero_division=0, labels=unique_labels_s_rf, target_names=[str(l) for l in unique_labels_s_rf]) # Ensure target_names are strings
        print(report_rf_s_full)
        models_and_evals['evaluations']['severity_reports_rf'][condition_name] = classification_report(y_test_s,
                                                                                                       y_pred_rf_s,
                                                                                                       zero_division=0,
                                                                                                       labels=unique_labels_s_rf,
                                                                                                       target_names=[str(l) for l in unique_labels_s_rf],
                                                                                                       output_dict=True)
    return models_and_evals


# --- Main execution block ---
if __name__ == '__main__':
    # Replace with the actual paths to your CSV files
    question_bank_file = 'QuestionBank.csv'
    new_merged_dataset_file = 'merged_dataset.csv' # The new file you just uploaded

    X_new, y_presence_new, y_severity_new = preprocess_data_for_model2_updated(question_bank_file, new_merged_dataset_file)

    if X_new is not None and y_presence_new is not None and y_severity_new is not None:
        print("\n--- (NEW) Processed Features (X_new) ---")
        print(f"Shape: {X_new.shape}")
        print(X_new.head())
        if not X_new.empty:
             print(f"Example row (user 0) non -1 values: {X_new.iloc[0][X_new.iloc[0] != -1].count()}")
        else:
            print("X_new is empty.")


        print("\n--- (NEW) Processed Presence Labels (y_presence_new) ---")
        print(f"Shape: {y_presence_new.shape}")
        print(y_presence_new.head())

        print("\n--- (NEW) Processed Severity Labels (y_severity_new) ---")
        print(f"Shape: {y_severity_new.shape}")
        print(y_severity_new.head())
        if 'Anxiety_Severity' in y_severity_new.columns:
            print("\nValue counts for Anxiety_Severity (new):")
            print(y_severity_new['Anxiety_Severity'].value_counts(dropna=False))
        else:
            print("Anxiety_Severity column not found in y_severity_new.")


        print("\nProceeding to train Model 2 prototypes...")
        trained_models_info = train_model2_prototypes(X_new, y_presence_new, y_severity_new)

        if trained_models_info:
            print("\n\n--- Model 2 Training Complete ---")
            print("Trained models and evaluation reports are available in the 'trained_models_info' dictionary.")

            # Example of how to access specific information:
            print("\nLogistic Regression Presence Model Report (dict format):")
            if 'evaluations' in trained_models_info and 'presence_lr_report' in trained_models_info['evaluations']:
                print(trained_models_info['evaluations']['presence_lr_report'])

            print("\nRandom Forest Anxiety Severity Model Report (dict format):")
            if 'evaluations' in trained_models_info and \
               'severity_reports_rf' in trained_models_info['evaluations'] and \
               'Anxiety' in trained_models_info['evaluations']['severity_reports_rf']:
                print(trained_models_info['evaluations']['severity_reports_rf']['Anxiety'])

            if 'severity_models_rf' in trained_models_info and 'Anxiety' in trained_models_info['severity_models_rf']:
                anxiety_rf_model = trained_models_info['severity_models_rf']['Anxiety']
                print(f"\nAnxiety RF Model: {anxiety_rf_model}")
        else:
            print("Model training failed or returned no information.")
    else:
        print("Data not processed correctly, cannot train Model 2.")
import joblib
import os
os.makedirs("models", exist_ok=True)

# Save the trained models
joblib.dump(trained_models_info['presence_model_rf'], "models/presence_model_rf.pkl")
joblib.dump(trained_models_info['severity_models_rf']['Stress'], "models/severity_rf_stress.pkl")
joblib.dump(trained_models_info['severity_models_rf']['Anxiety'], "models/severity_rf_anxiety.pkl")
joblib.dump(trained_models_info['severity_models_rf']['Depression'], "models/severity_rf_depression.pkl")