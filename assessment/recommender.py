import pandas as pd
import random


class SimpleQuestionRecommender:
    def __init__(self, question_bank_path):
        """
        Initializes the recommender with the question bank.

        Args:
            question_bank_path (str): Path to the QuestionBank.csv file.
        """
        try:
            self.question_bank_df = pd.read_csv(question_bank_path)
            if 'Question_Id' not in self.question_bank_df.columns or \
                    'Question_Topic' not in self.question_bank_df.columns:
                raise ValueError("QuestionBank.csv must contain 'Question_Id' and 'Question_Topic' columns.")
            # Basic preprocessing: remove rows with missing topics or IDs
            self.question_bank_df.dropna(subset=['Question_Id', 'Question_Topic'], inplace=True)
        except FileNotFoundError:
            print(f"Error: QuestionBank.csv not found at {question_bank_path}")
            self.question_bank_df = pd.DataFrame()  # Empty dataframe
        except ValueError as ve:
            print(f"Error: {ve}")
            self.question_bank_df = pd.DataFrame()

    def recommend_questions(self, num_questions_to_recommend=20, primary_concern_topic=None,
                            topic_distribution_goal=None, min_questions_per_covered_topic=1):
        """
        Recommends a diverse set of questions.

        Args:
            num_questions_to_recommend (int): Total number of questions to recommend.
            primary_concern_topic (str, optional): A primary topic of concern to prioritize.
            topic_distribution_goal (dict, optional): Desired number of questions per topic.
                                                      Example: {'Depression': 5, 'Anxiety': 5, 'Stress': 4}
                                                      If None, aims for broader, more even coverage.
            min_questions_per_covered_topic (int): Minimum questions to pick from a topic if it's chosen
                                                   for coverage (used if topic_distribution_goal is None).

        Returns:
            list: A list of selected Question_Id strings.
        """
        if self.question_bank_df.empty:
            print("Question bank is empty or not loaded properly. Cannot recommend questions.")
            return []

        available_questions = self.question_bank_df.copy()
        selected_question_ids = []
        selected_questions_df = pd.DataFrame()  # To keep track of questions already selected

        # --- Step 1: Prioritize Primary Concern Topic (if any) ---
        if primary_concern_topic and primary_concern_topic in available_questions['Question_Topic'].unique():
            concern_questions = available_questions[available_questions['Question_Topic'] == primary_concern_topic]

            # Determine how many to pick from primary concern
            num_from_primary = 0
            if topic_distribution_goal and primary_concern_topic in topic_distribution_goal:
                num_from_primary = topic_distribution_goal[primary_concern_topic]
            elif topic_distribution_goal:  # primary_concern not in specific goals, but goals exist
                # take a general larger portion if primary concern, e.g. 25-30%
                num_from_primary = max(min_questions_per_covered_topic, int(num_questions_to_recommend * 0.25))
            else:  # No specific distribution, take a decent chunk for primary concern
                num_from_primary = max(min_questions_per_covered_topic, int(num_questions_to_recommend * 0.3))

            num_to_select = min(len(concern_questions), num_from_primary)

            if num_to_select > 0:
                selected_primary = concern_questions.sample(n=num_to_select, replace=False)
                selected_question_ids.extend(selected_primary['Question_Id'].tolist())
                selected_questions_df = pd.concat([selected_questions_df, selected_primary])
                available_questions = available_questions.drop(selected_primary.index)

        # --- Step 2: Fulfill Topic Distribution Goal (if specified) ---
        if topic_distribution_goal:
            for topic, num_needed in topic_distribution_goal.items():
                if topic == primary_concern_topic:  # Already handled or partially handled
                    num_already_selected_for_topic = len(
                        selected_questions_df[selected_questions_df['Question_Topic'] == topic])
                    num_needed -= num_already_selected_for_topic

                if num_needed <= 0 or len(selected_question_ids) >= num_questions_to_recommend:
                    continue

                topic_questions = available_questions[available_questions['Question_Topic'] == topic]
                num_to_select = min(len(topic_questions), num_needed)

                if num_to_select > 0:
                    selected_topic_qs = topic_questions.sample(n=num_to_select, replace=False)
                    selected_question_ids.extend(selected_topic_qs['Question_Id'].tolist())
                    selected_questions_df = pd.concat([selected_questions_df, selected_topic_qs])
                    available_questions = available_questions.drop(selected_topic_qs.index)
                    if len(selected_question_ids) >= num_questions_to_recommend:
                        break

        # --- Step 3: Ensure Broader Topic Coverage & Fill Remaining Slots ---
        # This part runs if no specific distribution goal, or if goal is met but still need more Qs

        # Get all unique topics remaining in available_questions
        remaining_topics = available_questions['Question_Topic'].unique().tolist()
        random.shuffle(remaining_topics)  # Shuffle to vary selection order

        # Try to get at least 'min_questions_per_covered_topic' from several different topics
        for topic in remaining_topics:
            if len(selected_question_ids) >= num_questions_to_recommend:
                break

            # If topic_distribution_goal was set, don't over-select for already covered topics
            # unless we are just filling general slots
            if topic_distribution_goal and topic in topic_distribution_goal and \
                    len(selected_questions_df[selected_questions_df['Question_Topic'] == topic]) >= \
                    topic_distribution_goal[topic]:
                continue

            topic_questions = available_questions[available_questions['Question_Topic'] == topic]
            num_to_select_from_this_topic = min(len(topic_questions), min_questions_per_covered_topic)

            # If we need more questions overall than this min, take more if available
            if (num_questions_to_recommend - len(selected_question_ids)) > num_to_select_from_this_topic:
                num_to_select_from_this_topic = min(len(topic_questions),
                                                    (num_questions_to_recommend - len(selected_question_ids)))

            if num_to_select_from_this_topic > 0:
                selected_current_topic_qs = topic_questions.sample(n=num_to_select_from_this_topic, replace=False)
                selected_question_ids.extend(selected_current_topic_qs['Question_Id'].tolist())
                selected_questions_df = pd.concat([selected_questions_df, selected_current_topic_qs])
                available_questions = available_questions.drop(selected_current_topic_qs.index)

        # --- Step 4: If still not enough questions, fill randomly from what's left ---
        if len(selected_question_ids) < num_questions_to_recommend and not available_questions.empty:
            num_still_needed = num_questions_to_recommend - len(selected_question_ids)
            num_to_select = min(len(available_questions), num_still_needed)

            if num_to_select > 0:
                final_fill_qs = available_questions.sample(n=num_to_select, replace=False)
                selected_question_ids.extend(final_fill_qs['Question_Id'].tolist())

        return selected_question_ids[:num_questions_to_recommend]


# --- Example Usage ---
# Replace with the actual path to your QuestionBank.csv
question_bank_file = 'QuestionBank.csv'
recommender = SimpleQuestionRecommender(question_bank_file)

if not recommender.question_bank_df.empty:
    print("Recommender initialized.")

    # Scenario 1: General recommendation for 20 questions
    general_questions = recommender.recommend_questions(num_questions_to_recommend=20)
    print(f"\nGeneral 20 questions ({len(general_questions)}): {general_questions}")
    # Verify diversity by checking topics
    if general_questions:
        selected_q_details = recommender.question_bank_df[
            recommender.question_bank_df['Question_Id'].isin(general_questions)]
        print("Topics covered in general selection:")
        print(selected_q_details['Question_Topic'].value_counts())

    # Scenario 2: Prioritize "Anxiety", with a specific distribution goal
    anxiety_goal_dist = {'Anxiety': 7, 'Depression': 5, 'Stress': 5}  # Total 17, rest will be filled
    anxiety_focused_questions = recommender.recommend_questions(
        num_questions_to_recommend=20,
        primary_concern_topic='Anxiety',
        topic_distribution_goal=anxiety_goal_dist
    )
    print(f"\nAnxiety-focused 20 questions ({len(anxiety_focused_questions)}): {anxiety_focused_questions}")
    if anxiety_focused_questions:
        selected_q_details_anxiety = recommender.question_bank_df[
            recommender.question_bank_df['Question_Id'].isin(anxiety_focused_questions)]
        print("Topics covered in anxiety-focused selection:")
        print(selected_q_details_anxiety['Question_Topic'].value_counts())

    # Scenario 3: Specific distribution for 10 questions
    specific_dist_10 = {'Depression': 4, 'Stress': 3, 'Anxiety': 3}
    specific_10_questions = recommender.recommend_questions(
        num_questions_to_recommend=10,
        topic_distribution_goal=specific_dist_10
    )
    print(f"\nSpecific 10 questions ({len(specific_10_questions)}): {specific_10_questions}")
    if specific_10_questions:
        selected_q_details_10 = recommender.question_bank_df[
            recommender.question_bank_df['Question_Id'].isin(specific_10_questions)]
        print("Topics covered in specific 10 selection:")
        print(selected_q_details_10['Question_Topic'].value_counts())

    # Scenario 4: Recommend only 5 questions, prioritizing "Depression" without specific counts for other topics
    depression_5_questions = recommender.recommend_questions(
        num_questions_to_recommend=5,
        primary_concern_topic='Depression',
        min_questions_per_covered_topic=1  # Ensure some diversity if possible
    )
    print(f"\nDepression-focused 5 questions ({len(depression_5_questions)}): {depression_5_questions}")
    if depression_5_questions:
        selected_q_details_dep_5 = recommender.question_bank_df[
            recommender.question_bank_df['Question_Id'].isin(depression_5_questions)]
        print("Topics covered in depression-focused 5 selection:")
        print(selected_q_details_dep_5['Question_Topic'].value_counts())