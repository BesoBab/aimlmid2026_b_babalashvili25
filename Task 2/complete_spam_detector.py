"""
COMPLETE SPAM EMAIL DETECTION SYSTEM
AI and ML for Cybersecurity - Midterm Exam
Author: [Your Name]
Date: January 9, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
import os
import re
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. DATA LOADING AND PROCESSING
# ============================================================================

class EmailFeatureExtractor:
    """
    Extracts features from email text similar to the CSV dataset
    """

    SPAM_WORDS = [
        'free', 'win', 'winner', 'click', 'here', 'buy', 'now', 'discount',
        'offer', 'limited', 'time', 'urgent', 'money', 'cash', 'prize',
        'guaranteed', 'risk', 'trial', 'special', 'promotion',
        'congratulations', 'selected', 'lottery', 'million', 'billion',
        'dollars', 'credit', 'card', 'loan', 'mortgage', 'viagra',
        'cialis', 'pharmacy', 'prescription', 'drugs', 'amazing',
        'opportunity', 'exclusive', 'secret', 'instant', 'wealth',
        'double', 'bonus', 'extra', 'income', 'profit'
    ]

    @staticmethod
    def extract_features(email_text: str):
        """
        Extract features from email text
        Returns: Dictionary with features
        """
        if not email_text or not isinstance(email_text, str):
            return {'words': 0, 'links': 0, 'capital_words': 0, 'spam_word_count': 0}

        text_lower = email_text.lower()

        # 1. Count total words
        words = re.findall(r'\b\w+\b', email_text)
        words_count = len(words)

        # 2. Count links
        link_pattern = r'(https?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov|info|biz))'
        links = re.findall(link_pattern, text_lower, re.IGNORECASE)
        links_count = len(links)

        # 3. Count words in ALL CAPS (at least 2 characters)
        capital_words = re.findall(r'\b[A-Z]{2,}\b', email_text)
        capital_words_count = len(capital_words)

        # 4. Count spam words (more sophisticated counting)
        spam_word_count = 0
        email_words = set(re.findall(r'\b\w+\b', text_lower))
        for spam_word in EmailFeatureExtractor.SPAM_WORDS:
            if spam_word in text_lower:
                # Count all occurrences
                spam_word_count += len(re.findall(r'\b' + re.escape(spam_word) + r'\b', text_lower))

        return {
            'words': max(words_count, 1),  # Avoid zero division
            'links': links_count,
            'capital_words': capital_words_count,
            'spam_word_count': spam_word_count
        }


# ============================================================================
# 2. LOGISTIC REGRESSION MODEL
# ============================================================================

class SpamDetector:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ['words', 'links', 'capital_words', 'spam_word_count']
        self.df = None

    def load_data(self, csv_path):
        """Load and prepare data"""
        print("📁 LOADING DATASET")
        print("=" * 50)

        try:
            self.df = pd.read_csv(csv_path)
            print(f"✅ Dataset loaded successfully")
            print(f"   Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
            print(f"   Features: {', '.join(self.feature_names)}")

            # Display dataset statistics
            print(f"\n📊 DATASET STATISTICS:")
            print(f"   Total emails: {len(self.df)}")
            print(f"   Legitimate (0): {sum(self.df['is_spam'] == 0)}")
            print(f"   Spam (1): {sum(self.df['is_spam'] == 1)}")
            print(f"   Spam ratio: {(self.df['is_spam'].mean() * 100):.2f}%")

            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def prepare_data(self):
        """Prepare data for training"""
        print("\n⚙️ PREPARING DATA FOR TRAINING")
        print("=" * 50)

        # Prepare features and target
        X = self.df[self.feature_names].values
        y = self.df['is_spam'].values

        # Split data (70% train, 30% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"✅ Data split completed")
        print(f"   Training set: {len(self.X_train)} samples (70%)")
        print(f"   Test set: {len(self.X_test)} samples (30%)")

        # Display feature statistics
        print(f"\n📈 FEATURE STATISTICS (Training Set):")
        feature_stats = pd.DataFrame(self.X_train, columns=self.feature_names)
        print(feature_stats.describe().round(2))

    def train_logistic_regression(self):
        """Train logistic regression model"""
        print("\n🤖 TRAINING LOGISTIC REGRESSION MODEL")
        print("=" * 50)

        # Create model
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear',
            penalty='l2',
            C=1.0
        )

        # Train model
        self.model.fit(self.X_train, self.y_train)

        print("✅ Model training completed")

        # Display coefficients
        print("\n📊 MODEL COEFFICIENTS:")
        print("-" * 40)
        coefficients = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_[0],
            'Absolute_Value': np.abs(self.model.coef_[0])
        })
        coefficients = coefficients.sort_values('Absolute_Value', ascending=False)

        for _, row in coefficients.iterrows():
            effect = "INCREASES" if row['Coefficient'] > 0 else "DECREASES"
            print(f"{row['Feature']:20} : {row['Coefficient']:8.4f} ({effect} spam probability)")

        print(f"\nIntercept (Bias): {self.model.intercept_[0]:.4f}")

        # Save model
        joblib.dump(self.model, 'spam_model.pkl')
        print("💾 Model saved as 'spam_model.pkl'")

        return coefficients

    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📈 MODEL EVALUATION")
        print("=" * 50)

        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"✅ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n📋 CONFUSION MATRIX:")
        print(" " * 15 + "Predicted")
        print(" " * 15 + "0      1")
        print(" " * 11 + "+" + "-" * 15)
        print(f"Actual 0 | {cm[0, 0]:5}  {cm[0, 1]:5}")
        print(f"Actual 1 | {cm[1, 0]:5}  {cm[1, 1]:5}")

        # Detailed metrics
        tn, fp, fn, tp = cm.ravel()

        print(f"\n📊 DETAILED METRICS:")
        print(f"   True Negatives (Legitimate): {tn}")
        print(f"   False Positives: {fp}")
        print(f"   False Negatives: {fn}")
        print(f"   True Positives (Spam): {tp}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n🎯 PERFORMANCE SCORES:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        # Classification report
        print(f"\n📄 CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, y_pred,
                                    target_names=['Legitimate', 'Spam'],
                                    digits=4))

        return cm, accuracy

    def predict_email(self, email_text):
        """Predict if an email is spam"""
        print("\n" + "=" * 50)
        print("EMAIL ANALYSIS")
        print("=" * 50)
        print("📧 Email Text:")
        print("-" * 40)
        print(email_text[:200] + "..." if len(email_text) > 200 else email_text)
        print("-" * 40)

        # Extract features
        features = EmailFeatureExtractor.extract_features(email_text)

        print(f"\n🔍 EXTRACTED FEATURES:")
        for feature, value in features.items():
            print(f"   {feature:20} : {value}")

        # Prepare feature vector
        X_new = np.array([[features['words'], features['links'],
                           features['capital_words'], features['spam_word_count']]])

        # Make prediction
        prediction = self.model.predict(X_new)[0]
        probabilities = self.model.predict_proba(X_new)[0]

        print(f"\n🎯 PREDICTION RESULT:")
        print("-" * 40)
        if prediction == 1:
            print("🔴 CLASSIFIED AS: SPAM")
        else:
            print("🟢 CLASSIFIED AS: LEGITIMATE")

        print(f"\n📊 PROBABILITY DISTRIBUTION:")
        print(f"   Legitimate: {probabilities[0]:.4f} ({probabilities[0] * 100:.1f}%)")
        print(f"   Spam:       {probabilities[1]:.4f} ({probabilities[1] * 100:.1f}%)")

        # Confidence level
        confidence = max(probabilities)
        if confidence > 0.9:
            confidence_level = "VERY HIGH"
        elif confidence > 0.7:
            confidence_level = "HIGH"
        elif confidence > 0.5:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"

        print(f"\n💪 CONFIDENCE LEVEL: {confidence_level}")

        return prediction, probabilities, features


# ============================================================================
# 3. VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(detector, cm):
    """Create all required visualizations"""
    print("\n🎨 CREATING VISUALIZATIONS")
    print("=" * 50)

    # Create visualizations directory
    os.makedirs('report_visualizations', exist_ok=True)

    # Visualization 1: Class Distribution
    plt.figure(figsize=(10, 6))
    class_counts = detector.df['is_spam'].value_counts()
    colors = ['#4CAF50', '#F44336']
    bars = plt.bar(['Legitimate\n(Class 0)', 'Spam\n(Class 1)'],
                   class_counts.values, color=colors, alpha=0.8)

    plt.title('Email Class Distribution in Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Email Class', fontsize=12)
    plt.ylabel('Number of Emails', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 10,
                 f'{int(height)}\n({height / len(detector.df) * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('report_visualizations/1_class_distribution.png', dpi=150, bbox_inches='tight')
    print("✅ Created: 1_class_distribution.png")
    plt.close()

    # Visualization 2: Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Predicted Legitimate', 'Predicted Spam'],
                yticklabels=['Actual Legitimate', 'Actual Spam'],
                cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='black')

    plt.title('Confusion Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('report_visualizations/2_confusion_matrix_heatmap.png', dpi=150, bbox_inches='tight')
    print("✅ Created: 2_confusion_matrix_heatmap.png")
    plt.close()

    # Visualization 3: Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation = detector.df[['words', 'links', 'capital_words', 'spam_word_count', 'is_spam']].corr()

    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, mask=mask,
                cbar_kws={'label': 'Correlation Coefficient'},
                linewidths=1, linecolor='white')

    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('report_visualizations/3_feature_correlation.png', dpi=150, bbox_inches='tight')
    print("✅ Created: 3_feature_correlation.png")
    plt.close()

    # Visualization 4: Feature Importance (Coefficients)
    if detector.model is not None:
        plt.figure(figsize=(12, 6))
        coefficients = detector.model.coef_[0]

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(coefficients))[::-1]
        sorted_features = [detector.feature_names[i] for i in sorted_idx]
        sorted_coefficients = coefficients[sorted_idx]

        colors = ['red' if x > 0 else 'green' for x in sorted_coefficients]
        bars = plt.bar(range(len(sorted_features)), sorted_coefficients,
                       color=colors, alpha=0.7, edgecolor='black')

        plt.title('Logistic Regression Feature Coefficients\n(Impact on Spam Probability)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Coefficient Value', fontsize=12)
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height > 0 else 'top'
            offset = 0.01 if height > 0 else -0.01
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     height + offset,
                     f'{height:.3f}', ha='center', va=va,
                     fontweight='bold')

        plt.tight_layout()
        plt.savefig('report_visualizations/4_feature_coefficients.png', dpi=150, bbox_inches='tight')
        print("✅ Created: 4_feature_coefficients.png")
        plt.close()

    # Visualization 5: Feature Distribution by Class
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features = ['words', 'links', 'capital_words', 'spam_word_count']

    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]

        # Separate by class
        legit_data = detector.df[detector.df['is_spam'] == 0][feature]
        spam_data = detector.df[detector.df['is_spam'] == 1][feature]

        # Create histogram
        ax.hist(legit_data, bins=30, alpha=0.5, label='Legitimate', color='green')
        ax.hist(spam_data, bins=30, alpha=0.5, label='Spam', color='red')

        ax.set_title(f'Distribution of {feature}', fontsize=12)
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Feature Distributions by Email Class', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('report_visualizations/5_feature_distributions.png', dpi=150, bbox_inches='tight')
    print("✅ Created: 5_feature_distributions.png")
    plt.close()

    print(f"\n🎉 All visualizations saved in 'report_visualizations' folder!")


# ============================================================================
# 4. EXAMPLE EMAILS FOR REPORT
# ============================================================================

def create_example_emails():
    """Create example emails for the report"""

    # Example 1: Spam Email
    spam_email = """
    URGENT NOTICE: CONGRATULATIONS!

    DEAR WINNER,

    You have been SELECTED as the GRAND PRIZE WINNER of $2,500,000.00!
    This is a LIMITED TIME OFFER - ACT NOW to claim your CASH PRIZE!

    Click this SECRET link to claim: http://win-now-free-money.com/claim-prize
    This offer is 100% FREE with NO fees and FULLY GUARANTEED!

    SPECIAL BONUS: Reply within 24 HOURS to get DOUBLE your prize money!
    Don't miss this EXCLUSIVE opportunity to become a MILLIONAIRE!

    REMEMBER: This is a TIME-SENSITIVE offer with LIMITED availability.
    Reply URGENTLY to: winners@global-lottery-promo.net

    CONFIDENTIALITY NOTICE: This email contains PRIVILEGED information.
    """

    # Example 2: Legitimate Email
    legitimate_email = """
    Subject: Project Update and Meeting Schedule

    Hello Team,

    I hope this email finds you well. I'm writing to provide an update on our current project status 
    and to schedule our next progress review meeting.

    As discussed in our last meeting, I've attached the updated project timeline document for your review. 
    Please take some time to go through it and let me know if you have any feedback or suggestions.

    We're scheduled to have our next progress review meeting next Wednesday, January 15th, at 2:00 PM.
    The meeting will be held in Conference Room B on the 3rd floor. For those who cannot attend in person, 
    you can join via Zoom using this link: https://company.zoom.us/j/1234567890

    Meeting Agenda:
    1. Project status update (15 minutes)
    2. Timeline review and adjustments (20 minutes)
    3. Resource allocation discussion (15 minutes)
    4. Q&A session (10 minutes)

    Please send me the status updates for your respective tasks by Monday EOD.

    Best regards,
    Sarah Johnson
    Project Manager
    Innovative Solutions Inc.
    Email: sarah.johnson@innovative-solutions.com
    Phone: (555) 123-4567
    Office: Suite 300, 123 Business Ave
    """

    return spam_email, legitimate_email


# ============================================================================
# 5. MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print(" " * 20 + "SPAM EMAIL DETECTION SYSTEM")
    print("=" * 70)
    print("AI and ML for Cybersecurity - Midterm Exam")
    print("January 9, 2026")
    print("=" * 70)

    # Initialize detector
    detector = SpamDetector()

    # Load data (update this path to your CSV file)
    csv_path = "/Users/BB/PycharmProjects/Mid-Term_AIandMLinCybersecurity/b_babalashvili25_34521.csv"

    if not detector.load_data(csv_path):
        print("❌ Failed to load data. Exiting...")
        return

    # Prepare data
    detector.prepare_data()

    # Train model
    coefficients = detector.train_logistic_regression()

    # Evaluate model
    cm, accuracy = detector.evaluate_model()

    # Create visualizations
    create_visualizations(detector, cm)

    # Create example emails
    spam_example, legit_example = create_example_emails()

    # Test with example emails
    print("\n" + "=" * 70)
    print(" " * 15 + "EXAMPLE EMAIL TESTING")
    print("=" * 70)

    print("\n📧 TEST 1: SPAM EMAIL EXAMPLE")
    print("-" * 50)
    prediction, proba, features = detector.predict_email(spam_example)

    print(f"\n💡 WHY THIS IS CLASSIFIED AS SPAM:")
    print("1. Contains multiple CAPITALIZED words ({features['capital_words']} words)")
    print(f"2. Has {features['links']} suspicious link(s)")
    print(f"3. Contains {features['spam_word_count']} spam keywords")
    print("4. Creates false urgency with 'limited time', 'act now'")
    print("5. Promises unrealistic rewards ('$2,500,000', 'millionaire')")

    print("\n📧 TEST 2: LEGITIMATE EMAIL EXAMPLE")
    print("-" * 50)
    prediction, proba, features = detector.predict_email(legit_example)

    print(f"\n💡 WHY THIS IS CLASSIFIED AS LEGITIMATE:")
    print("1. Professional tone and formatting")
    print(f"2. Normal word count ({features['words']} words)")
    print(f"3. Legitimate link for meeting ({features['links']} link)")
    print(f"4. No spam keywords detected ({features['spam_word_count']})")
    print(f"5. Appropriate capitalization ({features['capital_words']} words)")

    # Interactive testing
    print("\n" + "=" * 70)
    print(" " * 15 + "INTERACTIVE TESTING")
    print("=" * 70)

    while True:
        print("\nOptions:")
        print("1. Test a custom email")
        print("2. View model information")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            print("\nEnter your email text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    if lines and lines[-1] == "":
                        lines.pop()
                        break
                lines.append(line)

            email_text = "\n".join(lines)
            if email_text.strip():
                detector.predict_email(email_text)
            else:
                print("No email text provided.")

        elif choice == '2':
            print("\n" + "=" * 50)
            print("MODEL INFORMATION")
            print("=" * 50)
            print(f"Algorithm: Logistic Regression")
            print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"Training samples: {len(detector.X_train)}")
            print(f"Test samples: {len(detector.X_test)}")

            print(f"\nFeature Coefficients:")
            for idx, feature in enumerate(detector.feature_names):
                coef = detector.model.coef_[0][idx]
                print(f"  {feature:20}: {coef:.4f}")

        elif choice == '3':
            print("\nThank you for using the Spam Detection System!")
            print("Visualizations are saved in 'report_visualizations' folder.")
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# ============================================================================
# 6. REPORT CONTENT GENERATOR
# ============================================================================

def generate_report_content():
    """Generate content for the report"""
    print("\n" + "=" * 70)
    print(" " * 15 + "REPORT CONTENT SUMMARY")
    print("=" * 70)

    report_content = """
    ============================================================
    SPAM EMAIL DETECTION - REPORT CONTENT
    ============================================================

    1. DATA FILE LINK
    ------------------------------------------------------------
    The dataset file 'b_babalashvili25_34521.csv' is located at:
    /Users/BB/PycharmProjects/Mid-Term_AIandMLinCybersecurity/b_babalashvili25_34521.csv

    2. MODEL TRAINING DETAILS
    ------------------------------------------------------------
    Data Loading Code:
    - Used pandas.read_csv() to load the CSV file
    - Dataset contains 2500 emails with 5 columns
    - Features: words, links, capital_words, spam_word_count
    - Target: is_spam (0=legitimate, 1=spam)

    Model Code:
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    )

    Model Coefficients Found:
    - words:          0.0059 (positive impact)
    - links:          0.6384 (strong positive impact)
    - capital_words:  0.3593 (positive impact)
    - spam_word_count: 0.6112 (strong positive impact)
    - Intercept:     -7.4214

    3. MODEL VALIDATION RESULTS
    ------------------------------------------------------------
    Accuracy: 0.9627 (96.27%)

    Confusion Matrix:
                    Predicted
                    0      1
                +-----------
    Actual 0 |   357     11
    Actual 1 |    17    365

    Code for evaluation:
    - accuracy_score() for accuracy calculation
    - confusion_matrix() for confusion matrix
    - classification_report() for detailed metrics

    4. EMAIL CHECKING FEATURE
    ------------------------------------------------------------
    The system can parse email text and extract:
    1. Total word count
    2. Number of links/URLs
    3. Number of ALL-CAPS words
    4. Count of spam keywords

    These features are then passed to the trained model for classification.

    5. SPAM EMAIL EXAMPLE
    ------------------------------------------------------------
    Email Text: (See spam_example variable in code)

    How it was created to be spam:
    - Uses CAPITAL LETTERS for urgency
    - Contains multiple spam keywords
    - Has suspicious links
    - Creates false urgency
    - Promises unrealistic rewards

    6. LEGITIMATE EMAIL EXAMPLE
    ------------------------------------------------------------
    Email Text: (See legitimate_email variable in code)

    How it was created to be legitimate:
    - Professional business format
    - Legitimate meeting details
    - Appropriate capitalization
    - No spam keywords
    - Realistic content

    7. VISUALIZATIONS
    ------------------------------------------------------------
    5 visualizations created:
    1. class_distribution.png - Bar chart of email classes
    2. confusion_matrix_heatmap.png - Heatmap of confusion matrix
    3. feature_correlation.png - Correlation between features
    4. feature_coefficients.png - Model coefficient values
    5. feature_distributions.png - Feature distributions by class

    All visualizations are saved in 'report_visualizations' folder.
    """

    print(report_content)

    # Save report to file
    with open('spam_detection_report.txt', 'w') as f:
        f.write(report_content)
    print("✅ Report content saved to 'spam_detection_report.txt'")


# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    # Install required packages if not installed
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib']

    try:
        main()

        # Ask if user wants to generate report
        print("\n" + "=" * 70)
        response = input("Generate report content? (y/n): ").strip().lower()
        if response == 'y':
            generate_report_content()

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install pandas numpy scikit-learn matplotlib seaborn joblib")