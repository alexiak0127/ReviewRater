# 🎬 ReviewRater: Predicting Movie Review Ratings with Classical ML

This project was created for a CS506 Midterm Kaggle competition that challenges participants to **predict the star rating** (1–5) associated with user-submitted movie reviews using only **classical machine learning methods** (no deep learning or boosting libraries allowed).

## 🧠 Project Goal

> **Objective:** Predict star ratings from user reviews of various movies using structured metadata and textual content (i.e., the review text and summary).  
> **Constraints:**  
> - No neural networks or deep learning libraries  
> - No boosting methods (e.g., XGBoost, LightGBM)

## 📂 Dataset

- `train.csv`: Contains ~1.7M user reviews with star ratings
- `test.csv`: Contains ~212k review IDs with missing star ratings (for prediction)

## 📊 Data Exploration

- Previewed rows with `.head()`, checked `.shape`, and visualized the `Score` distribution.
- Removed rows from `train.csv` that appeared in `test.csv` to prevent **data leakage**.
- Imputed missing `Score` values using random sampling from the observed score distribution.
- Filled missing `Text` and `Summary` fields with empty strings.
- Manually reviewed ~2,000 entries to create custom **positive/negative sentiment word lists**.
- Experimented on subsets (100k, 500k rows) before scaling up to the full dataset for final modeling.

## 🧰 Feature Engineering

Implemented a modular pipeline via `extract_all_features()`, which called:

### Text Cleaning
- Lowercased and stripped punctuation (`clean_text`)
- Applied to both `Text` and `Summary`

### Basic Text Features
- Lengths, word counts, average word lengths
- Sentence counts and average words per sentence

### Punctuation & Formatting
- Count and ratio of exclamations, questions
- ALL CAPS word counts and ratios

### Sentiment Features
- Custom word list for movie-specific positive/negative terms
- Sentiment counts, scores, densities, and ratios

### Summary-Text Relationship
- Sentiment alignment between text & summary
- Length ratio and word overlap percentage

### User & Product Statistics
- Average rating, rating variance, and review count per `UserId` and `ProductId`
- Handled unseen values via `fillna` defaults

### Combined Features
- `ComplexityScore` = Avg. word length + sentence complexity
- `SentimentComplexityScore` = Interaction term between sentiment & complexity

### TF-IDF Features
- For `Text`: ```TfidfVectorizer(max_features=300, ngram_range=(1,1))```
- For `Summary`: ```TfidfVectorizer(max_features=200, ngram_range=(1,1))```
- Combined structured and TF-IDF features via ```scipy.sparse.hstack```

## ⚙️ 3. Model Creation & Assumptions

### Assumptions
- Strong correlation between emotional tone (sentiment) and star rating
- Vocabulary consistency across user behavior
- User/Product history captures bias or popularity
- Longer/more complex reviews might indicate stronger emotion

### Models Used
```
models = {
  'Decision Tree': DecisionTreeClassifier(random_state=42),
  'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
  'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42),
  'Linear SVC': LinearSVC(random_state=42, dual=False, max_iter=2000)
}
```
Each model was trained using:
- Engineered features + TF-IDF vectors
- 75/25 train-test split
- Evaluation via 3-fold Stratified CV and confusion matrix

### 🔧 4. Model Tuning
- ```SelectKBest(f_classif, k=100)``` for feature selection
- ```TruncatedSVD(n_components=90)``` for dimensionality reduction
- ```StandardScaler()``` for scaling
- ```StratifiedKFold(n_splits=3)``` for cross-validation
- Attempted ```GridSearchCV``` but stopped due to runtime issues


## 📊 Model Evaluation

| Model               | CV Accuracy | Test Accuracy |
|--------------------|-------------|----------------|
| Decision Tree       | 53.6%       | 54.0%          |
| Random Forest       | 64.5%       | 64.8%          |
| **Logistic Regression** | **65.7%**   | **65.7%**      |
| Linear SVC          | 63.3%       | 63.4%          |

> ✅ **Final Model Chosen:** Logistic Regression

### Confusion Matrix Insights
- Strong performance on 5-star predictions
- Most confusion between adjacent ratings (e.g., 4-star vs. 5-star)
- Model leaned optimistic — likely due to review tone similarities
