# Company Classification System

This project implements a neural network-based classification system for companies using their descriptions and metadata. The system uses a combination of heuristic-based initial labeling and deep learning for accurate classification.

## Project Structure

```
.
├── data/
│   ├── labels.txt           # List of possible company labels
│   └── ml_insurance_challenge.csv  # Company data
├── src/
│   ├── main.py             # Main application and model training
│   ├── heuristics.py       # Initial labeling logic
│   └── printStuff.py       # Utility functions for printing results
└── README.md
```

## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn

Install dependencies using:
```bash
pip install tensorflow numpy pandas scikit-learn
```

## How It Works

### 1. Initial Labeling (Heuristics)

The system first uses a heuristic-based approach (`DFT` function in `heuristics.py`) to assign initial labels to companies. The scoring system works as follows:

- Description match: +0.1 points
- Business tag match: +0.1 points per match
- Sector match: +0.5 points
- Category match: +1.0 points
- Niche match: +1.5 points

A company is assigned a label if its total score exceeds a threshold (default: 0.5). This approach ensures that:
- Companies are only labeled when there's sufficient evidence
- Different types of matches have different weights based on their importance
- The system can handle cases where multiple fields contribute to the classification

The threshold value of 0.5 was selected based on analysis of the relationship between threshold values and the number of classified companies. As shown in [threshold-plot.pdf](src/threshold-plot.pdf), this value represents an optimal balance point where:
- The initial rapid decline in classified companies has stabilized
- The remaining companies have strong enough evidence for reliable labeling
- The system maintains a good balance between precision and coverage

### 2. Neural Network Model

The classification model uses a deep learning approach with the following architecture:

```python
Sequential([
    Embedding(10000, 128, input_length=200),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
```

#### Architecture Choices:

1. **Embedding Layer**
   - Vocabulary size: 10,000 most frequent words
   - Embedding dimension: 128
   - Input length: 200 tokens
   - Rationale: Captures semantic relationships between words while keeping the model size manageable

2. **LSTM Layers**
   - First LSTM: 64 units with return_sequences=True
   - Second LSTM: 32 units
   - Rationale: 
     - Two-layer LSTM captures both local and global patterns in the text
     - Decreasing units in the second layer helps in feature compression
     - return_sequences=True in first layer allows the second LSTM to see the full sequence

3. **Dropout Layers**
   - Rate: 0.2
   - Rationale: Prevents overfitting by randomly dropping units during training

4. **Output Layer**
   - Softmax activation
   - Number of units equal to unique labels
   - Rationale: Provides probability distribution over possible labels

### 3. Training Process

1. **Data Preparation**
   - Combines company description, business tags, sector, category, and niche into a single text input
   - Tokenizes and pads sequences to a fixed length of 200 tokens
   - Splits data into 80% training and 20% testing sets

2. **Model Training**
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Batch size: 32
   - Epochs: 10
   - Validation split: 20% of training data

## Usage

1. Prepare your data:
   - Place company data in `data/ml_insurance_challenge.csv`
   - Place possible labels in `data/labels.txt`

2. Run the classification:
   ```bash
   python src/main.py
   ```

3. The program will:
   - Apply initial labeling using heuristics
   - Train the neural network model
   - Print accuracy metrics and label distribution
   - Make the prediction function available for new companies

## Prediction

To predict labels for new companies, use the `predict_company_label` function:

```python
predicted_label = predict_company_label(company_text)
```

## Performance Considerations

- The model uses only companies that have been confidently labeled by the heuristic system
- The threshold (0.5) can be adjusted to balance between precision and recall
- The model architecture can be tuned based on the specific dataset characteristics

## Future Improvements

1. Implement cross-validation for more robust performance evaluation
2. Add hyperparameter tuning capabilities
3. Implement early stopping to prevent overfitting
4. Add support for multi-label classification
5. Implement model persistence for reuse
