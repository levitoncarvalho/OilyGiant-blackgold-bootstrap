# 🛢️ OilyGiant — Black Gold Bootstrap
### *Finding the Most Profitable Oil Region Through Machine Learning & Risk Simulation*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Bootstrap-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://oilygiant-blackgold-bootstrap-levitoncarvalho.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br>

**[🚀 Try the Live App](https://oilygiant-blackgold-bootstrap-levitoncarvalho.streamlit.app/)** &nbsp;|&nbsp; **[📓 View Full Notebook](https://github.com/levitoncarvalho/OilyGiant-blackgold-bootstrap/blob/main/notebook/exploration_v1.ipynb)**

</div>

---

> ⚠️ **Disclaimer:** OilyGiant is a **fictional mining company** created exclusively for academic and portfolio purposes. This project was developed as part of a Data Science training program and is intended solely to demonstrate technical skills in machine learning, financial analytics, and statistical risk assessment. No real company, geological data, or business relationship is represented here.

---

## 🧩 The Business Problem

> *"Drilling the wrong well doesn't just cost money — it destroys capital at scale."*

**OilyGiant**, a fictional mining company, needed to make a high-stakes decision: **which of three candidate regions should be chosen for the development of 200 new oil wells?**

With a fixed budget of **$100,000,000 USD** and zero tolerance for excessive financial risk, the company needed a data-driven framework to:

1. Predict the volume of reserves in unexplored wells using geological survey data.
2. Simulate the financial outcome of the real-world exploration strategy (sampling 500 points, selecting 200).
3. Quantify the **risk of loss** for each region using statistical bootstrapping.
4. Recommend the region that maximizes profit while staying within the **2.5% maximum risk threshold**.

### Why Machine Learning?

Random drilling is financially catastrophic. The break-even volume per well is **111.11 thousand barrels** — far above the regional averages. Only a model capable of pinpointing the highest-yield wells can make this investment viable.

---

## 📊 Final Results

<div align="center">

### 🏆 Region 1 was the only approved region — and the clear winner

</div>

| Region | Expected Profit (USD) | 95% CI — Lower Bound | Risk of Loss | Status |
|---|---|---|---|---|
| Region 0 | ~$4,080,000 | Negative | **5.20%** | ❌ Rejected |
| **Region 1** | **~$4,730,000** | **Positive** | **1.20%** | **✅ Approved** |
| Region 2 | ~$4,280,000 | Negative | **6.10%** | ❌ Rejected |

> **Region 1 is the only region where the 95% confidence interval lower bound remains positive**, meaning even in unfavorable sampling scenarios the investment does not result in a loss. Regions 0 and 2 both breach the 2.5% risk limit and expose the company to significant financial downside.

---

## 🧠 End-to-End Data Science Workflow

```
📥 Raw Data  →  🔎 EDA  →  🛠️ Preprocessing  →  📐 Linear Regression  →  💰 Break-Even Analysis  →  📊 Bootstrapping  →  🎯 Final Recommendation  →  🚀 Deployment
```

---

### 1. 🔎 Exploratory Data Analysis (EDA)

Three geological datasets were provided — one per region — each containing:

- **100,000 records** of candidate well locations
- **3 geological features** (`f0`, `f1`, `f2`) — anonymized exploration metrics
- **1 target variable** (`product`) — volume of reserves in thousands of barrels
- **1 identifier column** (`id`) — removed during preprocessing (no predictive value)

**Key findings:**

- No missing values or duplicate rows in any of the three datasets.
- The `id` column was dropped from all DataFrames before modeling.
- Regional distributions differ significantly — Region 1 exhibits a much tighter, more predictable reserve distribution, while Regions 0 and 2 are far more dispersed.

**Feature Overview:**

| Feature | Type | Description |
|---|---|---|
| `f0` | Numerical | Geological exploration metric A |
| `f1` | Numerical | Geological exploration metric B |
| `f2` | Numerical | Geological exploration metric C |
| `product` | Numerical (Target) | Volume of oil reserves (thousands of barrels) |

---

### 2. 📐 Model Training & Validation

A **Linear Regression** model was used exclusively — as mandated by the business constraint requiring predictability and interpretability. A reusable `train_and_evaluate()` function was built to split each region's data (75% train / 25% validation), train the model, and compute the RMSE and average predicted volume.

**Modeling Results:**

| Region | Average Predicted Volume | RMSE | Interpretation |
|---|---|---|---|
| Region 0 | ~92.6k barrels | ~37.6k barrels | High volume, low precision |
| **Region 1** | **~68.7k barrels** | **~0.89k barrels** | Lower volume, near-perfect precision |
| Region 2 | ~95.0k barrels | ~40.1k barrels | High volume, low precision |

**Critical insight:** Despite Regions 0 and 2 showing higher average volumes, their RMSE values are **40–45 times higher** than Region 1's. This means the model cannot reliably identify which specific wells in those regions will actually be profitable — which becomes a decisive factor in the financial risk analysis.

---

### 3. 💰 Break-Even Analysis

Before any financial projection, the minimum viable production volume per well was calculated:

```
Total Budget:         $100,000,000 USD
Wells to Develop:     200
Cost per well:        $500,000 USD
Revenue per unit:     $4,500 USD (per 1,000 barrels)
Break-Even Volume:    111.11 thousand barrels per well
```

Since all regional averages fall **below** the 111.11k break-even threshold, random drilling would guarantee financial losses. The ML model's role is to identify the concentrated pockets of high-yield wells that exceed this threshold — turning a loss-making region into a profitable investment.

---

### 4. 📊 Risk Assessment via Bootstrapping

The bootstrapping simulation (1,000 iterations) replicates OilyGiant's actual field exploration strategy:

1. **Sample** 500 candidate wells randomly from the validation set.
2. **Select** the top 200 wells based on the model's predictions.
3. **Calculate** profit using the true reserve volumes of those 200 wells.
4. **Repeat** 1,000 times to build a profit distribution.

From each distribution, three key statistics were extracted: **average expected profit**, **95% confidence interval**, and **risk of loss** (probability of negative profit).

**Bootstrapping Results:**

```
--- Region 0 ---
Expected Average Profit:  USD ~4,080,000
95% Confidence Interval:  [Negative] to [Positive]
Risk of Loss:             5.20%
STATUS: ❌ REJECTED (Risk > 2.5%)

--- Region 1 ---
Expected Average Profit:  USD ~4,730,000
95% Confidence Interval:  [Positive] to [Positive]
Risk of Loss:             1.20%
STATUS: ✅ APPROVED (Risk < 2.5%)

--- Region 2 ---
Expected Average Profit:  USD ~4,280,000
95% Confidence Interval:  [Negative] to [Positive]
Risk of Loss:             6.10%
STATUS: ❌ REJECTED (Risk > 2.5%)
```

---

## 🎯 Conclusion & Final Recommendation

The goal of this project was to identify the safest and most profitable region for OilyGiant's new wells, with a strict risk ceiling of 2.5%.

**Region 1 is unequivocally the recommended location for development.**

Although Regions 0 and 2 initially appeared more attractive due to their higher raw volume averages, the modeling phase revealed their fundamental weakness: massive prediction errors (RMSE > 37k). The Linear Regression model cannot reliably distinguish profitable wells from unprofitable ones in those regions — making them high-risk gambles rather than sound investments.

Region 1's near-perfect predictability (RMSE < 1k) is the cornerstone of its safety profile. When the model says a well in Region 1 will be productive, it almost certainly will be. This precision directly translates into controlled risk during the bootstrapping phase: **only Region 1 satisfies the 2.5% loss-risk requirement**, with a risk of just 1.2% and a confidence interval whose lower bound remains firmly in profitable territory.

> **The lesson is counterintuitive but powerful:** in oil exploration, model precision matters more than raw volume potential. A region where you can predict *with certainty* which wells are worth drilling is more valuable than a region with higher theoretical yields but unpredictable outcomes.

---

### 5. 🚀 Deployment

The final model and business logic were serialized with `joblib` and deployed as an interactive **Streamlit web app**, allowing analysts to input well data for any region and receive real-time profitability and risk predictions.

**[➡️ Try the live app here](https://oilygiant-blackgold-bootstrap-levitoncarvalho.streamlit.app/)**

---

## 🗂️ Project Structure

```text
OilyGiant-blackgold-bootstrap/
│
├── 📂 .devcontainer/
│   └── ⚙️ devcontainer.json                        # Dev container configuration
│
├── 📂 .streamlit/
│   └── ⚙️ config.toml                              # Streamlit app configuration
│
├── 📂 data/
│   ├── 📊 geo_data_0.csv                           # Geological data — Region 0
│   ├── 📊 geo_data_1.csv                           # Geological data — Region 1
│   └── 📊 geo_data_2.csv                           # Geological data — Region 2
│
├── 📓 notebook/
│   └── 📓 exploration_v1.ipynb                     # Full analysis + modeling + bootstrap
│
├── 🐍 src/
│   ├── 🐍 __init__.py                              # Package initialization
│   ├── 🐍 bootstrap.py                             # Bootstrapping risk simulation
│   ├── 🐍 config.py                                # Centralized configuration
│   ├── 🐍 data_loader.py                           # Data loading pipeline
│   ├── 🐍 profit.py                                # Profit calculation logic
│   └── 🐍 train_model.py                           # Model training & evaluation
│
├── 🚫 .gitignore                                   # Ignored files and folders
├── ⚖️ LICENSE                                      # MIT License
├── 📄 README.md                                    # Project documentation
├── 🌐 app.py                                       # Streamlit interactive app
├── 🖼️ blackgold-price.png                          # visual asset
├── 🐍 main.py                                      # Main execution script
└── 📦 requirements.txt                             # Python dependencies
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/levitoncarvalho/OilyGiant-blackgold-bootstrap.git
cd OilyGiant-blackgold-bootstrap

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train and save models
python train_and_save.py

# 5. Launch the app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (Linear Regression) |
| **Risk Simulation** | NumPy (Bootstrapping, 1000 iterations) |
| **Visualization** | Matplotlib, Seaborn |
| **Serialization** | Joblib |
| **Deployment** | Streamlit, Streamlit Community Cloud |

---

## 💡 Key Technical Takeaways

- **Model precision beats raw volume potential** — in high-stakes financial scenarios, a model with a low RMSE is more actionable than one predicting high but uncertain values. Predictability is itself a form of value.
- **Bootstrapping is essential for financial risk quantification** — simple point estimates of profit are insufficient for real business decisions. Simulating thousands of sampling scenarios reveals the true distribution of outcomes and the probability of loss.
- **Break-even analysis anchors the modeling objective** — knowing that 111.11k barrels is the minimum viable production per well transforms model selection from an academic exercise into a financially grounded decision.
- **The modular code structure** (`src/`) promotes maintainability, testability, and scalability — a best practice for production ML systems.
- **Linear Regression can outperform complex models** when the underlying data structure is well-behaved — as demonstrated by Region 1's near-zero RMSE, simplicity is often the most powerful tool.

---

## 👨‍💻 Author

<div align="center">

**Leviton Lima Carvalho**
*Data Scientist | Machine Learning | Python*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-levitoncarvalho-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/levitoncarvalho/)
[![GitHub](https://img.shields.io/badge/GitHub-levitoncarvalho-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/levitoncarvalho)
[![Email](https://img.shields.io/badge/Email-levitoncarvalho@icloud.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:levitoncarvalho@icloud.com)

</div>

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more details.
