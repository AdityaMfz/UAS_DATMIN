import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Sidebar Navigasi
st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", use_container_width=True)
st.sidebar.title("Menu")
selected_page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Home", "Linear Regression", "Random Forest Regressor", "Decision Tree Regressor", "Support Vector Regressor", "Gradient Boosting Regressor"]
)

# Fungsi untuk Memuat Dataset
@st.cache_data
def load_dataset():
    data = pd.read_csv("Regression.csv")
    return data

# Load Data
data = load_dataset()

# Preprocessing
categorical_columns = ["sex", "smoker", "region"]
for column in categorical_columns:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column])

# Define target and features
target = "charges"
features = [col for col in data.columns if col != target]
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fungsi Evaluasi Model
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display evaluation metrics
    st.subheader(f"Evaluasi Model: {model_name}")
    st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"**R² (R-squared):** {r2_score(y_test, y_pred):.2f}")

    # Residual Plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    ax.axhline(0, linestyle="--", color="red")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot: {model_name}")
    st.pyplot(fig)

    # Actual vs Predicted Plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", color="red")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Actual vs Predicted: {model_name}")
    st.pyplot(fig)

# Halaman Home
if selected_page == "Home":
    st.title("Selamat Datang!")
    st.markdown(
        """
        ### Fitur Aplikasi:
        - Implementasi model regresi populer.
        - Visualisasi interaktif dan evaluasi performa.
        - Eksperimen parameter model.
        """
    )
    st.dataframe(data)

    # Visualisasi Distribusi Biaya Medis
    st.subheader("Distribusi Biaya Medis")
    fig = px.histogram(data, x="charges", nbins=30, title="Distribusi Biaya Medis", template="plotly_dark")
    st.plotly_chart(fig)

    # Distribusi Perokok
    st.subheader("Distribusi Perokok")
    fig, ax = plt.subplots()
    sns.countplot(x="smoker", data=data, palette="pastel", ax=ax)
    ax.set_title("Jumlah Perokok vs Non-Perokok")
    st.pyplot(fig)

     # **Grafik Usia Perokok Aktif**
    st.write("### Grafik Usia Perokok Aktif")
    smoker_age = data[data["smoker"] == 1]  # Filter untuk perokok aktif
    fig = px.histogram(
        smoker_age,
        x="age",
        nbins=15,
        title="Distribusi Usia Perokok Aktif",
        template="plotly_dark",
        color_discrete_sequence=["orange"],
    )
    st.plotly_chart(fig)

    # **Grafik Jenis Kelamin Perokok Aktif**
    st.write("### Grafik Jenis Kelamin Perokok Aktif")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(
        x="sex",
        data=smoker_age,
        palette="coolwarm",
        ax=ax,
    )
    ax.set_title("Distribusi Jenis Kelamin Perokok Aktif")
    ax.set_xlabel("Jenis Kelamin (0 = Female, 1 = Male)")
    ax.set_ylabel("Jumlah Perokok Aktif")
    st.pyplot(fig)

    # **Grafik Region Perokok Aktif**
    st.write("### Grafik Kota/Region Perokok Aktif")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        x="region",
        data=smoker_age,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Distribusi Perokok Aktif Berdasarkan Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Jumlah Perokok Aktif")
    st.pyplot(fig)

    # **Grafik BMI Para Perokok Aktif**
    st.write("### Grafik BMI Para Perokok Aktif")
    smoker_bmi = data[data["smoker"] == 1]  # Filter untuk perokok aktif
    fig = px.histogram(
        smoker_bmi,
        x="bmi",
        nbins=20,
        title="Distribusi BMI Perokok Aktif",
        template="plotly_dark",
        color_discrete_sequence=["teal"],
    )
    st.plotly_chart(fig)

    # Statistik BMI Perokok Aktif
    st.write("### Statistik BMI Perokok Aktif")
    st.write(smoker_bmi["bmi"].describe())

    # Scatter Plot BMI vs Biaya Medis (charges) untuk Perokok Aktif
    st.write("### Hubungan BMI dengan Biaya Medis untuk Perokok Aktif")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=smoker_bmi["bmi"], y=smoker_bmi["charges"], alpha=0.7, ax=ax)
    ax.set_title("Scatter Plot BMI vs Biaya Medis (Perokok Aktif)")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Biaya Medis (charges)")
    st.pyplot(fig)

# Halaman Linear Regression
elif selected_page == "Linear Regression":
    st.title("Linear Regression")
    model = LinearRegression()
    evaluate_model(model, "Linear Regression")

    # Heatmap Korelasi
    st.subheader("Korelasi antar Fitur")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Halaman Random Forest Regressor
elif selected_page == "Random Forest Regressor":
    st.title("Random Forest Regressor")
    n_estimators = st.slider("Jumlah Trees (n_estimators):", 10, 200, 100, step=10)
    max_depth = st.slider("Kedalaman Maksimum (max_depth):", 1, 20, 10, step=1)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    evaluate_model(model, "Random Forest Regressor")

    # Feature Importances
    st.subheader("Feature Importances")
    feature_importances = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
    fig = px.bar(feature_importances, x="Importance", y="Feature", orientation="h", title="Feature Importances")
    st.plotly_chart(fig)

# Halaman Decision Tree Regressor
elif selected_page == "Decision Tree Regressor":
    st.title("Decision Tree Regressor")
    max_depth = st.slider("Kedalaman Maksimum (max_depth):", 1, 20, 5, step=1)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    evaluate_model(model, "Decision Tree Regressor")

    # Visualisasi Pohon Keputusan
    st.subheader("Visualisasi Pohon Keputusan")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, feature_names=features, filled=True, ax=ax)
    st.pyplot(fig)

# Halaman Support Vector Regressor
elif selected_page == "Support Vector Regressor":
    st.title("Support Vector Regressor")
    kernel = st.selectbox("Pilih Kernel:", ["linear", "poly", "rbf", "sigmoid"])
    C = st.slider("Nilai C:", 0.1, 10.0, 1.0, step=0.1)
    model = SVR(kernel=kernel, C=C)
    evaluate_model(model, "Support Vector Regressor")

# Halaman Gradient Boosting Regressor
elif selected_page == "Gradient Boosting Regressor":
    st.title("Gradient Boosting Regressor")
    learning_rate = st.slider("Learning Rate:", 0.01, 0.3, 0.1, step=0.01)
    n_estimators = st.slider("Jumlah Trees (n_estimators):", 10, 200, 100, step=10)
    model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)
    evaluate_model(model, "Gradient Boosting Regressor")
