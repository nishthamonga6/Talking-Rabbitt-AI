import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Rabbit AI", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "datasets.pkl")


def load_data():
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        try:
            with open(DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_data(data):
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)


datasets = load_data()

st.sidebar.title("🐇 Rabbit AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Upload Dataset",
        "Dataset Manager (CRUD)",
        "Analytics Dashboard",
        "AI Chat Analyst",
        "Data Explorer",
    ],
)

if menu == "Upload Dataset":
    st.title("Upload Business Data")

    files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    if files:
        for file in files:
            try:
                df = pd.read_csv(file, engine="python", on_bad_lines="skip")
                datasets[file.name] = df
                st.success(f"{file.name} uploaded")
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")

        save_data(datasets)

    if datasets:
        st.subheader("Uploaded Datasets")
        for name, df in datasets.items():
            st.write(name)
            st.dataframe(df.head())

elif menu == "Dataset Manager (CRUD)":
    st.title("Dataset Manager")

    if not datasets:
        st.warning("Upload dataset first")
        st.stop()

    dataset_names = list(datasets.keys())
    dataset = st.selectbox("Select Dataset", dataset_names)
    df = datasets[dataset]

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.divider()

    operation = st.radio(
        "Choose Operation",
        [
            "Add Row",
            "Update Row",
            "Delete Row",
            "Rename Dataset",
            "Delete Dataset",
        ],
    )

    if operation == "Add Row":
        st.subheader("Add New Row")

        new_data = {}
        for col in df.columns:
            new_data[col] = st.text_input(col)

        if st.button("Add Row"):
            new_row = pd.DataFrame([new_data])
            df = pd.concat([df, new_row], ignore_index=True)
            datasets[dataset] = df
            save_data(datasets)
            st.success("Row added successfully")

    elif operation == "Update Row":
        row_index = st.number_input(
            "Row Index",
            min_value=0,
            max_value=len(df) - 1,
            step=1,
        )

        updated_data = {}
        for col in df.columns:
            updated_data[col] = st.text_input(
                col,
                value=str(df.iloc[row_index][col]),
            )

        if st.button("Update Row"):
            for col in df.columns:
                df.at[row_index, col] = updated_data[col]

            datasets[dataset] = df
            save_data(datasets)
            st.success("Row updated successfully")

    elif operation == "Delete Row":
        row_index = st.number_input(
            "Row Index to Delete",
            min_value=0,
            max_value=len(df) - 1,
            step=1,
        )

        if st.button("Delete Row"):
            df = df.drop(row_index).reset_index(drop=True)
            datasets[dataset] = df
            save_data(datasets)
            st.success("Row deleted successfully")

    elif operation == "Rename Dataset":
        new_name = st.text_input("New Dataset Name")

        if st.button("Rename"):
            if new_name in datasets:
                st.error("Dataset name already exists")
            else:
                datasets[new_name] = datasets.pop(dataset)
                save_data(datasets)
                st.success("Dataset renamed")
                st.experimental_rerun()

    elif operation == "Delete Dataset":
        st.warning("This will permanently remove the dataset")
        confirm = st.checkbox("Confirm deletion")

        if confirm and st.button("Delete Dataset"):
            del datasets[dataset]
            save_data(datasets)
            st.success("Dataset deleted")
            st.experimental_rerun()

elif menu == "Analytics Dashboard":
    st.title("Analytics Dashboard")

    if not datasets:
        st.warning("Upload dataset first")
        st.stop()

    dataset = st.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[dataset]

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))

    if numeric:
        c3.metric("Total", round(df[numeric].sum().sum(), 2))

    st.divider()

    chart_type = st.selectbox(
        "Chart Type",
        ["Bar", "Line", "Scatter", "Histogram", "Box"],
    )

    x_col = st.selectbox("X Axis", df.columns)

    y_col = None
    if numeric:
        y_col = st.selectbox("Y Axis", numeric)

    fig = None

    if chart_type in ["Bar", "Line", "Scatter", "Box"] and not numeric:
        st.warning("This chart type requires at least one numeric column.")
    else:
        if chart_type == "Bar":
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col)
            else:
                st.warning("Please select a numeric Y axis for the bar chart.")

        elif chart_type == "Line":
            if y_col:
                fig = px.line(df, x=x_col, y=y_col)
            else:
                st.warning("Please select a numeric Y axis for the line chart.")

        elif chart_type == "Scatter":
            if y_col:
                fig = px.scatter(df, x=x_col, y=y_col)
            else:
                st.warning("Please select a numeric Y axis for the scatter plot.")

        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col)

        elif chart_type == "Box":
            if y_col:
                fig = px.box(df, x=x_col, y=y_col)
            else:
                st.warning("Please select a numeric Y axis for the box plot.")

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

elif menu == "AI Chat Analyst":
    st.title("🤖 Rabbit AI Chat Analyst")

    if not datasets:
        st.warning("Upload dataset first")
        st.stop()

    dataset = st.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[dataset]

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include="object").columns.tolist()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.info(
        """
Example questions:

• What is the total sales
• Show average revenue
• Show sales trend
• Show distribution of revenue
• Plot sales by category
"""
    )

    prompt = st.chat_input("Ask about your dataset")

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt},
        )

        with st.chat_message("user"):
            st.write(prompt)

        q = prompt.lower()

        with st.chat_message("assistant"):
            if "total" in q:
                if numeric:
                    st.write(df[numeric].sum())
                else:
                    st.write(
                        "This dataset has no numeric columns to calculate totals.",
                    )

            elif "average" in q:
                if numeric:
                    st.write(df[numeric].mean())
                else:
                    st.write(
                        "This dataset has no numeric columns to calculate averages.",
                    )

            elif "trend" in q:
                if numeric:
                    fig = px.line(df, y=numeric[0])
                    st.plotly_chart(fig)
                else:
                    st.write(
                        "This dataset has no numeric columns to plot a trend.",
                    )

            elif "distribution" in q:
                if numeric:
                    fig = px.box(df, y=numeric[0])
                    st.plotly_chart(fig)
                else:
                    st.write(
                        "This dataset has no numeric columns to show a distribution.",
                    )

            elif "by" in q:
                if categorical and numeric:
                    cat = categorical[0]
                    num = numeric[0]

                    chart = df.groupby(cat)[num].sum().reset_index()
                    fig = px.bar(chart, x=cat, y=num)
                    st.plotly_chart(fig)
                else:
                    st.write(
                        "Need at least one categorical and one numeric column to plot 'by' charts.",
                    )

            else:
                st.write(
                    "Try asking about totals, averages, trends, or charts.",
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": "Response generated"},
        )

elif menu == "Data Explorer":
    st.title("Data Explorer")

    if not datasets:
        st.warning("Upload dataset first")
        st.stop()

    dataset = st.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[dataset]

    st.subheader("Preview")
    st.dataframe(df)

    st.subheader("Column Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistics")
    st.write(df.describe())
