import io
import re
from typing import Optional
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

try:
    import ollama
except ImportError:
    ollama = None

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


st.set_page_config(layout="wide", page_title="AI Data Analyst Dashboard")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 14px;
    padding: 0.9rem;
    color: white; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Data Analyst Dashboard")
st.caption(
    "Upload a CSV or Excel file to profile the data, clean it, visualize it, test hypotheses, "
    "chat with it, and export a report."
)


def display_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.select_dtypes(include=["object", "category"]).columns:
        safe[col] = safe[col].astype(str)
    return safe


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def match_columns(question: str, columns: pd.Index) -> list[str]:
    normalized_question = normalize_text(question)
    matches = []
    for col in columns:
        normalized_col = normalize_text(col)
        if normalized_col and normalized_col in normalized_question:
            matches.append(col)
    return matches


@st.cache_data(show_spinner=False)
def load_dataset(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


def get_profile(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical": int(df.select_dtypes(exclude=np.number).shape[1]),
    }


def sample_values(series: pd.Series, limit: int = 3) -> str:
    values = [str(v) for v in series.dropna().astype(str).unique()[:limit]]
    return ", ".join(values) if values else "-"


def detect_date_like_columns(df):
    converted = df.copy()
    converted_columns = []

    for col in converted.select_dtypes(include=["object"]).columns:
        sample = converted[col].dropna().astype(str).head(20)

        # only check if looks like date
        if not sample.str.contains(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", regex=True).any():
            continue

        parsed = pd.to_datetime(sample, errors="coerce")

        if parsed.notna().mean() > 0.8:
            converted[col] = pd.to_datetime(converted[col], errors="coerce")
            converted_columns.append(col)

    return converted, converted_columns


def apply_cleaning(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool,
    trim_text: bool,
    numeric_strategy: str,
    categorical_strategy: str,
    detect_dates: bool,
) -> tuple[pd.DataFrame, list[str]]:
    cleaned = df.copy()
    notes: list[str] = []

    if trim_text:
        object_cols = cleaned.select_dtypes(include=["object", "category"]).columns
        for col in object_cols:
            cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        notes.append("Trimmed whitespace from text columns.")

    if detect_dates:
        cleaned, converted_columns = detect_date_like_columns(cleaned)
        if converted_columns:
            notes.append(f"Converted date-like columns: {', '.join(converted_columns)}.")

    if drop_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        removed = before - len(cleaned)
        notes.append(f"Removed {removed} duplicate rows.")

    numeric_cols = cleaned.select_dtypes(include=np.number).columns
    if numeric_strategy != "Do nothing":
        for col in numeric_cols:
            if cleaned[col].isna().any():
                if numeric_strategy == "Median":
                    fill_value = cleaned[col].median()
                elif numeric_strategy == "Mean":
                    fill_value = cleaned[col].mean()
                else:
                    fill_value = 0
                cleaned[col] = cleaned[col].fillna(fill_value)
        notes.append(f"Filled missing numeric values using: {numeric_strategy.lower()}.")

    categorical_cols = cleaned.select_dtypes(exclude=np.number).columns
    if categorical_strategy != "Do nothing":
        for col in categorical_cols:
            if cleaned[col].isna().any():
                if categorical_strategy == "Mode":
                    mode = cleaned[col].mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else "Missing"
                else:
                    fill_value = "Missing"
                cleaned[col] = cleaned[col].fillna(fill_value)
        notes.append(f"Filled missing categorical values using: {categorical_strategy.lower()}.")

    if not notes:
        notes.append("No cleaning operations were applied.")

    return cleaned, notes


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Filters")
    st.sidebar.caption("Filters affect the analysis pages only. Cleaning always edits the full working dataset.")

    filter_columns = st.sidebar.multiselect(
        "Columns to filter",
        options=list(df.columns),
        key="filter_columns",
    )

    filtered = df.copy()
    for col in filter_columns:
        series = filtered[col]

        if is_numeric_dtype(series):
            valid = series.dropna()
            if valid.empty:
                continue
            min_val = float(valid.min())
            max_val = float(valid.max())
            if min_val == max_val:
                st.sidebar.info(f"{col} has one value: {min_val}")
                continue
            selected_range = st.sidebar.slider(
                f"{col} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                key=f"filter_{col}",
            )
            filtered = filtered[filtered[col].between(selected_range[0], selected_range[1], inclusive="both")]

        elif is_datetime64_any_dtype(series):
            valid = series.dropna()
            if valid.empty:
                continue
            min_date = valid.min().date()
            max_date = valid.max().date()
            selected_dates = st.sidebar.date_input(
                f"{col} date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key=f"filter_{col}",
            )
            if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
                mask = series.dt.date.between(start_date, end_date)
                filtered = filtered[mask]

        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            if not options:
                continue
            selected_values = st.sidebar.multiselect(
                f"{col} values",
                options=options,
                default=options,
                key=f"filter_{col}",
            )
            filtered = filtered[series.astype(str).isin(selected_values)]

    return filtered


def build_ai_context(df: pd.DataFrame) -> str:
    profile = get_profile(df)
    numeric_df = df.select_dtypes(include=np.number)
    categorical_df = df.select_dtypes(exclude=np.number)

    numeric_summary = (
        numeric_df.describe().round(2).to_string()
        if not numeric_df.empty
        else "No numeric columns."
    )

    categorical_lines = []
    for col in categorical_df.columns[:5]:
        counts = df[col].astype(str).value_counts(dropna=False).head(5).to_dict()
        categorical_lines.append(f"{col}: {counts}")
    categorical_summary = "\n".join(categorical_lines) if categorical_lines else "No categorical columns."

    missing_by_column = df.isna().sum()
    missing_summary = missing_by_column[missing_by_column > 0].sort_values(ascending=False).head(10)
    missing_text = missing_summary.to_string() if not missing_summary.empty else "No missing values."

    sample_rows = df.head(8).to_csv(index=False)

    return (
        f"Rows: {profile['rows']}\n"
        f"Columns: {profile['columns']}\n"
        f"Missing values: {profile['missing']}\n"
        f"Duplicate rows: {profile['duplicates']}\n\n"
        f"Numeric summary:\n{numeric_summary}\n\n"
        f"Categorical summary:\n{categorical_summary}\n\n"
        f"Missing by column:\n{missing_text}\n\n"
        f"Sample rows:\n{sample_rows}"
    )


def run_ollama(prompt: str, model_name: str) -> tuple[Optional[str], Optional[str]]:
    if ollama is None:
        return None, "The ollama package is not installed in this environment."
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"], None
    except Exception as exc:
        return None, str(exc)


def answer_question(df: pd.DataFrame, question: str) -> tuple[Optional[str], Optional[pd.DataFrame]]:
    q = question.lower()
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    matched_columns = match_columns(question, df.columns)
    matched_numeric = [col for col in matched_columns if col in numeric_cols]
    matched_non_numeric = [col for col in matched_columns if col not in numeric_cols]

    if "missing" in q:
        if matched_columns:
            col = matched_columns[0]
            missing = int(df[col].isna().sum())
            return f"Missing values in `{col}`: {missing}", None
        return f"Total missing values: {int(df.isna().sum().sum())}", None

    if "duplicate" in q:
        return f"Duplicate rows: {int(df.duplicated().sum())}", None

    if any(phrase in q for phrase in ["how many rows", "number of rows", "row count", "total rows"]):
        return f"Total rows: {len(df)}", None

    if any(phrase in q for phrase in ["how many columns", "number of columns", "column count"]):
        return f"Total columns: {df.shape[1]}", None

    if "unique" in q or "distinct" in q:
        if matched_columns:
            col = matched_columns[0]
            return f"Unique values in `{col}`: {int(df[col].nunique(dropna=True))}", None

    if any(word in q for word in ["average", "mean"]) and matched_numeric:
        col = matched_numeric[0]
        return f"Average `{col}`: {df[col].mean():.4f}", None

    if "median" in q and matched_numeric:
        col = matched_numeric[0]
        return f"Median `{col}`: {df[col].median():.4f}", None

    if "sum" in q and matched_numeric:
        col = matched_numeric[0]
        return f"Sum of `{col}`: {df[col].sum():.4f}", None

    if any(word in q for word in ["max", "highest", "largest"]) and matched_numeric:
        col = matched_numeric[0]
        value = df[col].max()
        row = df[df[col] == value].head(1)
        return f"Maximum `{col}`: {value}", row

    if any(word in q for word in ["min", "lowest", "smallest"]) and matched_numeric:
        col = matched_numeric[0]
        value = df[col].min()
        row = df[df[col] == value].head(1)
        return f"Minimum `{col}`: {value}", row

    if any(phrase in q for phrase in ["most common", "top value", "top category"]) and matched_non_numeric:
        col = matched_non_numeric[0]
        counts = df[col].astype(str).value_counts(dropna=False).head(5).reset_index(name="count")
        counts.columns = [col, "count"]
        top_value = counts.iloc[0][col]
        top_count = int(counts.iloc[0]["count"])
        return f"Most common value in `{col}`: {top_value} ({top_count} rows)", counts

    if ("correlation" in q or "relationship" in q) and len(matched_numeric) >= 2:
        col1, col2 = matched_numeric[:2]
        corr = df[[col1, col2]].corr().iloc[0, 1]
        return f"Correlation between `{col1}` and `{col2}`: {corr:.4f}", None

    return None, None


def build_pdf_report(df: pd.DataFrame, ai_text: str, scope_label: str) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed.")

    profile = get_profile(df)
    numeric_only = df.select_dtypes(include=np.number)
    numeric_summary = numeric_only.describe().round(2) if not numeric_only.empty else pd.DataFrame()
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(10)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AI Data Analyst Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Scope: {escape(scope_label)}", styles["BodyText"]),
        Spacer(1, 6),
        Paragraph(f"Rows: {profile['rows']}", styles["BodyText"]),
        Paragraph(f"Columns: {profile['columns']}", styles["BodyText"]),
        Paragraph(f"Missing values: {profile['missing']}", styles["BodyText"]),
        Paragraph(f"Duplicate rows: {profile['duplicates']}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Top Missing Columns", styles["Heading2"]),
        Spacer(1, 6),
    ]

    if missing.empty:
        story.append(Paragraph("No missing values detected.", styles["BodyText"]))
    else:
        for col, value in missing.items():
            story.append(Paragraph(f"{escape(str(col))}: {int(value)}", styles["BodyText"]))

    story.extend(
        [
            Spacer(1, 12),
            Paragraph("Numeric Summary", styles["Heading2"]),
            Spacer(1, 6),
        ]
    )

    if numeric_summary.empty:
        story.append(Paragraph("No numeric summary available.", styles["BodyText"]))
    else:
        for row in numeric_summary.reset_index().to_dict("records"):
            line = ", ".join([f"{key}={value}" for key, value in row.items()])
            story.append(Paragraph(escape(line), styles["BodyText"]))
            story.append(Spacer(1, 4))

    story.extend(
        [
            Spacer(1, 12),
            Paragraph("AI Narrative", styles["Heading2"]),
            Spacer(1, 6),
        ]
    )

    narrative = ai_text.strip() if ai_text.strip() else "No AI narrative was included in this report."
    for block in [part.strip() for part in narrative.split("\n\n") if part.strip()]:
        story.append(Paragraph(escape(block).replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def reset_state_for_new_file(df: pd.DataFrame, signature: tuple[str, int]) -> None:
    st.session_state.file_signature = signature
    st.session_state.source_df = df.copy()
    st.session_state.working_df = df.copy()
    st.session_state.chat_history = []
    st.session_state.latest_ai_insight = ""
    st.session_state.report_bytes = None
    st.session_state.report_preview = ""


st.sidebar.title("Controls")
page = st.sidebar.radio(
    "Sections",
    [
        "Overview",
        "Data Profile",
        "Cleaning",
        "Visualization",
        "Statistics",
        "AI Insights",
        "Chat",
        "Report",
    ],
)

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if not uploaded_file:
    st.info("Upload a dataset from the sidebar to start exploring it.")
    st.stop()

signature = (uploaded_file.name, uploaded_file.size)
dataset = load_dataset(uploaded_file.name, uploaded_file.getvalue())

if st.session_state.get("file_signature") != signature:
    reset_state_for_new_file(dataset, signature)

working_df = st.session_state.working_df.copy()
analysis_df = apply_sidebar_filters(working_df)

if analysis_df is None or len(analysis_df) == 0:
    analysis_df = working_df.copy()



st.sidebar.markdown("---")
st.sidebar.download_button(
    "Download working data as CSV",
    data=to_csv_bytes(working_df),
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)
st.sidebar.download_button(
    "Download filtered data as CSV",
    data=to_csv_bytes(analysis_df),
    file_name="filtered_dataset.csv",
    mime="text/csv",
)

if analysis_df.empty:
    st.warning("The active filters returned zero rows. Clear or relax the filters in the sidebar.")
    st.stop()


if page == "Overview":
    profile = get_profile(working_df)
    filtered_profile = get_profile(analysis_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", filtered_profile["rows"], delta=filtered_profile["rows"] - profile["rows"])
    c2.metric("Columns", filtered_profile["columns"])
    c3.metric("Missing", filtered_profile["missing"])
    c4.metric("Duplicates", filtered_profile["duplicates"])

    c5, c6, c7 = st.columns(3)
    c5.metric("Numeric Columns", filtered_profile["numeric"])
    c6.metric("Categorical Columns", filtered_profile["categorical"])
    c7.metric("Filter Retention", f"{(len(analysis_df) / max(len(working_df), 1)) * 100:.1f}%")

    st.subheader("Preview")
    preview_rows = st.slider(
        "Rows to preview",
        min_value=1,
        max_value=min(100, len(analysis_df)),
        value=min(10, len(analysis_df)),
    )
    st.dataframe(display_safe(analysis_df.head(preview_rows)), width="stretch")

    st.subheader("Quick Data Health")
    quick_health = pd.DataFrame(
        {
            "Metric": ["Rows", "Columns", "Missing", "Duplicates"],
            "Working Dataset": [profile["rows"], profile["columns"], profile["missing"], profile["duplicates"]],
            "Filtered Dataset": [
                filtered_profile["rows"],
                filtered_profile["columns"],
                filtered_profile["missing"],
                filtered_profile["duplicates"],
            ],
        }
    )
    st.dataframe(quick_health, width="stretch", hide_index=True)

elif page == "Data Profile":
    st.subheader("Schema Summary")
    schema = pd.DataFrame(
        [
            {
                "Column": col,
                "Type": str(analysis_df[col].dtype),
                "Missing": int(analysis_df[col].isna().sum()),
                "Missing %": round(float(analysis_df[col].isna().mean() * 100), 2),
                "Unique": int(analysis_df[col].nunique(dropna=True)),
                "Sample Values": sample_values(analysis_df[col]),
            }
            for col in analysis_df.columns
        ]
    )
    st.dataframe(display_safe(schema), width="stretch")

    numeric_part = analysis_df.select_dtypes(include=np.number)
    if not numeric_part.empty:
        st.subheader("Numeric Summary")
        st.dataframe(display_safe(numeric_part.describe().T.round(2)), width="stretch")

    categorical_part = analysis_df.select_dtypes(exclude=np.number)
    if not categorical_part.empty:
        st.subheader("Top Categories")
        cat_col = st.selectbox("Categorical column", categorical_part.columns, key="profile_cat_col")
        counts = categorical_part[cat_col].astype(str).value_counts(dropna=False).head(15).reset_index(name="count")
        counts.columns = [cat_col, "count"]
        st.dataframe(display_safe(counts), width="stretch", hide_index=True)

elif page == "Cleaning":
    st.subheader("Current Working Data")
    st.dataframe(display_safe(working_df.head(10)), width="stretch")

    with st.form("cleaning_form"):
        drop_dupes = st.checkbox("Drop duplicate rows", value=True)
        trim_text = st.checkbox("Trim whitespace in text columns", value=True)
        detect_dates = st.checkbox("Convert date-like text columns", value=True)
        numeric_strategy = st.selectbox(
            "Numeric missing values",
            ["Median", "Mean", "Zero", "Do nothing"],
            index=0,
        )
        categorical_strategy = st.selectbox(
            "Categorical missing values",
            ["Mode", "Label as Missing", "Do nothing"],
            index=0,
        )
        apply_changes = st.form_submit_button("Apply Cleaning")

    if apply_changes:
        cat_strategy_value = "Label as Missing" if categorical_strategy == "Label as Missing" else categorical_strategy
        cleaned, notes = apply_cleaning(
            st.session_state.working_df,
            drop_duplicates=drop_dupes,
            trim_text=trim_text,
            numeric_strategy=numeric_strategy,
            categorical_strategy=cat_strategy_value,
            detect_dates=detect_dates,
        )
        st.session_state.working_df = cleaned
        working_df = cleaned
        st.success("Cleaning steps applied to the working dataset.")
        for note in notes:
            st.write(f"- {note}")

    if st.button("Restore Original Dataset"):
        st.session_state.working_df = st.session_state.source_df.copy()
        st.success("Working dataset restored to the original uploaded file.")
        working_df = st.session_state.working_df

    st.subheader("After Cleaning")
    st.dataframe(display_safe(working_df.head(10)), width="stretch")

elif page == "Visualization":
    numeric_cols = list(analysis_df.select_dtypes(include=np.number).columns)
    categorical_cols = list(analysis_df.select_dtypes(exclude=np.number).columns)

    tab1, tab2, tab3 = st.tabs(["Distribution", "Relationship", "Correlation"])

    with tab1:
        feature = st.selectbox("Column to explore", analysis_df.columns, key="dist_feature")
        if is_numeric_dtype(analysis_df[feature]):
            chart_type = st.radio("Chart", ["Histogram", "Box", "Violin"], horizontal=True)
            if chart_type == "Histogram":
                fig = px.histogram(analysis_df, x=feature, nbins=30, marginal="box", title=f"Distribution of {feature}")
            elif chart_type == "Box":
                fig = px.box(analysis_df, y=feature, title=f"Box Plot of {feature}")
            else:
                fig = px.violin(analysis_df, y=feature, box=True, title=f"Violin Plot of {feature}")
        else:
            counts = analysis_df[feature].astype(str).value_counts(dropna=False).head(20).reset_index(name="count")
            counts.columns = [feature, "count"]
            fig = px.bar(counts, x=feature, y="count", title=f"Top Categories in {feature}")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        x_col = st.selectbox("X-axis", analysis_df.columns, key="rel_x")
        y_options = ["None"] + numeric_cols
        y_col = st.selectbox("Y-axis", y_options, key="rel_y")
        color_options = ["None"] + categorical_cols
        color_col = st.selectbox("Color by", color_options, key="rel_color")
        color_arg = None if color_col == "None" else color_col

        if y_col == "None":
            if is_numeric_dtype(analysis_df[x_col]):
                fig = px.histogram(analysis_df, x=x_col, color=color_arg, nbins=30, title=f"Histogram of {x_col}")
            else:
                counts = analysis_df[x_col].astype(str).value_counts(dropna=False).head(20).reset_index(name="count")
                counts.columns = [x_col, "count"]
                fig = px.bar(counts, x=x_col, y="count", title=f"Counts for {x_col}")
        else:
            if is_numeric_dtype(analysis_df[x_col]):
                fig = px.scatter(
                    analysis_df,
                    x=x_col,
                    y=y_col,
                    color=color_arg,
                    title=f"{y_col} vs {x_col}",
                )
            else:
                fig = px.box(
                    analysis_df,
                    x=x_col,
                    y=y_col,
                    color=color_arg,
                    title=f"{y_col} by {x_col}",
                )
        st.plotly_chart(fig, width="stretch")

    with tab3:
        if len(numeric_cols) >= 2:
            corr = analysis_df[numeric_cols].corr().round(2)
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Heatmap",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Add at least two numeric columns to view a correlation heatmap.")

elif page == "Statistics":
    if stats is None:
        st.warning("SciPy is not installed, so hypothesis testing is unavailable.")
    else:
        numeric_cols = list(analysis_df.select_dtypes(include=np.number).columns)
        categorical_cols = list(analysis_df.select_dtypes(exclude=np.number).columns)
        analysis_type = st.selectbox("Choose analysis", ["Correlation", "Two-group t-test", "One-way ANOVA"])

        if analysis_type == "Correlation":
            if len(numeric_cols) < 2:
                st.info("At least two numeric columns are required.")
            else:
                col1 = st.selectbox("First numeric column", numeric_cols, key="corr_col1")
                col2 = st.selectbox(
                    "Second numeric column",
                    [col for col in numeric_cols if col != col1],
                    key="corr_col2",
                )
                sample = analysis_df[[col1, col2]].dropna()
                if sample[col1].nunique() < 2 or sample[col2].nunique() < 2:
                    st.warning("Both columns need variation for a correlation test.")
                else:
                    corr, p_value = stats.pearsonr(sample[col1], sample[col2])
                    c1, c2 = st.columns(2)
                    c1.metric("Pearson r", round(float(corr), 4))
                    c2.metric("P-value", round(float(p_value), 6))

        elif analysis_type == "Two-group t-test":
            if not numeric_cols or not categorical_cols:
                st.info("You need at least one numeric and one categorical column.")
            else:
                num_col = st.selectbox("Numeric column", numeric_cols, key="ttest_num")
                group_col = st.selectbox("Group column", categorical_cols, key="ttest_group")
                groups = sorted(analysis_df[group_col].dropna().astype(str).unique().tolist())

                if len(groups) < 2:
                    st.info("The selected group column needs at least two groups.")
                else:
                    group_a = st.selectbox("Group A", groups, key="ttest_group_a")
                    group_b = st.selectbox(
                        "Group B",
                        [group for group in groups if group != group_a],
                        key="ttest_group_b",
                    )
                    sample_a = analysis_df.loc[analysis_df[group_col].astype(str) == str(group_a), num_col].dropna()
                    sample_b = analysis_df.loc[analysis_df[group_col].astype(str) == str(group_b), num_col].dropna()

                    if len(sample_a) < 2 or len(sample_b) < 2:
                        st.warning("Each group needs at least two numeric observations.")
                    else:
                        t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Group A Mean", round(float(sample_a.mean()), 4))
                        c2.metric("Group B Mean", round(float(sample_b.mean()), 4))
                        c3.metric("T-stat", round(float(t_stat), 4))
                        c4.metric("P-value", round(float(p_value), 6))

        else:
            if not numeric_cols or not categorical_cols:
                st.info("You need at least one numeric and one categorical column.")
            else:
                num_col = st.selectbox("Numeric column", numeric_cols, key="anova_num")
                group_col = st.selectbox("Group column", categorical_cols, key="anova_group")
                grouped = []
                labels = []
                for label, values in analysis_df.groupby(group_col)[num_col]:
                    clean_values = values.dropna()
                    if len(clean_values) >= 2:
                        labels.append(str(label))
                        grouped.append(clean_values)

                if len(grouped) < 2:
                    st.warning("At least two groups with two or more observations are required.")
                else:
                    f_stat, p_value = stats.f_oneway(*grouped)
                    c1, c2 = st.columns(2)
                    c1.metric("F-stat", round(float(f_stat), 4))
                    c2.metric("P-value", round(float(p_value), 6))
                    st.write("Groups included:", ", ".join(labels))

elif page == "AI Insights":
    st.subheader("Generate Narrative Insights")
    model_name = st.text_input("Ollama model", value="llama3")
    scope = st.radio("Data scope", ["Filtered dataset", "Full working dataset"], horizontal=True)
    preset = st.selectbox(
        "Insight type",
        [
            "Executive summary",
            "Data quality review",
            "Business opportunities",
            "Suggested charts and next questions",
        ],
    )
    custom_focus = st.text_area("Optional focus", placeholder="Example: highlight risks, seasonality, outliers, and business recommendations.")

    if st.button("Generate Insights"):
        target_df = analysis_df if scope == "Filtered dataset" else working_df
        prompt = (
            "You are a senior data analyst. Review the dataset summary below and return concise markdown "
            "with sections for key findings, risks, anomalies, and recommendations.\n\n"
            f"Requested focus: {preset}. {custom_focus}\n\n"
            f"{build_ai_context(target_df)}\n\nColumn Names:\n{list(target_df.columns)}"
        )
        with st.spinner("Asking Ollama for insights..."):
            response, error = run_ollama(prompt, model_name)
        if error:
            st.error(f"Could not generate AI insights: {error}")
        else:
            st.session_state.latest_ai_insight = response
            st.markdown(response)

    if st.session_state.get("latest_ai_insight"):
        st.download_button(
            "Download insights as text",
            data=st.session_state.latest_ai_insight,
            file_name="ai_insights.txt",
            mime="text/plain",
        )

elif page == "Chat":
    st.subheader("Ask Questions About the Data")
    st.caption("The app tries a fast local answer first, then falls back to Ollama for open-ended questions.")
    model_name = st.text_input("Ollama model for fallback chat", value="llama3", key="chat_model")
    explain_local_answers = st.checkbox("Ask Ollama to explain direct answers", value=False)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

    for entry in st.session_state.get("chat_history", []):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("table") is not None:
                st.dataframe(display_safe(pd.DataFrame(entry["table"])), width="stretch")

    user_question = st.chat_input("Ask a question about your dataset")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        result, table = answer_question(analysis_df, user_question)
        assistant_text = ""
        table_records = None

        if result:
            assistant_text = result
            if explain_local_answers:
                prompt = (
                    "Explain the following data answer in simple business language and suggest one follow-up question.\n\n"
                    f"Answer: {result}\n\nDataset context:\n{build_ai_context(analysis_df)}"
                )
                explanation, error = run_ollama(prompt, model_name)
                if not error and explanation:
                    assistant_text = f"{result}\n\n{explanation}"
            if table is not None:
                table_records = table.to_dict("records")
        else:
            prompt = (
                "Answer the user's question using the dataset context below. If the answer is uncertain, say what extra "
                "data would help. Keep the answer concise.\n\n"
                f"Dataset context:\n{build_ai_context(analysis_df)}\n\n"
                f"User question: {user_question}"
            )
            answer, error = run_ollama(prompt, model_name)
            assistant_text = answer if answer else f"I could not answer that with Ollama: {error}"

        with st.chat_message("assistant"):
            st.markdown(assistant_text)
            if table_records is not None:
                st.dataframe(display_safe(pd.DataFrame(table_records)), width="stretch")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_text, "table": table_records}
        )

elif page == "Report":
    st.subheader("Export a PDF Report")
    report_scope = st.radio("Report scope", ["Filtered dataset", "Full working dataset"], horizontal=True)
    include_ai = st.checkbox("Include AI narrative", value=True)
    report_model = st.text_input("Ollama model for report", value="llama3", key="report_model")

    if st.button("Generate PDF Report"):
        target_df = analysis_df if report_scope == "Filtered dataset" else working_df
        ai_text = ""

        if include_ai:
            prompt = (
                "Create a compact report narrative with sections for overview, key patterns, risks, and recommendations.\n\n"
                f"{build_ai_context(target_df)}"
            )
            with st.spinner("Generating report narrative..."):
                ai_text, error = run_ollama(prompt, report_model)
            if error:
                st.warning(f"AI narrative was skipped: {error}")
                ai_text = ""

        try:
            st.session_state.report_bytes = build_pdf_report(target_df, ai_text, report_scope)
            st.session_state.report_preview = ai_text or "Report generated without an AI narrative."
            st.success("PDF report generated.")
        except Exception as exc:
            st.error(f"Could not build PDF report: {exc}")

    if st.session_state.get("report_bytes"):
        st.download_button(
            "Download PDF Report",
            data=st.session_state.report_bytes,
            file_name="data_analysis_report.pdf",
            mime="application/pdf",
        )
        st.text_area("Latest report narrative", value=st.session_state.get("report_preview", ""), height=220)
