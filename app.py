import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score
import io
import xlsxwriter
from datetime import datetime
from PIL import Image
import base64

# Setup with enhanced UI
st.set_page_config(page_title="Energy Analytics Dashboard", layout="wide", page_icon="📊")
st.title("🔋 Smart Energy Analytics Dashboard")

# File Upload with enhanced UI
uploaded_file = st.file_uploader("📤 Upload your dataset (CSV/XLSX)", type=["csv", "xlsx"])

# Keywords to detect energy dataset
ENERGY_KEYWORDS = ['energy', 'consumption', 'production', 'electricity', 'power', 'fuel', 'renewable', 'kwh', 'megawatt', 'grid']

def is_energy_dataset(df):
    return any(any(kw in col.lower() for kw in ENERGY_KEYWORDS) for col in df.columns)

def create_excel_dashboard(df, dataset_type, target_col, problem_type=None, model_results=None, feature_importances=None):
    """Create an Excel dashboard with actual visualizations"""
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    
    # Add a worksheet for the dashboard
    worksheet = workbook.add_worksheet('Dashboard')
    worksheet.set_landscape()
    
    # Formats
    header_format = workbook.add_format({
        'bold': True, 'font_size': 16, 'align': 'center', 'valign': 'vcenter',
        'border': 1, 'bg_color': '#2a3f5f', 'font_color': 'white'
    })
    title_format = workbook.add_format({
        'bold': True, 'font_size': 14, 'align': 'left', 'valign': 'vcenter'
    })
    metric_format = workbook.add_format({
        'bold': True, 'font_size': 12, 'align': 'left', 'valign': 'vcenter',
        'border': 1, 'bg_color': '#f8f9fa'
    })
    
    # Write header
    worksheet.merge_range('A1:H1', f"{dataset_type} Analytics Dashboard", header_format)
    
    # Dataset summary
    worksheet.write('A3', 'Dataset Summary', title_format)
    worksheet.write('A4', 'Total Records', metric_format)
    worksheet.write('B4', len(df))
    worksheet.write('A5', 'Total Features', metric_format)
    worksheet.write('B5', len(df.columns))
    worksheet.write('A6', 'Numeric Features', metric_format)
    worksheet.write('B6', len(df.select_dtypes(include=np.number).columns))
    worksheet.write('A7', 'Categorical Features', metric_format)
    worksheet.write('B7', len(df.select_dtypes(include='object').columns))
    
    # Create a sheet for raw data
    data_sheet = workbook.add_worksheet('Raw Data')
    data_sheet.write_row(0, 0, df.columns)
    for row_num, row_data in enumerate(df.values, 1):
        data_sheet.write_row(row_num, 0, row_data)
    
    # Initialize variables that might be used in summary
    preds = None
    energy_col = None
    date_cols = []
    energy_cols = []
    
    # Create visualizations based on dataset type
    if dataset_type == "Energy":
        date_cols = [col for col in df.columns if "year" in col.lower() or "date" in col.lower() or "time" in col.lower()]
        energy_cols = [col for col in df.select_dtypes(include=np.number).columns if any(kw in col.lower() for kw in ENERGY_KEYWORDS)]
        
        if date_cols and energy_cols:
            time_col = date_cols[0]
            energy_col = energy_cols[0]  # Take first energy column
            
            # Time series chart in Excel
            chart_sheet = workbook.add_worksheet('Time Series')
            chart_sheet.write_row(0, 0, [time_col, energy_col])
            
            time_data = df[[time_col, energy_col]].sort_values(time_col)
            for row_num, (_, row) in enumerate(time_data.iterrows(), 1):
                chart_sheet.write_row(row_num, 0, row)
            
            # Create a chart object only if we have data
            if len(time_data) > 0:
                try:
                    chart = workbook.add_chart({'type': 'line'})
                    chart.add_series({
                        'name': energy_col,
                        'categories': f"='Time Series'!$A$2:$A${len(time_data)+1}",
                        'values': f"='Time Series'!$B$2:$B${len(time_data)+1}",
                        'line': {'color': '#4CAF50', 'width': 2}
                    })
                    chart.set_title({'name': f'{energy_col} Over Time'})
                    chart.set_x_axis({'name': time_col})
                    chart.set_y_axis({'name': energy_col})
                    chart.set_legend({'position': 'none'})
                    chart_sheet.insert_chart('D2', chart)
                except Exception as e:
                    chart_sheet.write(0, 3, f"Could not create time series chart: {str(e)}")
            
            # Add forecast data
            try:
                lr = LinearRegression()
                lr.fit(df[[time_col]], df[energy_col])
                future_years = [df[time_col].max() + x for x in [5, 10, 15]]
                preds = lr.predict(np.array(future_years).reshape(-1, 1))
                
                forecast_sheet = workbook.add_worksheet('Forecast')
                forecast_sheet.write_row(0, 0, ['Year', 'Forecast'])
                for i, (year, val) in enumerate(zip(future_years, preds), 1):
                    forecast_sheet.write_row(i, 0, [year, val])
                
                # Forecast chart only if we have predictions
                if preds is not None and len(preds) > 0:
                    try:
                        forecast_chart = workbook.add_chart({'type': 'line'})
                        forecast_chart.add_series({
                            'name': 'Forecast',
                            'categories': f"='Forecast'!$A$2:$A${len(future_years)+1}",
                            'values': f"='Forecast'!$B$2:$B${len(future_years)+1}",
                            'line': {'color': '#FF5722', 'width': 2, 'dash_type': 'dash'}
                        })
                        forecast_chart.set_title({'name': f'{energy_col} Forecast'})
                        forecast_chart.set_x_axis({'name': 'Year'})
                        forecast_chart.set_y_axis({'name': energy_col})
                        forecast_sheet.insert_chart('D2', forecast_chart)
                    except Exception as e:
                        forecast_sheet.write(0, 3, f"Could not create forecast chart: {str(e)}")
            except Exception as e:
                if 'forecast_sheet' in locals():
                    forecast_sheet.write(0, 0, f"Forecast failed: {str(e)}")
            
            # Correlation matrix only if we have multiple energy columns
            if len(energy_cols) > 1:
                corr_sheet = workbook.add_worksheet('Correlations')
                try:
                    corr_matrix = df[energy_cols].corr()
                    
                    # Only proceed if we have a valid correlation matrix
                    if not corr_matrix.empty and len(corr_matrix) > 1:
                        # Write correlation matrix
                        corr_sheet.write_row(0, 1, corr_matrix.columns)
                        for i, col in enumerate(corr_matrix.columns, 1):
                            corr_sheet.write_row(i, 1, corr_matrix[col].values)
                            corr_sheet.write(i, 0, col)
                        
                        # Create and configure the heatmap chart
                        try:
                            corr_chart = workbook.add_chart({'type': 'heatmap'})
                            corr_chart.add_series({
                                'name': 'Correlation',
                                'categories': f"='Correlations'!$B$1:${chr(65+len(corr_matrix.columns))}$1",
                                'values': f"='Correlations'!$B$2:${chr(65+len(corr_matrix.columns))}${len(corr_matrix)+1}",
                                'data_labels': {'value': True, 'font': {'color': 'black'}},
                                'gradient': {'colors': ['#FF0000', '#FFFFFF', '#0000FF']}
                            })
                            corr_chart.set_title({'name': 'Energy Metrics Correlation'})
                            corr_sheet.insert_chart('D2', corr_chart)
                        except Exception as e:
                            corr_sheet.write(0, 3, f"Could not create correlation chart: {str(e)}")
                    else:
                        corr_sheet.write(0, 0, "Not enough data to generate correlation matrix")
                except Exception as e:
                    corr_sheet.write(0, 0, f"Correlation calculation failed: {str(e)}")
    
    # Add model results if available
    if problem_type and model_results is not None and not model_results.empty:
        model_sheet = workbook.add_worksheet('Model Results')
        model_sheet.write_row(0, 0, model_results.columns)
        
        for i, row in enumerate(model_results.values, 1):
            model_sheet.write_row(i, 0, row)
        
        # Create a chart for model comparison
        try:
            model_chart = workbook.add_chart({'type': 'column'})
            
            if problem_type == 'classification':
                for col_num in range(1, len(model_results.columns)):
                    model_chart.add_series({
                        'name': model_results.columns[col_num],
                        'categories': f"='Model Results'!$A$2:$A${len(model_results)+1}",
                        'values': f"='Model Results'!${chr(65+col_num)}$2:${chr(65+col_num)}${len(model_results)+1}",
                        'data_labels': {'value': True}
                    })
            else:
                model_chart.add_series({
                    'name': 'R² Score',
                    'categories': f"='Model Results'!$A$2:$A${len(model_results)+1}",
                    'values': f"='Model Results'!$C$2:$C${len(model_results)+1}",
                    'data_labels': {'value': True}
                })
            
            model_chart.set_title({'name': 'Model Performance Comparison'})
            model_chart.set_x_axis({'name': 'Model'})
            model_sheet.insert_chart('D2', model_chart)
        except Exception as e:
            model_sheet.write(0, len(model_results.columns)+1, f"Could not create model chart: {str(e)}")
    
    # Add feature importances if available
    if feature_importances and any(fi for fi in feature_importances.values() if fi is not None):
        fi_sheet = workbook.add_worksheet('Feature Importance')
        selected_model = next((k for k in feature_importances.keys() if feature_importances[k] is not None), None)
        
        if selected_model:
            try:
                fi_data = pd.DataFrame({
                    'Feature': feature_importances[selected_model]['features'],
                    'Importance': feature_importances[selected_model]['importance']
                }).sort_values('Importance', ascending=False)
                
                fi_sheet.write_row(0, 0, ['Feature', 'Importance'])
                for i, (_, row) in enumerate(fi_data.iterrows(), 1):
                    fi_sheet.write_row(i, 0, row)
                
                # Feature importance chart only if we have data
                if not fi_data.empty:
                    try:
                        fi_chart = workbook.add_chart({'type': 'bar'})
                        fi_chart.add_series({
                            'name': 'Importance',
                            'categories': f"='Feature Importance'!$A$2:$A${len(fi_data)+1}",
                            'values': f"='Feature Importance'!$B$2:$B${len(fi_data)+1}",
                            'data_labels': {'value': True}
                        })
                        fi_chart.set_title({'name': f'Feature Importance - {selected_model}'})
                        fi_chart.set_x_axis({'name': 'Importance'})
                        fi_chart.set_y_axis({'name': 'Feature'})
                        fi_sheet.insert_chart('D2', fi_chart)
                    except Exception as e:
                        fi_sheet.write(0, 3, f"Could not create feature importance chart: {str(e)}")
            except Exception as e:
                fi_sheet.write(0, 0, f"Feature importance processing failed: {str(e)}")
    
    # Add a summary sheet
    summary_sheet = workbook.add_worksheet('Summary')
    summary_sheet.write(0, 0, 'Key Insights', title_format)
    summary_sheet.write(1, 0, 'Dataset Type:')
    summary_sheet.write(1, 1, dataset_type)
    
    if dataset_type == "Energy" and date_cols and energy_cols and preds is not None:
        summary_sheet.write(2, 0, 'Primary Energy Metric:')
        summary_sheet.write(2, 1, energy_col)
        summary_sheet.write(3, 0, f'Forecast for {energy_col} in 15 years:')
        summary_sheet.write(3, 1, preds[-1][0] if isinstance(preds[-1], np.ndarray) else preds[-1])
    
    if problem_type and model_results is not None and not model_results.empty:
        try:
            if problem_type == 'classification':
                best_idx = model_results.iloc[:, 1].idxmax()
            else:
                best_idx = model_results.iloc[:, 2].idxmax()
            best_model = model_results.iloc[best_idx]
            
            summary_sheet.write(4, 0, 'Best Model:')
            summary_sheet.write(4, 1, best_model[0])
            summary_sheet.write(5, 0, 'Best Score:')
            summary_sheet.write(5, 1, best_model[1] if problem_type == 'classification' else best_model[2])
        except Exception as e:
            summary_sheet.write(4, 0, f"Could not determine best model: {str(e)}")
    
    workbook.close()
    return output.getvalue()

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    try:
        df_raw = pd.read_csv(uploaded_file) if ext == "csv" else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()
    
    # Data processing
    df = df_raw.dropna().copy()
    dataset_type = "Energy" if is_energy_dataset(df) else "Generic"
    
    # Display in Streamlit
    st.success(f"✅ Successfully uploaded {uploaded_file.name} ({dataset_type} Dataset)")
    
    with st.expander("📋 Dataset Preview", expanded=True):
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", df.shape[0])
        col2.metric("Total Features", df.shape[1])
        col3.metric("Missing Values", df_raw.isnull().sum().sum(), delta=f"{round(df_raw.isnull().sum().sum()/df_raw.size*100,2)}%")
    
    # Target selection
    target_col = st.selectbox("Select Target Variable", df.columns, index=len(df.columns)-1)
    
    # Enhanced visualizations in Streamlit
    if dataset_type == "Energy":
        date_cols = [col for col in df.columns if "year" in col.lower() or "date" in col.lower() or "time" in col.lower()]
        energy_cols = [col for col in df.select_dtypes(include=np.number).columns if any(kw in col.lower() for kw in ENERGY_KEYWORDS)]
        
        if date_cols and energy_cols:
            time_col = date_cols[0]
            selected_energy_col = st.selectbox("Select Energy Metric", energy_cols)
            
            # Time series plot
            fig = px.line(df, x=time_col, y=selected_energy_col, 
                         title=f"Time Series of {selected_energy_col}",
                         template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecasting
            st.subheader("🔮 Energy Forecasting")
            forecast_years = st.slider("Select forecast period (years)", 1, 30, 10)
            
            lr = LinearRegression()
            lr.fit(df[[time_col]], df[selected_energy_col])
            future_years = np.arange(df[time_col].min(), df[time_col].max() + forecast_years + 1)
            preds = lr.predict(future_years.reshape(-1, 1))
            
            forecast_df = pd.DataFrame({
                time_col: future_years,
                selected_energy_col: preds,
                'Type': ['Historical' if year <= df[time_col].max() else 'Forecast' for year in future_years]
            })
            
            fig = px.line(forecast_df, x=time_col, y=selected_energy_col, color='Type',
                         title=f"{selected_energy_col} Forecast",
                         template="plotly_white")
            fig.add_vline(x=df[time_col].max(), line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("🔥 Energy Metrics Correlation")
            corr_matrix = df[energy_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}"
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    # Model evaluation
    st.subheader("🤖 Machine Learning Analysis")
    df_ml = df.select_dtypes(include=np.number)
    
    if target_col not in df_ml.columns:
        st.warning("⚠️ Target column not in numeric data. Skipping ML analysis.")
    else:
        X = df_ml.drop(columns=[target_col])
        y = df_ml[target_col]
        
        # Detect problem type
        unique_values = len(y.unique())
        if y.dtype == 'object':
            problem_type = 'classification'
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.dtype in ['int64', 'float64'] and unique_values < 15 and unique_values / len(y) < 0.05:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
        
        st.markdown(f"**Problem Type:** {problem_type.capitalize()}")
        
        # Model selection and training
        models = {
            "classification": {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            },
            "regression": {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }
        }[problem_type]
        
        # Data scaling and splitting
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Model training and evaluation
        results = []
        feature_importances = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if problem_type == 'classification':
                results.append([
                    name,
                    accuracy_score(y_test, preds),
                    precision_score(y_test, preds, average="macro", zero_division=0),
                    recall_score(y_test, preds, average="macro", zero_division=0),
                    f1_score(y_test, preds, average="macro", zero_division=0)
                ])
            else:
                results.append([
                    name,
                    mean_squared_error(y_test, preds),
                    r2_score(y_test, preds),
                    np.sqrt(mean_squared_error(y_test, preds))
                ])
            
            # Store feature importances if available
            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = {
                    'features': X.columns,
                    'importance': model.feature_importances_
                }
            elif hasattr(model, 'coef_'):
                feature_importances[name] = {
                    'features': X.columns,
                    'importance': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                }
        
        # Display results
        if problem_type == "classification":
            result_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
            fig = px.bar(result_df.melt(id_vars="Model"), 
                        x="Model", y="value", color="variable",
                        barmode="group", 
                        title="Classification Model Performance")
            fig.update_layout(yaxis_range=[0, 1])
        else:
            result_df = pd.DataFrame(results, columns=["Model", "MSE", "R²", "RMSE"])
            fig = px.bar(result_df, x="Model", y="R²", 
                        title="Regression Model Performance (R²)")
        
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(result_df.style.highlight_max(axis=0, color='lightgreen'))
        
        # Feature importance visualization
        if feature_importances:
            st.subheader("📊 Feature Importance")
            selected_model = st.selectbox("Select model", list(feature_importances.keys()))
            
            fi_df = pd.DataFrame({
                'Feature': feature_importances[selected_model]['features'],
                'Importance': feature_importances[selected_model]['importance']
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        title=f"Feature Importance - {selected_model}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Generate and download Excel report
    st.subheader("📥 Download Full Report")
    if st.button("Generate Excel Dashboard"):
        with st.spinner("Creating comprehensive Excel report..."):
            excel_data = create_excel_dashboard(
                df=df,
                dataset_type=dataset_type,
                target_col=target_col,
                problem_type=problem_type if 'problem_type' in locals() else None,
                model_results=result_df if 'result_df' in locals() else None,
                feature_importances=feature_importances if 'feature_importances' in locals() else None
            )
            
            st.success("✅ Excel report generated successfully!")
            st.download_button(
                label="Download Excel Dashboard",
                data=excel_data,
                file_name=f"energy_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )